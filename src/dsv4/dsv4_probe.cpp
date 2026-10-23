// src/dsv4/dsv4_probe.cpp - the dsv4 MVP weight-mapping probe (K8b): opens the GGUF (metadata only - the
// header parse never touches tensor payloads) and the pack directory, and proves that EVERY tensor of the
// layout table (include/strata/dsv4/layout.hpp, frozen from check_architecture_deepseek4) resolves to one
// of the three sources the engine reads weights from:
//
//   1. native    - served from the GGUF itself (an index.txt row with the native marker: kind 0, field 9 = 8;
//                  the engine's NativeDense path reads Q8_0 / I32 straight out of the file)
//   2. dense.bin - an index.txt row with kind 2/3/4/5 (F32 / F32-narrowed-to-F16 / BF16 / F16) carrying the
//                  stored offset and byte count into dense.bin
//   3. experts.bin - the NVFP4 expert mergers (ffn_gate/up/down_exps.weight), one blob per layer per the
//                  native_experts.txt table (43 x 256 experts; scale vectors live in dense.bin, not here)
//
// Host-only, no CUDA: GgufFile is header-only, so this builds with a bare
//   g++ -std=c++17 -I include src/dsv4/dsv4_probe.cpp -o dsv4-probe
//
// index.txt row syntax (tools/iq_pack.py, write_index / index_standalone), 19 fields:
//   [0] name  [1] file (0 = dense.bin)  [2] kind (0 native / 2 F32 / 3 F32->F16 / 4 BF16 / 5 F16)
//   [3] stored offset  [4] stored bytes  [5] arena offset  [6] destination bytes
//   [7] ne0  [8] ne1  [9] 8 for native rows, else 0  [10..11] native alignment fields / [11] = 1 for dense
//   [12..18] reserved zeros
// native_experts.txt v3 (one row per layer, tools/iq_pack.py): layer gu_type d_type offset blob_bytes
//   [gate_off up_off down_off [shard]] - the role offsets default to the packed gate/up/down order and are
//   omitted when they do; blob_bytes is ONE expert's bytes incl. the 16-byte NVFP4 tail, so a layer's blob
//   spans n_expert * blob_bytes.
//
// Exit 0 only at 100% coverage: every layout-table tensor resolved, every index.txt row consumed by the
// table, every native_experts.txt row covering its layer's three mergers, and every table tensor's shape
// agreeing with the GGUF directory.
#include "strata/dsv4/layout.hpp"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <sys/stat.h>

namespace {

struct IndexRow {
    std::string name;
    int kind = 0;             // field 2: 0 native, 2 F32, 3 F32->F16, 4 BF16, 5 F16
    uint64_t offset = 0;      // field 3: stored offset into dense.bin
    uint64_t bytes = 0;       // field 4: stored bytes
    uint64_t dst_bytes = 0;   // field 6
    int64_t ne0 = 0, ne1 = 0; // fields 7,8
    bool native = false;      // field 9 == 8
};

struct ExpertRow {
    int64_t layer = -1;
    uint32_t gu_type = 0, d_type = 0;
    uint64_t offset = 0;      // layer blob start in experts.bin
    uint64_t blob_bytes = 0;  // ONE expert's bytes (incl. tail); the blob spans n_expert * blob_bytes
};

uint64_t file_size(const std::string& path) {
    struct stat st {};
    if (stat(path.c_str(), &st) != 0) return 0;
    return (uint64_t) st.st_size;
}

// Parses pack/index.txt.  False with `err` on a malformed row.
bool read_index(const std::string& path, std::map<std::string, IndexRow>& rows, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    char line[1024];
    int64_t lineno = 0;
    while (std::fgets(line, sizeof line, f)) {
        ++lineno;
        if (line[0] == '#' || line[0] == '\n') continue;
        char name[512];
        long long f1, kind, off, bytes, f5, dst, ne0, ne1, f9, f10, f11;
        const int got = std::sscanf(line, "%511s %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld", name,
                                    &f1, &kind, &off, &bytes, &f5, &dst, &ne0, &ne1, &f9, &f10, &f11);
        if (got != 12) {
            err = path + ":" + std::to_string(lineno) + ": an index row has 19 fields, this one parses as " +
                  std::to_string(got);
            std::fclose(f);
            return false;
        }
        IndexRow r;
        r.name = name;
        r.kind = (int) kind;
        r.offset = (uint64_t) off;
        r.bytes = (uint64_t) bytes;
        r.dst_bytes = (uint64_t) dst;
        r.ne0 = ne0;
        r.ne1 = ne1;
        r.native = f9 == 8;
        if (!rows.emplace(r.name, r).second) {
            err = path + ": duplicate index row for " + r.name;
            std::fclose(f);
            return false;
        }
    }
    std::fclose(f);
    return true;
}

// Parses pack/native_experts.txt (v3: 5-9 fields; the role offsets/shard are omitted at defaults).
bool read_native_experts(const std::string& path, std::map<int64_t, ExpertRow>& rows, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    char line[512];
    int64_t lineno = 0;
    while (std::fgets(line, sizeof line, f)) {
        ++lineno;
        if (line[0] == '#' || line[0] == '\n') continue;
        ExpertRow r;
        long long layer = -1;
        const int got = std::sscanf(line, "%lld %u %u %llu %llu", &layer, &r.gu_type, &r.d_type,
                                    (unsigned long long*) &r.offset, (unsigned long long*) &r.blob_bytes);
        r.layer = layer;
        if (got < 5) {
            err = path + ":" + std::to_string(lineno) + ": a native-experts row has at least 5 fields";
            std::fclose(f);
            return false;
        }
        if (!rows.emplace(r.layer, r).second) {
            err = path + ": duplicate native-experts row for layer " + std::to_string(r.layer);
            std::fclose(f);
            return false;
        }
    }
    std::fclose(f);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::printf("usage: dsv4_probe <model.gguf> <pack-dir>\n");
        return 2;
    }
    const std::string gguf_path = argv[1], pack = argv[2];
    try {
        // ---- the GGUF: header + directory only (mmap is lazy; the payload pages are never touched)
        strata::GgufFile gguf(gguf_path);
        strata::dsv4::Dsv4Geometry g;
        std::string err;
        if (!strata::dsv4::dsv4_geometry_from_gguf(gguf, g, err)) {
            std::printf("ERROR: geometry/guard refused %s: %s\n", gguf_path.c_str(), err.c_str());
            return 1;
        }
        size_t gguf_checked = 0;
        if (!strata::dsv4::dsv4_check_gguf_layout(gguf, g, err, &gguf_checked)) {
            std::printf("ERROR: layout vs GGUF directory: %s\n", err.c_str());
            return 1;
        }
        std::printf("GGUF %s\n  %zu tensors, layout table verified against the directory: %zu entries\n",
                    gguf_path.c_str(), gguf.tensors().size(), gguf_checked);

        // ---- the pack
        std::map<std::string, IndexRow> index;
        if (!read_index(pack + "/index.txt", index, err)) {
            std::printf("ERROR: %s\n", err.c_str());
            return 1;
        }
        std::map<int64_t, ExpertRow> experts;
        if (!read_native_experts(pack + "/native_experts.txt", experts, err)) {
            std::printf("ERROR: %s\n", err.c_str());
            return 1;
        }
        const uint64_t dense_size = file_size(pack + "/dense.bin");
        const uint64_t experts_size = file_size(pack + "/experts.bin");
        if (!dense_size || !experts_size) {
            std::printf("ERROR: dense.bin (%" PRIu64 ") or experts.bin (%" PRIu64 ") missing/empty\n", dense_size,
                        experts_size);
            return 1;
        }
        std::printf("pack %s\n  index.txt %zu rows   native_experts.txt %zu rows   dense.bin %" PRIu64
                    " B   experts.bin %" PRIu64 " B\n",
                    pack.c_str(), index.size(), experts.size(), dense_size, experts_size);

        // ---- resolve every table tensor to one route
        struct Counts {
            size_t native = 0, dense = 0, experts = 0;
            size_t total() const { return native + dense + experts; }
        };
        std::map<strata::dsv4::Dsv4LayerKind, Counts> per_kind;
        Counts root_counts;
        std::vector<std::string> gaps;
        std::set<std::string> index_used;

        auto resolve = [&](const std::string& name, const strata::dsv4::Dsv4TensorWant& w, Counts& c) {
            const bool is_expert = name.size() > 12 && name.compare(name.size() - 12, 12, "_exps.weight") == 0;
            if (is_expert) {
                // route 3: experts.bin via native_experts.txt (blk.<L>.ffn_<role>_exps.weight)
                const size_t a = name.find('.'), b = name.find('.', a + 1);
                const int64_t layer = std::stoll(name.substr(a + 1, b - a - 1));
                const auto it = experts.find(layer);
                if (it == experts.end()) {
                    gaps.push_back(name + ": no native_experts.txt row for layer " + std::to_string(layer));
                    return;
                }
                const ExpertRow& e = it->second;
                if (e.gu_type != strata::dsv4::kGgufNVFP4 || e.d_type != strata::dsv4::kGgufNVFP4) {
                    gaps.push_back(name + ": native_experts.txt layer " + std::to_string(layer) + " types " +
                                   std::to_string(e.gu_type) + "/" + std::to_string(e.d_type) + ", not NVFP4 (40)");
                    return;
                }
                const uint64_t span = (uint64_t) g.n_expert * e.blob_bytes;
                if (e.offset > experts_size || span > experts_size - e.offset) {
                    gaps.push_back(name + ": layer " + std::to_string(layer) + " blob [" +
                                   std::to_string(e.offset) + ", +" + std::to_string(span) + ") escapes experts.bin (" +
                                   std::to_string(experts_size) + ")");
                    return;
                }
                ++c.experts;
                return;
            }
            const auto it = index.find(name);
            if (it == index.end()) {
                gaps.push_back(name + ": not in index.txt (and not an expert merger)");
                return;
            }
            index_used.insert(name);
            const IndexRow& r = it->second;
            if (r.native) {
                // route 1: served from the GGUF - the directory must hold it with the table's shape
                const strata::TensorInfo* t = gguf.find(name);
                if (!t) {
                    gaps.push_back(name + ": index.txt marks it native, but it is not in the GGUF");
                    return;
                }
                bool same = t->shape.size() == w.shape.size();
                for (size_t i = 0; same && i < w.shape.size(); ++i) same = t->shape[i] == (uint64_t) w.shape[i];
                if (!same) {
                    gaps.push_back(name + ": native row, but the GGUF shape disagrees with the layout table");
                    return;
                }
                ++c.native;
                return;
            }
            // route 2: dense.bin
            if (r.kind != 2 && r.kind != 3 && r.kind != 4 && r.kind != 5) {
                gaps.push_back(name + ": index kind " + std::to_string(r.kind) + " is not a dense.bin kind (2/3/4/5)");
                return;
            }
            if (r.offset > dense_size || r.bytes > dense_size - r.offset) {
                gaps.push_back(name + ": dense.bin slice [" + std::to_string(r.offset) + ", +" +
                               std::to_string(r.bytes) + ") escapes dense.bin (" + std::to_string(dense_size) + ")");
                return;
            }
            const int64_t want1 = w.shape.size() > 1 ? w.shape[1] : 0;
            if (r.ne0 != w.shape[0] || r.ne1 != want1) {
                gaps.push_back(name + ": index ne0/ne1 " + std::to_string(r.ne0) + "/" + std::to_string(r.ne1) +
                               ", the layout table wants " + std::to_string(w.shape[0]) + "/" +
                               std::to_string(want1));
                return;
            }
            ++c.dense;
        };

        for (const strata::dsv4::Dsv4TensorWant& w : strata::dsv4::dsv4_root_tensors(g))
            resolve(w.name, w, root_counts);
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const strata::dsv4::Dsv4LayerKind kind = strata::dsv4::dsv4_layer_kind(l);
            const std::string b = "blk." + std::to_string(l) + ".";
            for (const strata::dsv4::Dsv4TensorWant& w : strata::dsv4::dsv4_layer_tensors(g, l))
                resolve(b + w.name, w, per_kind[kind]);
        }

        // ---- the reverse direction: every index row must be a table tensor
        for (const auto& [name, r] : index)
            if (!index_used.count(name)) gaps.push_back(name + ": in index.txt but not in the dsv4 layout table");
        // every native_experts row must belong to a trunk layer
        for (const auto& [layer, e] : experts)
            if (layer < 0 || layer >= g.n_layers)
                gaps.push_back("native_experts.txt layer " + std::to_string(layer) + " is outside 0.." +
                               std::to_string(g.n_layers - 1));

        // ---- the report
        std::printf("\ncoverage by layer type (tensors: native / dense.bin / experts.bin / total):\n");
        size_t layers_of[4] = {};  // one slot per Dsv4LayerKind
        for (int64_t l = 0; l < g.n_layers; ++l) layers_of[(size_t) strata::dsv4::dsv4_layer_kind(l)]++;
        Counts all;
        for (const auto& [kind, c] : per_kind) {
            const size_t nl = layers_of[(size_t) kind];
            std::printf("  %-22s %2zu layers: %4zu / %4zu / %4zu / %4zu  (%zu per layer)\n",
                        strata::dsv4::dsv4_layer_kind_name(kind), nl, c.native, c.dense, c.experts, c.total(),
                        nl ? c.total() / nl : 0);
            all.native += c.native;
            all.dense += c.dense;
            all.experts += c.experts;
        }
        std::printf("  %-22s %2zu roots  : %4zu / %4zu / %4zu / %4zu\n", "root", (size_t) 6, root_counts.native,
                    root_counts.dense, root_counts.experts, root_counts.total());
        all.native += root_counts.native;
        all.dense += root_counts.dense;
        all.experts += root_counts.experts;
        const size_t table_total = all.total() + gaps.size();
        std::printf("\ntotal: %zu of %zu layout tensors resolved (native %zu, dense.bin %zu, experts.bin %zu); "
                    "%zu gap(s)\n",
                    all.total(), table_total, all.native, all.dense, all.experts, gaps.size());
        if (!gaps.empty()) {
            std::printf("\nGAPS:\n");
            for (const std::string& s : gaps) std::printf("  - %s\n", s.c_str());
            return 1;
        }
        std::printf("coverage: 100%%\n");
        return 0;
    } catch (const std::exception& e) {
        std::printf("ERROR: %s\n", e.what());
        return 1;
    }
}
