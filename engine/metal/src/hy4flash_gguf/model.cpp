// engine/metal/src/hy4flash_gguf/model.cpp - see model.h.
#include "model.h"

#include "ref_quant.h"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/hy4flash_gguf/geometry.hpp"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

namespace strata::hy4_flash {

namespace {

constexpr const char* kExpertRoles[3] = {"gate", "up", "down"};

/// What `ExpertStream::load_dense` will put in the buffer, from the header alone.  Its packing is the contract:
/// one tensor after another, each start rounded up to 256 B, and the whole buffer to a 16 KiB page.  Knowing it
/// here is what lets `--dry-run` state the resident bytes without reading 8 GB (and what the loader test checks
/// against the GGUF header probe).
uint64_t plan_dense(const Model& m, int32_t* n_tensors) {
    uint64_t total = 0;
    int32_t n = 0;
    for (const auto& kv : m.tensors.tensors) {
        const gguf::Tensor& t = kv.second;
        if (t.name.find("_exps") != std::string::npos) continue;
        if (t.name == "per_layer_token_embd.weight") continue;
        if (std::find(m.skipped.begin(), m.skipped.end(), t.name) != m.skipped.end()) continue;
        total += (t.bytes + 255) / 256 * 256;
        ++n;
    }
    *n_tensors = n;
    return total;
}

}  // namespace

std::string open(Model& m, const std::string& shard1, int32_t max_context, bool with_mtp) {
    m.shard1_path = shard1;

    // ---- the geometry, from BOTH shards' metadata.  The Engram table's TensorInfo comes from whichever shard
    // holds it, because read_geometry checks that its type can express a 96-value row and that the row count
    // matches the per-head bucket counts - the check that catches a file whose metadata and payload disagree.
    try {
        const GgufModel model = GgufModel::open(shard1);
        const GgufFile& meta = model.meta();
        const std::string guard = check_architecture(meta);
        if (!guard.empty()) return guard;
        std::size_t shard = 0;
        const TensorInfo* engram = model.find("per_layer_token_embd.weight", &shard);
        if (engram == nullptr) return "per_layer_token_embd.weight is in none of the shards";
        m.engram_path = model.shard(shard).path();
        // Where the table really starts in that file.  `TensorInfo::offset` is relative to its OWN shard's data
        // start, which is what `PleReader` (and so `EngramRows`) needs as an absolute file offset.
        m.engram_offset = model.shard(shard).data_start() + engram->offset;
        m.engram_rows = engram->shape.size() > 1 ? engram->shape[1] : 0;
        const std::string err = hy4_flash::read_geometry(meta, m.geo, max_context, engram);
        if (!err.empty()) return err;
        m.engram_row_bytes = m.geo.engram_row_bytes;
    } catch (const std::exception& e) {
        return e.what();
    }

    // ---- shard 1's tensor offsets.  Shard 2 is never opened with this reader (model.h).
    std::string err;
    if (!m.tensors.open(shard1, err)) return err;

    // ---- the skip list: the MTP block's tensors.  By block number from the geometry, not by the literal
    // "blk.42", so the MTP switch lives in one place.
    m.skipped.clear();
    for (int32_t b = m.geo.trunk_blocks; b < m.geo.block_count && !with_mtp; ++b) {
        const std::string prefix = "blk." + std::to_string(b) + ".";
        for (const auto& kv : m.tensors.tensors)
            if (kv.first.compare(0, prefix.size(), prefix) == 0) m.skipped.push_back(kv.first);
    }
    // ---- read from the file instead of resident (model.h on_file): offsets recorded NOW, before load_dense zeroes
    // a skipped tensor's offset
    m.on_file.clear();
    auto to_file = [&](const gguf::Tensor& t) {
        m.on_file[t.name] = t.offset;
        if (std::find(m.skipped.begin(), m.skipped.end(), t.name) == m.skipped.end()) m.skipped.push_back(t.name);
    };
    const char* er = std::getenv("HY4_FLASH_EMBD_RESIDENT");
    if (er == nullptr || std::atoi(er) == 0)
        if (const gguf::Tensor* te = m.tensors.find("token_embd.weight")) to_file(*te);
    const char* mq = std::getenv("HY4_FLASH_MTP_QUANT");
    if (with_mtp && (mq == nullptr || std::strcmp(mq, "bf16") != 0))
        for (int32_t b = m.geo.trunk_blocks; b < m.geo.block_count; ++b) {
            const std::string prefix = "blk." + std::to_string(b) + ".";
            for (const auto& kv : m.tensors.tensors)
                if (kv.first.compare(0, prefix.size(), prefix) == 0 && kv.second.type == 30 &&
                    kv.second.shape.size() >= 2 && kv.first.find("_exps") == std::string::npos)
                    to_file(kv.second);
        }

    // ---- the expert table, built BEFORE load_dense (model.h: it rewrites offsets).  One entry per ordinal,
    // including the MTP block's, which decode never reads but whose slices are wanted in the report.
    m.experts.assign((std::size_t) m.geo.n_expert_ordinals(), ExpertSlices{});
    for (const hy4_flash::LayerSpec& s : m.geo.layers) {
        if (s.expert_ordinal < 0) continue;
        ExpertSlices e;
        e.ordinal = s.expert_ordinal;
        e.block_id = s.block_id;
        for (int r = 0; r < 3; ++r) {
            const std::string name =
                "blk." + std::to_string(s.block_id) + ".ffn_" + kExpertRoles[r] + "_exps.weight";
            const gguf::Tensor* t = m.tensors.find(name);
            if (t == nullptr) return name + " is not in " + shard1;
            // GGUF order: [in, out, expert], so the expert count is the last dimension and one expert's bytes
            // are the tensor's divided by it.  A 2-D expert tensor would mean a model of another shape.
            if (t->shape.size() != 3 || t->shape[2] == 0)
                return name + " is not [in, out, expert]; this model stores one matrix per expert";
            if (t->shape[2] != (uint64_t) m.geo.n_experts)
                return name + " holds " + std::to_string(t->shape[2]) + " experts, but expert_count is " +
                       std::to_string(m.geo.n_experts);
            e.off[r] = t->offset;
            e.stride[r] = t->bytes / t->shape[2];
            e.type[r] = t->type;
            e.n_experts = (int32_t) t->shape[2];
        }
        m.experts[(std::size_t) s.expert_ordinal] = e;
    }
    for (std::size_t i = 0; i < m.experts.size(); ++i)
        if (m.experts[i].ordinal < 0)
            return "expert ordinal " + std::to_string(i) + " has no block; the geometry and the tensors disagree";

    m.dense_plan_bytes = plan_dense(m, &m.dense_plan_tensors);
    return {};
}

bool Model::tensor_bytes(const std::string& name, uint64_t at, uint64_t len, void* dst) const {
    const auto it = on_file.find(name);
    if (it != on_file.end()) return stream.read(dst, len, it->second + at);
    const gguf::Tensor* t = find(name);
    if (t == nullptr || dense.ptr == nullptr) return false;
    std::memcpy(dst, (const uint8_t*) dense.ptr + t->offset + at, (size_t) len);
    return true;
}

std::string load_dense(Model& m, mtl::Context& cx) {
    std::string err;
    if (!m.stream.open_file(m.shard1_path, err)) return err;
    const auto t0 = std::chrono::steady_clock::now();
    if (!m.stream.load_dense(m.tensors, cx, m.skipped, m.dense, err)) return err;
    m.load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    m.dense_bytes = m.dense_plan_bytes;
    if (m.dense.bytes < m.dense_bytes)
        return "the dense buffer is " + std::to_string(m.dense.bytes) + " B, less than the " +
               std::to_string(m.dense_bytes) + " B the header planned";
    return {};
}

std::string describe(const Model& m) {
    std::string out = hy4_flash::describe(m.geo);
    char b[1024];
    std::snprintf(b, sizeof b,
                  "\n  shard 1        %s\n"
                  "  engram shard   %s\n"
                  "  dense          %d tensors, %" PRIu64 " B = %.4f GiB resident (%zu of the MTP block left out)\n"
                  "  experts        %zu ordinals, %d per block",
                  m.shard1_path.c_str(), m.engram_path.c_str(), m.dense_plan_tensors, m.dense_plan_bytes,
                  (double) m.dense_plan_bytes / (double) (1ull << 30), m.skipped.size(), m.experts.size(),
                  m.geo.n_experts);
    out += b;
    // The expert types per ordinal, as runs: 22 blocks store gate/up as Q3_K and 19 as Q4_K, and which is
    // which is the thing a kernel dispatch has to get right.
    std::size_t i = 0;
    while (i < m.experts.size()) {
        std::size_t j = i;
        while (j + 1 < m.experts.size() && m.experts[j + 1].type[kGate] == m.experts[i].type[kGate] &&
               m.experts[j + 1].type[kUp] == m.experts[i].type[kUp] &&
               m.experts[j + 1].type[kDown] == m.experts[i].type[kDown])
            ++j;
        std::snprintf(b, sizeof b, "\n                 ordinal %2zu-%2zu (block %d-%d): gate %s, up %s, down %s", i, j,
                      m.experts[i].block_id, m.experts[j].block_id, ref::type_name(m.experts[i].type[kGate]),
                      ref::type_name(m.experts[i].type[kUp]), ref::type_name(m.experts[i].type[kDown]));
        out += b;
        i = j + 1;
    }
    return out;
}

}  // namespace strata::hy4_flash
