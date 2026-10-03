// tests/core/weights_carve_test.cpp - the layer-range carve in the dense-weight loader.
//
// THE CARVE HAS ONE JOB AND IT IS ARITHMETIC: a stage that runs layers [lo, hi) must hold only those
// layers' weights, and the arena it is given must be exactly as large as what the loader writes into it.
// Both halves are decided by `layer_out_of_range`, and if the two ever disagree the failure is not a
// crash - a too-small arena is caught, but a too-LARGE one silently costs the expert cache, and a
// MIS-COUNTED one puts a tensor at an offset another tensor also claims, which decodes to a plausible
// weight and produces plausible logits.  So this test checks the sizes against each other, over every
// range, and against a real pack when one is named.
//
// No GPU and no model: `pool_bytes` and `layer_bytes` are pure file parsing, which is the whole point of
// pricing a placement before committing VRAM to it.

#include "strata/core/weights.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct TempDirectory {
    fs::path path;

    TempDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("strata-weights-carve-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }

    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

struct Row {
    std::string name;
    uint64_t dst_bytes = 0;
};

/// One index.txt in the loader's own format: 19 space-separated fields, the same shape `pack_index.py`
/// writes.  Written whole so the same file could be handed to `load` - a carve test that fed the parser a
/// shortened row would be testing a format nothing else produces.
std::string row_line(const Row& r) {
    //      name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 cb cbi ge cbk ho codes scales off sfp act
    return r.name + " 0 0 0 0 0 " + std::to_string(r.dst_bytes) + " 2560 1 0 0 0 0 0 0 0 0 0 0";
}

uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

void write_index(const fs::path& dir, uint64_t align, const std::vector<Row>& rows) {
    uint64_t pool = 0;
    for (const Row& r : rows) pool += align_up(r.dst_bytes, align);
    std::ofstream out(dir / "index.txt", std::ios::trunc);
    require((bool) out, "could not write a synthetic index.txt");
    out << "# strata pack index v3 -- synthetic\n";
    out << "# align " << align << " pool " << pool << " tensors " << rows.size() << "\n";
    for (const Row& r : rows) out << row_line(r) << "\n";
    require((bool) out, "could not write a synthetic index.txt");
}

/// The invariant the carve rests on, checked over EVERY range against a pack (synthetic or real):
/// the loader's own compaction and the price model must agree, and the whole-model range must reproduce
/// the pack's declared pool exactly - which is what makes a full-range carve a no-op.
void check_ranges(const fs::path& pack, int64_t n_layers, const std::set<std::string>* skip,
                  const std::set<std::string>* keep) {
    std::string err;
    std::vector<uint64_t> per_layer;
    uint64_t globals = 0, keep_bytes = 0;
    require(strata::core::WeightTable::layer_bytes(pack.string(), n_layers, per_layer, globals, keep_bytes, err,
                                                   skip, keep),
            "layer_bytes: " + err);

    uint64_t priced_all = globals + keep_bytes;
    for (uint64_t b : per_layer) priced_all += b;

    uint64_t full = 0;
    require(strata::core::WeightTable::pool_bytes(pack.string(), full, err, skip, 0, n_layers, keep),
            "pool_bytes (full): " + err);
    require(full == priced_all,
            "the full range prices " + std::to_string(priced_all) + " B but compacts to " + std::to_string(full));

    // WITH NO FILTER, THE COMPACTION MUST REPRODUCE THE PACK'S OWN POOL.  This is the claim that makes a
    // full-range carve a no-op: every row keeps the size it was written with, so the arena is the same
    // arena and every tensor lands at the offset it always did.  Without `skip` the loader does not even
    // enter the re-placement loop, so this is the assertion that the range rule cannot cost anything when
    // it drops nothing.
    if (skip == nullptr) {
        std::ifstream in(pack / "index.txt");
        std::string line;
        uint64_t declared = 0;
        while (std::getline(in, line)) {
            if (line.rfind("# align", 0) != 0) continue;
            const size_t at = line.find("pool ");
            if (at != std::string::npos) declared = std::stoull(line.substr(at + 5));
            break;
        }
        require(declared != 0, "the pack has no '# align ... pool ...' header");
        require(priced_all == declared, "the full range prices " + std::to_string(priced_all) +
                                            " B but the pack declares a pool of " + std::to_string(declared));
    }

    for (int64_t lo = 0; lo <= n_layers; ++lo) {
        for (int64_t hi = lo; hi <= n_layers; ++hi) {
            uint64_t want = globals + keep_bytes;
            for (int64_t l = lo; l < hi; ++l) want += per_layer[(size_t) l];
            uint64_t got = 0;
            require(strata::core::WeightTable::pool_bytes(pack.string(), got, err, skip, lo, hi, keep),
                    "pool_bytes [" + std::to_string(lo) + "," + std::to_string(hi) + "): " + err);
            require(got == want, "[" + std::to_string(lo) + "," + std::to_string(hi) + "): the range prices " +
                                     std::to_string(want) + " B but compacts to " + std::to_string(got));
        }
    }

    // `hi < 0` is the disabled carve, and it must be the pack's own pool - the byte-identical no-op.
    uint64_t off = 0;
    require(strata::core::WeightTable::pool_bytes(pack.string(), off, err, skip, 0, -1, keep),
            "pool_bytes (disabled): " + err);
    require(off == priced_all, "the disabled carve is not the full range");
}

void check_name_parsing() {
    using strata::core::layer_out_of_range;
    using strata::core::tensor_layer;

    require(tensor_layer("blk.0.attn_qkv.weight") == 0, "blk.0 is layer 0");
    require(tensor_layer("blk.47.ffn_gate_inp.weight") == 47, "blk.47 is layer 47");
    require(tensor_layer("blk.1.ple_key.weight") == 1, "the PLE key is layer 1's");
    require(tensor_layer("blk.12.ssm_conv1d.weight") == 12, "blk.12 is layer 12");

    // A model-global name has no layer, and the range must never drop one: the head and the embedding are
    // needed by stages whose layer range excludes nothing - they are simply not the stage's layers.
    require(tensor_layer("token_embd.weight") == -1, "the embedding is not a layer's");
    require(tensor_layer("output.weight") == -1, "the head is not a layer's");
    require(tensor_layer("output_hc_down.weight") == -1, "the head's hyper-connections are not a layer's");
    require(!layer_out_of_range("output.weight", 10, 19), "the head survives any range");
    require(!layer_out_of_range("token_embd.weight", 10, 19), "the embedding survives any range");

    // Names that LOOK per-layer but are not: a carve that guessed would put them in the wrong stage.
    require(tensor_layer("blk..attn_qkv.weight") == -1, "an empty ordinal is not layer 0");
    require(tensor_layer("blk.x.attn_qkv.weight") == -1, "a non-numeric ordinal is not a layer");
    require(tensor_layer("blk.1") == -1, "a name with no suffix is not a tensor name");
    require(tensor_layer("blk1.attn_qkv.weight") == -1, "a missing dot is not a layer");
    require(tensor_layer("xblk.1.weight") == -1, "a prefix is not the blk. prefix");

    require(layer_out_of_range("blk.3.a", 10, 19), "layer 3 is outside [10,19)");
    require(!layer_out_of_range("blk.10.a", 10, 19), "the range is half-open at the bottom");
    require(!layer_out_of_range("blk.18.a", 10, 19), "the range is half-open at the top");
    require(layer_out_of_range("blk.19.a", 10, 19), "layer 19 is outside [10,19)");
    require(!layer_out_of_range("blk.3.a", 0, -1), "a negative hi disables the carve");
    require(layer_out_of_range("blk.3.a", 0, 0), "an empty range drops every layer");

    // `keep` is for a tensor a stage needs even though it does not run its layer - the split's router
    // lookahead reads every layer's router from one stage's table.
    const std::set<std::string> keep = {"blk.3.ffn_gate_inp.weight"};
    require(layer_out_of_range("blk.3.ffn_gate_inp.weight", 10, 19), "without keep, layer 3's router is dropped");
    require(!layer_out_of_range("blk.3.ffn_gate_inp.weight", 10, 19, &keep), "with keep, it is not");
    require(layer_out_of_range("blk.4.ffn_gate_inp.weight", 10, 19, &keep), "keep does not exempt its neighbours");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        check_name_parsing();

        // ---- a synthetic pack built so the carve arithmetic is checkable by hand: uneven layer sizes
        // (the real packs are uneven - layer 1 carries the PLE tensors and is 3x the others), a global
        // tensor, a router that `keep` holds, and sizes that are not multiples of the alignment so the
        // rounding is exercised rather than assumed.
        constexpr uint64_t kAlign = 64;
        constexpr int64_t kLayers = 6;
        TempDirectory dir;
        std::vector<Row> rows;
        rows.push_back({"token_embd.weight", 0});
        rows.push_back({"output.weight", 4096});
        rows.push_back({"output_hc_down.weight", 100});   // deliberately unaligned
        for (int64_t l = 0; l < kLayers; ++l) {
            const uint64_t base = l == 1 ? 100000 : 1000 + (uint64_t) l * 7;
            rows.push_back({"blk." + std::to_string(l) + ".attn_qkv.weight", base});
            rows.push_back({"blk." + std::to_string(l) + ".ffn_gate_inp.weight", 33});   // unaligned, kept
            rows.push_back({"blk." + std::to_string(l) + ".ssm_conv1d.weight", 65});     // one byte into the next block
        }
        write_index(dir.path, kAlign, rows);

        const std::set<std::string> keep_set = {"blk.0.ffn_gate_inp.weight", "blk.1.ffn_gate_inp.weight",
                                                "blk.2.ffn_gate_inp.weight", "blk.3.ffn_gate_inp.weight",
                                                "blk.4.ffn_gate_inp.weight", "blk.5.ffn_gate_inp.weight"};

        std::string err;
        check_ranges(dir.path, kLayers, nullptr, nullptr);
        check_ranges(dir.path, kLayers, nullptr, &keep_set);

        // A `skip` set and a range compose: the native-served names are skipped, then the range carves.
        const std::set<std::string> skip_set = {"blk.2.attn_qkv.weight"};
        check_ranges(dir.path, kLayers, &skip_set, nullptr);

        // The kept routers are resident on EVERY stage, so they must not appear in any range's per-layer
        // total - a double count is a stage priced for bytes it does not hold.
        std::vector<uint64_t> per_layer;
        uint64_t globals = 0, keep_bytes = 0;
        require(strata::core::WeightTable::layer_bytes(dir.path.string(), kLayers, per_layer, globals, keep_bytes,
                                                       err, nullptr, &keep_set),
                "layer_bytes: " + err);
        require(keep_bytes == 6 * align_up(33, kAlign), "the kept routers total " + std::to_string(keep_bytes));
        for (int64_t l = 0; l < kLayers; ++l)
            require(per_layer[(size_t) l] == align_up(l == 1 ? 100000 : 1000 + (uint64_t) l * 7, kAlign) +
                                               align_up(65, kAlign),
                    "layer " + std::to_string(l) + " holds its own two tensors and not its router");
        require(globals == align_up(4096, kAlign) + align_up(100, kAlign),
                "the globals are the head's two tensors (the embedding is 0 B)");

        // A layer ordinal beyond the model is refused rather than written past the vector.
        bool refused = false;
        try {
            refused = !strata::core::WeightTable::layer_bytes(dir.path.string(), 3, per_layer, globals,
                                                              keep_bytes, err, nullptr, nullptr);
        } catch (const std::exception&) {
            refused = true;
        }
        require(refused, "a layer beyond the model's count must be refused");

        // ---- the real pack, when one is named: the same invariants on bytes nobody made up.
        if (argc > 1) {
            const std::string pack = argv[1];
            std::FILE* probe = std::fopen((pack + "/index.txt").c_str(), "rb");
            require(probe != nullptr, "no index.txt under " + pack);
            std::fclose(probe);
            check_ranges(pack, 48, nullptr, nullptr);
            std::cout << "weights_carve_test: OK (synthetic; and the real pack at " << pack << ")\n";
            return 0;
        }
        std::cout << "weights_carve_test: OK (synthetic)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "weights_carve_test: FAILED - " << e.what() << "\n";
        return 1;
    }
}
