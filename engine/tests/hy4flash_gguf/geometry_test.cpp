// Validate model geometry from synthetic GGUF metadata.
#include "../core/gguf_fixture.hpp"

#include "strata/hy4flash_gguf/geometry.hpp"
#include "strata/hy4flash_gguf/fp8.hpp"
#include "strata/hy4flash_gguf/model_file.hpp"

#include <array>
#include <chrono>
#include <cmath>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using fixture::Kv;
using strata::hy4_flash::Attention;
using strata::hy4_flash::Geometry;

namespace {
int g_fail = 0;

void check(bool ok, const std::string& what) {
    std::printf("  %-84s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

/// Start from valid metadata and vary one field per test.
std::vector<Kv> keys(int blocks = 43, int mtp = 1, int tables = 4, int ngram = 3, int heads_per_ngram = 8,
                     uint64_t bucket = 10312007) {
    const int trunk = blocks - mtp;
    const int columns = ngram > 0 ? (ngram - 1) * heads_per_ngram : 0;
    // The real pattern: blocks 0, 1, 5, 9, ... are sparse, the rest sliding, and the draft block is sparse.
    std::vector<uint64_t> swa, head_count, idx_types;
    for (int b = 0; b < blocks; ++b) {
        const bool sliding = b >= 2 && b < trunk && (b - 2) % 4 != 3;
        swa.push_back(sliding ? 1 : 0);
        head_count.push_back(sliding ? 64 : 32);
        if (b < trunk) idx_types.push_back(sliding ? 0 : 1);
    }
    std::vector<uint64_t> ple_layers, offsets, vocabs, mults;
    uint64_t base = 0;
    for (int t = 0; t < tables; ++t) {
        ple_layers.push_back((uint64_t) (1 + t * 10));
        for (int c = 0; c < columns; ++c) {
            offsets.push_back(base);
            vocabs.push_back(bucket + (uint64_t) c);
            base += bucket + (uint64_t) c;
        }
        // ~10^14, which is why the fixture needs a u64 array: through u32s these would be truncated.
        for (int o = 0; o < ngram; ++o) mults.push_back(81853575799317ull + (uint64_t) (o * 1000));
    }
    std::vector<Kv> kv = {
        fixture::str("general.architecture", "hy_v4_flash"),
        fixture::u32("hy_v4_flash.block_count", (uint64_t) blocks),
        fixture::u32("hy_v4_flash.nextn_predict_layers", (uint64_t) mtp),
        fixture::u32("hy_v4_flash.leading_dense_block_count", 1),
        fixture::u32("hy_v4_flash.embedding_length", 3072),
        fixture::u32("hy_v4_flash.feed_forward_length", 13824),
        fixture::u32("hy_v4_flash.vocab_size", 120832),
        fixture::f32("hy_v4_flash.attention.layer_norm_rms_epsilon", 1e-5f),
        fixture::u32("hy_v4_flash.hyper_connection.count", 4),
        fixture::f32("hy_v4_flash.hyper_connection.epsilon", 1e-6f),
        fixture::f32("hy_v4_flash.hyper_connection.magnitude", 2.0f),
        fixture::u32("hy_v4_flash.expert_count", 512),
        fixture::u32("hy_v4_flash.expert_used_count", 16),
        fixture::u32("hy_v4_flash.expert_latent_length", 1536),
        fixture::u32("hy_v4_flash.expert_feed_forward_length", 1536),
        fixture::u32("hy_v4_flash.expert_shared_count", 1),
        fixture::u32("hy_v4_flash.expert_gating_func", 2),
        fixture::f32("hy_v4_flash.expert_weights_scale", 3.998f),
        fixture::u32("hy_v4_flash.expert_weights_norm", 1),
        fixture::u32("hy_v4_flash.rope.dimension_count", 64),
        fixture::f32("hy_v4_flash.rope.freq_base", 1e7f),
        fixture::f32("hy_v4_flash.rope.freq_base_swa", 1e4f),
        fixture::u32("hy_v4_flash.attention.head_count_kv", 1),
        fixture::u32("hy_v4_flash.attention.q_lora_rank", 1024),
        fixture::u32("hy_v4_flash.attention.kv_lora_rank", 512),
        fixture::u32("hy_v4_flash.attention.key_length", 576),
        fixture::u32("hy_v4_flash.attention.key_length_mla", 256),
        fixture::u32("hy_v4_flash.attention.value_length_mla", 256),
        fixture::u32("hy_v4_flash.attention.sliding_window", 512),
        fixture::u32("hy_v4_flash.attention.indexer.head_count", 16),
        fixture::u32("hy_v4_flash.attention.indexer.key_length", 128),
        fixture::u32("hy_v4_flash.attention.indexer.kpool", 4),
        fixture::u32("hy_v4_flash.attention.indexer.top_k", 2048),
        fixture::u32("hy_v4_flash.attention.indexer.kpool_select_tail", 1),
        fixture::u32("hy_v4_flash.embedding_length_per_layer_input", 96),
        fixture::u32("hy_v4_flash.ple.ngram_size", (uint64_t) ngram),
        fixture::u32("hy_v4_flash.ple.heads_per_ngram", (uint64_t) heads_per_ngram),
        fixture::u32("hy_v4_flash.engram.pad_token_id", 120002),
        fixture::u32("hy_v4_flash.context_length", 1048576),
        fixture::u32s("hy_v4_flash.attention.head_count", head_count),
        fixture::u32s("hy_v4_flash.attention.sliding_window_pattern", swa),
        fixture::u32s("hy_v4_flash.attention.indexer.types", idx_types),
        fixture::u32s("hy_v4_flash.ple.layers", ple_layers),
        fixture::u64s("hy_v4_flash.ple.head_offsets", offsets),
        fixture::u64s("hy_v4_flash.ple.head_vocab_sizes", vocabs),
        fixture::u64s("hy_v4_flash.ple.layer_multipliers", mults),
    };
    return kv;
}

Geometry parsed;

std::string read(const fs::path& dir, const std::string& stem, const std::vector<Kv>& kv, int max_context = 0,
                 const std::vector<fixture::Tensor>& tensors = {fixture::Tensor{"token_embd.weight",
                                                                                {32, 4}, 0, 1}}) {
    const fs::path p = dir / (stem + ".gguf");
    fixture::write(p, kv, tensors);
    const strata::GgufFile f(p.string());
    const std::string guard = strata::check_architecture(f);
    if (!guard.empty()) return "guard: " + guard;
    return strata::hy4_flash::read_geometry(f, parsed, max_context);
}

/// A tiny stand-in for the merged Engram table, of `type` and `rows` rows of `width`. GGUF order puts the row
/// width in dim 0, so the tensor is [width, rows] - which is also how a reader has to take it.
fixture::Tensor table(uint32_t type, uint64_t rows, uint64_t width = 96, uint64_t bytes_override = 0) {
    return fixture::Tensor{"per_layer_token_embd.weight", {width, rows}, type, 3, bytes_override};
}

void check_model_shards(const fs::path& root) {
    const auto first = root / "model-00001-of-00002.gguf";
    const auto second = root / "model-00002-of-00002.gguf";
    const fixture::Tensor a{"token_embd.weight", {32, 2}, 0, 3};
    const fixture::Tensor b{"per_layer_token_embd.weight", {96, 3}, 43, 7};
    fixture::write(first, {fixture::str("general.architecture", "hy_v4_flash"),
        fixture::u16("split.no", 0), fixture::u16("split.count", 2),
        fixture::i32("split.tensors.count", 2)}, {a});
    fixture::write(second, {fixture::u16("split.no", 1), fixture::u16("split.count", 2),
        fixture::i32("split.tensors.count", 2)}, {b});
    {
        strata::hy4_flash::ModelFile model(second.string());
        for (const auto& tensor : {a, b}) {
            const auto* t = model.info(tensor.name);
            if (!t) throw std::runtime_error("missing tensor");
            const auto n = strata::tensor_payload_bytes(*t);
            const auto offset = model.data_start() + t->offset;
            const auto* ptr = static_cast<const uint8_t*>(model.at(offset, n));
            std::vector<uint8_t> read(n);
            if (!ptr || !model.read_at(offset, read.data(), read.size()))
                throw std::runtime_error("mapped/read tensor failed");
            for (std::size_t i = 0; i < n; ++i)
                if (read[i] != fixture::pattern(tensor.seed, i) || ptr[i] != read[i])
                    throw std::runtime_error("wrong shard payload");
            std::vector<uint8_t> direct(n);
            if (!model.pread_at(offset, direct.data(), n) || direct != read)
                throw std::runtime_error("direct shard read mismatch");
            std::string path;
            uint64_t file_offset;
            if (!model.physical(offset, n, path, file_offset)) throw std::runtime_error("physical offset");
            strata::GgufFile physical(path);
            if (file_offset != physical.data_start() + physical.find(tensor.name)->offset)
                throw std::runtime_error("physical offset mismatch");
        }
        const auto* t = model.info(b.name);
        const auto end = model.data_start() + t->offset + strata::tensor_payload_bytes(*t);
        uint8_t byte;
        if (model.at(end, 1) || model.pread_at(end, &byte, 1)) throw std::runtime_error("out of bounds read accepted");
    }
    fs::rename(second, root / "held.gguf");
    bool missing = false;
    try { strata::hy4_flash::ModelFile model(first.string()); }
    catch (const std::exception&) { missing = true; }
    if (!missing) throw std::runtime_error("missing shard accepted");
}

void check_fp8() {
    std::array<uint8_t, 256> bytes{};
    std::array<float, 256> values{};
    for (int i = 0; i < 256; ++i) bytes[i] = uint8_t(i);
    strata::hy4_flash::dequant_f8_e4m3_row(bytes.data(), 256, values.data());
    for (int i = 0; i < 256; ++i) {
        const int mag = i & 127;
        if (mag == 127) {
            if (!std::isnan(values[i])) throw std::runtime_error("FP8 NaN");
            continue;
        }
        // Integer multiples of 2^-9; independent of the decoder's ldexp implementation.
        const int units = mag < 8 ? mag : (8 + (mag & 7)) * (1 << ((mag >> 3) - 1));
        const float expected = (i & 128 ? -1.f : 1.f) * float(units) / 512.f;
        if (values[i] != expected || std::signbit(values[i]) != bool(i & 128))
            throw std::runtime_error("FP8 value mismatch");
    }
}

int real_model(const char* shard) {
    const strata::GgufModel m = strata::GgufModel::open(shard);
    const std::string guard = strata::check_architecture(m.meta());
    std::printf("guard: %s\n", guard.empty() ? "ok" : guard.c_str());
    if (!guard.empty()) return 1;
    Geometry g;
    const std::string err = strata::hy4_flash::read_geometry(m.meta(), g, 0);
    if (!err.empty()) {
        std::printf("read_geometry: %s\n", err.c_str());
        return 1;
    }
    std::printf("%s\n", strata::hy4_flash::describe(g).c_str());
    for (const strata::hy4_flash::EngramTable& e : g.engram_tables)
        std::printf("  engram block %2d  rows %12llu  row_base %12llu  first offset %llu\n", e.block_id,
                    (unsigned long long) e.rows, (unsigned long long) e.row_base,
                    (unsigned long long) (e.head_offsets.empty() ? 0 : e.head_offsets[0]));
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--real") == 0) return real_model(argv[2]);

    const fs::path dir = fs::temp_directory_path() / ("strata-hy4-flash-gguf-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    try {
        check_model_shards(dir);
        check_fp8();
    } catch (const std::exception& err) {
        check(false, err.what());
    }

    std::printf("a complete file\n");
    check(read(dir, "ok", keys()).empty(), "read_geometry accepts it");
    check(parsed.block_count == 43 && parsed.trunk_blocks == 42 && parsed.n_mtp_blocks == 1,
          "43 blocks = 42 trunk + 1 draft");
    check(parsed.hidden_dim == 3072 && parsed.hc_mult == 4 && parsed.hc_width() == 12288,
          "the inter-block state is hc_mult x hidden = 12288, not 3072");
    check(parsed.layers.size() == 43 && parsed.layers[0].expert_ordinal == -1 &&
              parsed.layers[0].attention == Attention::Sparse && parsed.layers[0].n_heads == 32,
          "block 0: dense FFN, sparse, 32 heads");
    check(parsed.layers[2].attention == Attention::Sliding && parsed.layers[2].n_heads == 64 &&
              !parsed.layers[2].has_indexer,
          "block 2: sliding, 64 heads, no indexer");
    check(parsed.layers[1].attention == Attention::Sparse && parsed.layers[1].has_indexer,
          "block 1: sparse, owns an indexer");
    check(parsed.layers[42].is_mtp && parsed.layers[42].attention == Attention::Sparse,
          "block 42 is the draft block, shaped like a sparse one");

    std::printf("block id is not expert ordinal\n");
    check(parsed.layers[0].expert_ordinal == -1, "the dense block has none");
    check(parsed.layers[1].expert_ordinal == 0, "the first MoE block (block 1) is ordinal 0");
    check(parsed.layers[41].expert_ordinal == 40, "the last trunk block is ordinal 40");
    check(parsed.layers[42].expert_ordinal == 41, "the draft block takes the LAST ordinal, 41");
    check(parsed.n_expert_ordinals() == 42, "42 ordinals in the expert pack, not 41");

    std::printf("the Engram tables are views of one merged tensor\n");
    check(parsed.engram_tables.size() == 4 && parsed.engram_heads == 16 && parsed.engram_head_dim == 96,
          "4 tables, 16 columns of 96");
    check(parsed.engram_predecessors() == 2, "a 3-gram needs 2 predecessor tokens");
    check(parsed.engram_lookup_dim() == 1536, "16 x 96 = 1536 lookup features per token per table");
    check(parsed.engram_tables[0].row_base == 0 &&
              parsed.engram_tables[1].row_base == parsed.engram_tables[0].rows,
          "table 1 starts where table 0 ends");
    check(parsed.engram_total_rows ==
              parsed.engram_tables[3].row_base + parsed.engram_tables[3].rows,
          "the merged row count is the last table's end");
    check(parsed.layers[1].engram_table == 0 && parsed.layers[11].engram_table == 1 &&
              parsed.layers[21].engram_table == 2 && parsed.layers[31].engram_table == 3,
          "blocks 1/11/21/31 carry tables 0..3");
    check(parsed.layers[0].engram_table == -1 && parsed.layers[2].engram_table == -1,
          "every other block carries none");
    check(parsed.engram_tables[0].multipliers.size() == 3 &&
              parsed.engram_tables[0].multipliers[0] == 81853575799317ull,
          "the multipliers are read at full width (~10^14), not truncated to u32");

    std::printf("a file whose two descriptions disagree is refused\n");
    {   // the pattern says sliding, the indexer array says it owns one
        std::vector<Kv> kv = keys();
        for (Kv& k : kv)
            if (k.key == "hy_v4_flash.attention.indexer.types") k.arr[2] = 1;
        check(read(dir, "idx-mismatch", kv).find("block 2") != std::string::npos,
              "an indexer on a sliding block is refused, naming the block");
    }
    {   // every block sliding: there is no sparse block to select from
        std::vector<Kv> kv = keys();
        for (Kv& k : kv) {
            if (k.key == "hy_v4_flash.attention.sliding_window_pattern")
                for (uint64_t& x : k.arr) x = 1;
            if (k.key == "hy_v4_flash.attention.indexer.types")
                for (uint64_t& x : k.arr) x = 0;
        }
        check(read(dir, "all-sliding", kv).find("mixes the two") != std::string::npos,
              "a pattern with no sparse block is refused");
    }
    {   // offsets rebased per table: the hash would land every table on the first one's rows
        std::vector<Kv> kv = keys();
        for (Kv& k : kv)
            if (k.key == "hy_v4_flash.ple.head_offsets") {
                const uint64_t first = k.arr[16];
                for (size_t i = 16; i < k.arr.size(); ++i) k.arr[i] -= first;
            }
        check(read(dir, "rebased-offsets", kv).find("not rebased per table") != std::string::npos,
              "per-table head offsets are refused");
    }
    {
        std::vector<Kv> kv = keys();
        for (auto it = kv.begin(); it != kv.end(); ++it)
            if (it->key == "hy_v4_flash.hyper_connection.count") { kv.erase(it); break; }
        check(read(dir, "no-hc", kv).find("hyper_connection.count") != std::string::npos,
              "a file declaring no hyper-connections is refused (every buffer is sized from it)");
    }
    {
        std::vector<Kv> kv = keys();
        for (Kv& k : kv)
            if (k.key == "hy_v4_flash.attention.key_length") k.u = 512;   // dropped the rope part
        check(read(dir, "bad-key-length", kv).find("one shared key per token") != std::string::npos,
              "key_length that is not kv_lora_rank + rope_dim is refused");
    }
    {
        std::vector<Kv> kv = keys();
        for (auto it = kv.begin(); it != kv.end(); ++it)
            if (it->key == "hy_v4_flash.attention.sliding_window") { kv.erase(it); break; }
        check(read(dir, "no-window", kv).find("sliding_window") != std::string::npos,
              "a missing sliding window is refused");
    }

    std::printf("the architecture guard: a file that cannot be read as a layer pattern\n");
    {   // the guard runs before read_geometry, so its refusals come back prefixed
        std::vector<Kv> kv = keys();
        for (auto it = kv.begin(); it != kv.end(); ++it)
            if (it->key == "hy_v4_flash.expert_count") { kv.erase(it); break; }
        check(read(dir, "g-no-experts", kv).find("expert_count") != std::string::npos,
              "a missing scalar key is refused, naming it");
    }
    {
        std::vector<Kv> kv = keys();
        for (auto it = kv.begin(); it != kv.end(); ++it)
            if (it->key == "hy_v4_flash.ple.head_offsets") { kv.erase(it); break; }
        check(read(dir, "g-no-offsets", kv).find("ple.head_offsets") != std::string::npos,
              "a missing per-head array is refused, naming it");
    }
    {   // the head count covers the draft block too: 43 entries, not the trunk's 42
        std::vector<Kv> kv = keys();
        for (Kv& k : kv)
            if (k.key == "hy_v4_flash.attention.head_count") k.arr.resize(42);
        check(read(dir, "g-heads-42", kv).find("has 42 entries, not 43") != std::string::npos,
              "attention.head_count is one per block, draft block included");
    }
    {   // the indexer array covers the TRUNK only: 42 entries, one fewer than the two above
        std::vector<Kv> kv = keys();
        for (Kv& k : kv)
            if (k.key == "hy_v4_flash.attention.indexer.types") k.arr.push_back(1);
        check(read(dir, "g-idx-43", kv).find("has 43 entries, not 42") != std::string::npos,
              "attention.indexer.types is one per TRUNK block, not per block");
    }
    {
        std::vector<Kv> kv = keys();
        for (Kv& k : kv)
            if (k.key == "hy_v4_flash.ple.head_vocab_sizes") k.arr.pop_back();
        check(read(dir, "g-short-vocabs", kv).find("entries, not 64") != std::string::npos,
              "head_vocab_sizes is tables x columns");
    }
    {
        std::vector<Kv> kv = keys();
        for (Kv& k : kv)
            if (k.key == "hy_v4_flash.ple.layer_multipliers") k.arr.pop_back();
        check(read(dir, "g-short-mults", kv).find("entries, not 12") != std::string::npos,
              "layer_multipliers is tables x ngram_size");
    }
    {   // the draft block has no table, so a ple layer there would be a different hash contract
        std::vector<Kv> kv = keys();
        for (Kv& k : kv)
            if (k.key == "hy_v4_flash.ple.layers") k.arr.back() = 42;
        check(read(dir, "g-ple-on-mtp", kv).find("past the 42 trunk blocks") != std::string::npos,
              "a ple layer outside the trunk is refused");
    }
    {
        std::vector<Kv> kv = keys(43, 43);
        check(read(dir, "g-all-mtp", kv).find("not smaller than") != std::string::npos,
              "nextn_predict_layers must leave trunk blocks");
    }
    {
        std::vector<Kv> kv = keys(43, 1, 4, 1);   // a 1-gram has no predecessor to hash
        check(read(dir, "g-ngram-1", kv).find("no lookup column") != std::string::npos,
              "ngram_size 1 is refused");
    }

    std::printf("the merged table's stored type: a 96-value row rules most quant types out\n");
    {
        // A tiny bucket count, because the fixture writes the payload: 4 tables x 16 columns x 32 rows is
        // 2,048 rows, which at 192 B/row is 393 KB rather than 127 GB.
        constexpr uint64_t kBucket = 32;
        std::vector<Kv> kv = keys(43, 1, 4, 3, 8, kBucket);
        uint64_t rows = 0;
        for (int t = 0; t < 4; ++t)
            for (uint64_t c = 0; c < 16; ++c) rows += kBucket + c;
        // 8 = Q8_0 (block 32): 3 blocks of 34 B = 102 B per row, the type this checkpoint ships.
        check(read(dir, "t-q8", kv, 0, {table(8, rows)}).empty() && parsed.engram_row_bytes == 102,
              "Q8_0 (block 32) gives 102 B rows");
        // 30 = BF16 (block 1): 192 B, the quality baseline.
        check(read(dir, "t-bf16", kv, 0, {table(30, rows)}).empty() && parsed.engram_row_bytes == 192,
              "BF16 (block 1) gives 192 B rows");
        // 20 = IQ4_NL (block 32): 54 B, the capacity candidate. The type is read, not pinned, so a deliberate
        // change of the table's quantisation is not blocked by this check.
        check(read(dir, "t-iq4nl", kv, 0, {table(20, rows)}).empty() && parsed.engram_row_bytes == 54,
              "IQ4_NL (block 32) gives 54 B rows - a lower-bit table is allowed");
        // 12 = Q4_K, block 256: a 96-value row cannot be expressed in it at all.
        check(read(dir, "t-q4k", kv, 0, {table(12, rows, 96, rows * 64)}).find("block that divides the row") !=
                  std::string::npos,
              "Q4_K (block 256) is refused: 96 is not a multiple of 256");
        // 23 = IQ4_XS, also block 256 - the one a reader might reach for by name.
        check(read(dir, "t-iq4xs", kv, 0, {table(23, rows, 96, rows * 52)}).find("block that divides the row") !=
                  std::string::npos,
              "IQ4_XS (block 256) is refused for the same reason");
        check(read(dir, "t-rows", kv, 0, {table(8, rows - 1)}).find("bucket counts sum to") != std::string::npos,
              "a table whose rows disagree with the bucket counts is refused");
        check(read(dir, "t-width", kv, 0, {table(8, rows, 64)}).find("wide, not the 96") != std::string::npos,
              "a table whose rows are not 96 wide is refused");
        check(read(dir, "t-absent", kv).empty() && parsed.engram_row_bytes == 0,
              "a shard without the table reads fine, leaving the type unknown (a split model)");
    }

    std::printf("max_context is the engine's limit, not the file's\n");
    check(read(dir, "ctx", keys(), 32768).empty() && parsed.max_context == 32768,
          "a given limit wins over the file's 1,048,576");
    check(read(dir, "ctx0", keys(), 0).empty() && parsed.max_context == 1048576,
          "0 falls back to the file's own context_length");

    fs::remove_all(dir);
    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
