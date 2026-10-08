// src/core/dflash_load_test.cpp - the DFlash artifact contract (docs/DFLASH.md): metadata parsing,
// the two accepted naming families, and every refusal the loader promises.  Host-only: GGUF
// parsing needs no GPU and no model.  The fixtures are tiny synthetic GGUFs with a consistent
// (2-layer) geometry; the canonical-geometry fast-path check runs on the geometry struct alone.
#include "strata/core/dflash.hpp"
#include "gguf_fixture.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++g_fail; }
}
/// Expect `load` to succeed and return the geometry.
bool loads(const fs::path& p, DFlashGeometry* out = nullptr) {
    DFlashArtifact a;
    std::string err;
    if (!a.open(p.string(), err)) {
        std::fprintf(stderr, "FAIL: %s did not load: %s\n", p.string().c_str(), err.c_str());
        ++g_fail;
        return false;
    }
    if (out) *out = a.geom();
    return true;
}
/// Expect `load` to fail, printing the loader's message once.
bool refuses(const fs::path& p, const char* what) {
    DFlashArtifact a;
    std::string err;
    if (a.open(p.string(), err)) {
        std::fprintf(stderr, "FAIL: %s: the artifact loaded but should have been refused (%s)\n", what,
                     p.string().c_str());
        ++g_fail;
        return false;
    }
    std::printf("refused %-28s %s\n", what, err.c_str());
    return true;
}

// A consistent tiny artifact: hidden 32, 2 layers, 4Q/1KV x 8, MLP 16, vocab 64, block 3, taps [0,1].
constexpr int kH = 32, kL = 2, kQ = 32, kKV = 8, kD = 8, kI = 16, kF = 64;   // kF = hidden * 2 taps
std::vector<fixture::Kv> meta(bool anchor = true, bool causal = false, int64_t markov = 0,
                              int64_t selector = 0, bool with_taps = true) {
    std::vector<fixture::Kv> kv = {
        fixture::str("general.architecture", "dflash"),
        fixture::u32("dflash.embedding_length", kH),
        fixture::u32("dflash.block_count", kL),
        fixture::u32("dflash.attention.head_count", 4),
        fixture::u32("dflash.attention.head_count_kv", 1),
        fixture::u32("dflash.attention.key_length", kD),
        fixture::u32("dflash.feed_forward_length", kI),
        fixture::u32("dflash.vocab_size", 64),
        fixture::u32("dflash.block_size", 3),
        fixture::str("dflash.sample_from_anchor", anchor ? "true" : "false"),
        fixture::str("dflash.attention.causal", causal ? "true" : "false"),
        fixture::u32("dflash.mask_token_id", 7),
        fixture::u32("dflash.markov_rank", (uint64_t) markov),
        fixture::u32("dflash.selector_top_k", (uint64_t) selector),
    };
    if (with_taps) kv.push_back(fixture::arr_u32("dflash.target_layers", {0, 1}));
    return kv;
}
std::vector<fixture::Tensor> tensors_llama() {
    std::vector<fixture::Tensor> ts = {
        {"fc.weight", {(uint64_t) kF, (uint64_t) kH}, 30, 1},
        {"enc.output_norm.weight", {(uint64_t) kH}, 30, 2},
        {"output_norm.weight", {(uint64_t) kH}, 30, 3},
    };
    char buf[128];
    for (int l = 0; l < kL; ++l) {
        struct M { const char* fmt; std::vector<uint64_t> shape; };
        const M ms[] = {
            {"blk.%d.attn_norm.weight", {(uint64_t) kH}},
            {"blk.%d.ffn_norm.weight", {(uint64_t) kH}},
            {"blk.%d.attn_q.weight", {(uint64_t) kH, (uint64_t) kQ}},
            {"blk.%d.attn_k.weight", {(uint64_t) kH, (uint64_t) kKV}},
            {"blk.%d.attn_v.weight", {(uint64_t) kH, (uint64_t) kKV}},
            {"blk.%d.attn_output.weight", {(uint64_t) kQ, (uint64_t) kH}},
            {"blk.%d.attn_q_norm.weight", {(uint64_t) kD}},
            {"blk.%d.attn_k_norm.weight", {(uint64_t) kD}},
            {"blk.%d.ffn_gate.weight", {(uint64_t) kH, (uint64_t) kI}},
            {"blk.%d.ffn_up.weight", {(uint64_t) kH, (uint64_t) kI}},
            {"blk.%d.ffn_down.weight", {(uint64_t) kI, (uint64_t) kH}},
        };
        for (const auto& m : ms) {
            std::snprintf(buf, sizeof buf, m.fmt, l);
            ts.push_back({buf, m.shape, 30, (uint8_t) (41 + ts.size())});
        }
    }
    return ts;
}
}  // namespace

int main() {
    fs::path dir = fs::temp_directory_path() / "strata_dflash_load_test";
    fs::create_directories(dir);

    {   // a valid artifact in llama.cpp's naming family: parsed, all 25 tensors resolved, bytes counted
        const fs::path p = dir / "valid_llama.gguf";
        const auto ts = tensors_llama();
        fixture::write(p, meta(), ts);
        DFlashGeometry g;
        check(loads(p, &g), "valid llama-family artifact loads");
        check(g.hidden == kH && g.layers == kL && g.n_head == 4 && g.n_head_kv == 1 && g.head_dim == kD &&
                  g.intermediate == kI && g.vocab == 64 && g.block_size == 3 && g.mask_token_id == 7 &&
                  g.target_layers == std::vector<int32_t>{0, 1} && g.sample_from_anchor && !g.causal,
              "geometry parsed from metadata");
        DFlashArtifact a;
        std::string err;
        check(a.open(p.string(), err), "reopen for the inventory");
        check(a.tensors().size() == ts.size(), "every tensor resolved");
        check(a.tensor("fc") != nullptr && a.tensor("layers.0.mlp.down_proj") != nullptr &&
                  a.tensor("hidden_norm") != nullptr && a.tensor("norm") != nullptr,
              "canonical names resolve");
        uint64_t aligned_bytes = 0;
        for (const auto& t : ts) {
            uint64_t n = 1;
            for (auto d : t.shape) n *= d;
            aligned_bytes += (n * 2 + 31) & ~uint64_t(31);
        }
        check(a.weight_bytes() == aligned_bytes, "weight_bytes counts the aligned device inventory");
    }
    {   // the same artifact in the raw-HF naming family resolves to the same canonical inventory
        const fs::path p = dir / "valid_hf.gguf";
        std::vector<fixture::Tensor> ts = {
            {"fc.weight", {(uint64_t) kF, (uint64_t) kH}, 30, 1},
            {"hidden_norm.weight", {(uint64_t) kH}, 30, 2},
            {"norm.weight", {(uint64_t) kH}, 30, 3},
        };
        char buf[128];
        for (int l = 0; l < kL; ++l) {
            struct M { const char* fmt; std::vector<uint64_t> shape; };
            const M ms[] = {
                {"layers.%d.input_layernorm.weight", {(uint64_t) kH}},
                {"layers.%d.post_attention_layernorm.weight", {(uint64_t) kH}},
                {"layers.%d.self_attn.q_proj.weight", {(uint64_t) kH, (uint64_t) kQ}},
                {"layers.%d.self_attn.k_proj.weight", {(uint64_t) kH, (uint64_t) kKV}},
                {"layers.%d.self_attn.v_proj.weight", {(uint64_t) kH, (uint64_t) kKV}},
                {"layers.%d.self_attn.o_proj.weight", {(uint64_t) kQ, (uint64_t) kH}},
                {"layers.%d.self_attn.q_norm.weight", {(uint64_t) kD}},
                {"layers.%d.self_attn.k_norm.weight", {(uint64_t) kD}},
                {"layers.%d.mlp.gate_proj.weight", {(uint64_t) kH, (uint64_t) kI}},
                {"layers.%d.mlp.up_proj.weight", {(uint64_t) kH, (uint64_t) kI}},
                {"layers.%d.mlp.down_proj.weight", {(uint64_t) kI, (uint64_t) kH}},
            };
            for (const auto& m : ms) {
                std::snprintf(buf, sizeof buf, m.fmt, l);
                ts.push_back({buf, m.shape, 30, (uint8_t) (61 + ts.size())});
            }
        }
        fixture::write(p, meta(), ts);
        DFlashArtifact a;
        std::string err;
        check(a.open(p.string(), err), "valid hf-family artifact loads");
        check(a.tensors().size() == ts.size() && a.tensor("fc") && a.tensor("hidden_norm") &&
                  a.tensor("norm") && a.tensor("layers.1.self_attn.q_proj"),
              "hf-family file resolves the same canonical names");
    }
    {   // wrong architecture
        auto kv = meta();
        kv[0] = fixture::str("general.architecture", "qwen3");
        check(refuses([&] { fixture::write(dir / "arch.gguf", kv, tensors_llama()); return dir / "arch.gguf"; }(),
                      "wrong architecture"),
              "wrong architecture refused");
    }
    {   // a required metadata key missing
        auto kv = meta();
        kv.erase(std::remove_if(kv.begin(), kv.end(), [](const fixture::Kv& k) {
                     return k.key == "dflash.block_size";
                 }), kv.end());
        check(refuses([&] { fixture::write(dir / "nokey.gguf", kv, tensors_llama()); return dir / "nokey.gguf"; }(),
                      "missing block_size"),
              "missing required key refused");
    }
    {   // a required tensor missing
        auto ts = tensors_llama();
        ts.erase(ts.begin() + 5);   // blk.1.attn_q.weight (3 fixed + 11 per layer)
        check(refuses([&] { fixture::write(dir / "missing.gguf", meta(), ts); return dir / "missing.gguf"; }(),
                      "missing tensor"),
              "missing tensor refused");
    }
    {   // a tensor whose shape disagrees with the metadata
        auto ts = tensors_llama();
        ts[5].shape = {(uint64_t) kH, (uint64_t) kQ + 8};
        check(refuses([&] { fixture::write(dir / "shape.gguf", meta(), ts); return dir / "shape.gguf"; }(),
                      "wrong shape"),
              "wrong shape refused");
    }
    for (const uint32_t type : {2u, 6u, 8u}) {
        auto ts = tensors_llama();
        ts[0].type = type;   // the fusion matrix has a width divisible by 32
        const fs::path path = dir / ("quant-" + std::to_string(type) + ".gguf");
        fixture::write(path, meta(), ts);
        check(loads(path), "quantized matrix loaded");
        DFlashArtifact a;
        std::string e;
        check(a.open(path.string(), e), "quantized payload parsed");
        check(a.tensor("fc") && a.tensor("fc")->type == (int) type, "matrix type preserved");
        check(a.tensor("fc") && a.tensor("fc")->bytes < (uint64_t) kF * kH * 2, "quantized weight bytes counted");
        ts[1].type = type;   // norms must retain BF16
        fixture::write(path, meta(), ts);
        check(refuses(path, "quantized norm"), "quantized norm refused");
    }
    {   // a non-BF16 tensor
        auto ts = tensors_llama();
        ts[4].type = 0;   // F32
        check(refuses([&] { fixture::write(dir / "f32.gguf", meta(), ts); return dir / "f32.gguf"; }(),
                      "F32 tensor"),
              "non-BF16 tensor refused");
    }
    {   // the layout refusals, each with its own file
        auto bad_taps = meta();
        bad_taps.back() = fixture::arr_u32("dflash.target_layers", {1, 0});
        check(refuses([&] { fixture::write(dir / "taps.gguf", bad_taps, tensors_llama()); return dir / "taps.gguf"; }(),
                      "taps not increasing"),
              "non-increasing taps refused");
        check(refuses([&] {
                  auto kv = meta();
                  kv.back() = fixture::arr_u32("dflash.target_layers", {0, 2, 1});   // not increasing
                  fixture::write(dir / "taps2.gguf", kv, tensors_llama());
                  return dir / "taps2.gguf";
              }(),
              "taps not increasing"),
              "non-increasing taps refused");
        {   // a tap count the loader accepts structurally but the fast path does not implement
            const fs::path p = dir / "onetap.gguf";
            auto kv = meta();
            kv.back() = fixture::arr_u32("dflash.target_layers", {0});
            auto ts = tensors_llama();
            ts[0].shape = {(uint64_t) kH, (uint64_t) kH};   // fc over one tap
            fixture::write(p, kv, ts);
            DFlashGeometry g;
            check(loads(p, &g) && g.target_layers == std::vector<int32_t>{0},
                  "a one-tap artifact parses (the loader does not hardcode five)");
            std::string err;
            check(!DFlashArtifact::validate_supported(g, err), "the fast path refuses one tap");
        }
        auto kv = meta();
        kv.erase(std::remove_if(kv.begin(), kv.end(), [](const fixture::Kv& k) {
                     return k.key == "dflash.target_layers";
                 }), kv.end());
        check(refuses([&] { fixture::write(dir / "notaps.gguf", kv, tensors_llama()); return dir / "notaps.gguf"; }(),
                      "no target_layers"),
              "missing taps refused");
        check(refuses([&] { fixture::write(dir / "generic.gguf", meta(false), tensors_llama()); return dir / "generic.gguf"; }(),
                      "1+N layout"),
              "sample_from_anchor=false refused");
        check(refuses([&] { fixture::write(dir / "causal.gguf", meta(true, true), tensors_llama()); return dir / "causal.gguf"; }(),
                      "causal attention"),
              "causal=true refused");
        check(refuses([&] { fixture::write(dir / "markov.gguf", meta(true, false, 4), tensors_llama()); return dir / "markov.gguf"; }(),
                      "DSpark markov head"),
              "markov refused");
        check(refuses([&] { fixture::write(dir / "sel.gguf", meta(true, false, 0, 8), tensors_llama()); return dir / "sel.gguf"; }(),
                      "DFlash2 selector"),
              "selector refused");
        auto emb = tensors_llama();
        emb.push_back({"token_embd.weight", {64, (uint64_t) kH}, 30, 90});
        check(refuses([&] { fixture::write(dir / "emb.gguf", meta(), emb); return dir / "emb.gguf"; }(),
                      "own embedding"),
              "shipped embedding refused");
        auto extra = tensors_llama();
        extra.push_back({"something.else.weight", {(uint64_t) kH}, 30, 91});
        check(refuses([&] { fixture::write(dir / "extra.gguf", meta(), extra); return dir / "extra.gguf"; }(),
                      "unknown tensor"),
              "unknown tensor refused");
        auto huge = meta();
        for (auto& k : huge)
            if (k.key == "dflash.block_size") { k.u = 0; k.type = 4; }
        check(refuses([&] { fixture::write(dir / "bs0.gguf", huge, tensors_llama()); return dir / "bs0.gguf"; }(),
                      "block_size 0"),
              "block_size 0 refused");
    }
    {   // validate_supported: the canonical fast path on the geometry struct alone
        DFlashGeometry g;
        g.hidden = 2560; g.layers = 5; g.n_head = 24; g.n_head_kv = 2; g.head_dim = 256;
        g.intermediate = 7680; g.block_size = 7; g.rope_theta = 1e7;
        g.target_layers = {3, 15, 23, 35, 43};
        std::string err;
        check(DFlashArtifact::validate_supported(g, err), "canonical geometry is supported");

        DFlashGeometry bad = g;
        bad.hidden = 2048;
        check(!DFlashArtifact::validate_supported(bad, err) && err.find("embedding_length") != std::string::npos,
              "wrong hidden refused by name");
        bad = g;
        bad.target_layers = {3, 15, 23, 35};
        check(!DFlashArtifact::validate_supported(bad, err), "four taps refused");
        bad = g;
        bad.n_head_kv = 4;
        check(!DFlashArtifact::validate_supported(bad, err), "wrong kv heads refused");
    }
    {   // the batch attention's identity table: one [0, cap) row per query row
        std::vector<int32_t> ids(3 * 7, -1);
        dflash_identity_fill(ids.data(), 3, 7);
        bool ok = true;
        for (int r = 0; r < 3; ++r)
            for (int64_t i = 0; i < 7; ++i)
                if (ids[(size_t) r * 7 + (size_t) i] != (int32_t) i) ok = false;
        check(ok, "identity selections valid for every query row");
    }
    {   // a truncated file fails with a precise error, not a crash
        const fs::path p = dir / "cut.gguf";
        fixture::write(p, meta(), tensors_llama(), 64);
        check(refuses(p, "truncated"), "truncated artifact refused");
    }

    std::printf(g_fail ? "dflash_load_test: %d FAILURES\n" : "dflash_load_test: ok\n", g_fail);
    fs::remove_all(dir);
    return g_fail ? 1 : 0;
}
