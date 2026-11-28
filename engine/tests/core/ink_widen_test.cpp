// strata-inkling: widening of the drafter's 16-bit vectors from a BF16 / F16 mtp.safetensors (see strata/core/ink_widen.hpp).
#include "strata/core/ink_widen.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace strata::core;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

float widen_one(uint16_t bits, int fmt) {
    uint8_t src[2] = {static_cast<uint8_t>(bits & 0xFF), static_cast<uint8_t>(bits >> 8)};
    float f = 0.f;
    ink_widen16_to_f32(src, fmt, 1, reinterpret_cast<uint8_t*>(&f));
    return f;
}
}  // namespace

int main() {
    // which tensors are widened: every vector, the norms, the biases and fuse_weight; the 2-D weights stay narrow; F32 stays as is
    CHECK(ink_dspark_wants_f32("layers.0.input_layernorm.weight", 1, 102));
    CHECK(ink_dspark_wants_f32("layers.0.self_attn.q_norm.weight", 1, 102));
    CHECK(ink_dspark_wants_f32("markov_head.dflare_norm.weight", 2, 102));
    CHECK(ink_dspark_wants_f32("fuse_weight", 2, 102));
    CHECK(ink_dspark_wants_f32("hidden_out.bias", 1, 103));
    CHECK(!ink_dspark_wants_f32("layers.0.self_attn.o_proj.weight", 2, 102));
    CHECK(!ink_dspark_wants_f32("mask_embed", 2, 102));
    CHECK(!ink_dspark_wants_f32("fuse_weight", 2, 101));

    // raw HF safetensors trunk: non-matmul tensors must widen BF16/F16 -> F32; dense linear projections stay narrow; F32 stays as is
    CHECK(ink_trunk_wants_f32("model.llm.embed_norm.weight", 1, 102));
    CHECK(ink_trunk_wants_f32("model.llm.norm.weight", 1, 102));
    CHECK(ink_trunk_wants_f32("model.llm.layers.0.attn_norm.weight", 1, 102));
    CHECK(ink_trunk_wants_f32("model.llm.layers.0.mlp_norm.weight", 1, 102));
    CHECK(ink_trunk_wants_f32("model.llm.layers.5.q_input_norm.weight", 1, 102));
    CHECK(ink_trunk_wants_f32("model.llm.layers.0.attn_sconv.weight", 3, 102));
    CHECK(ink_trunk_wants_f32("model.llm.layers.0.mlp_sconv.weight", 3, 102));
    CHECK(ink_trunk_wants_f32("model.llm.layers.0.attn.k_sconv.weight", 3, 102));
    CHECK(ink_trunk_wants_f32("model.llm.layers.5.attn.k_sconv.static_w", 2, 102));
    CHECK(ink_trunk_wants_f32("model.llm.layers.0.attn.q_l2_clip.scale", 2, 102));
    CHECK(ink_trunk_wants_f32("model.llm.layers.0.attn.rel_l2_clip.scale", 2, 102));
    CHECK(ink_trunk_wants_f32("model.llm.layers.0.attn.rel_logits_proj.proj", 2, 102));
    CHECK(ink_trunk_wants_f32("model.llm.layers.0.mlp.gate.weight", 2, 102));
    CHECK(ink_trunk_wants_f32("model.llm.layers.6.controllers.attn.down_weight", 2, 102));
    CHECK(ink_trunk_wants_f32("model.llm.layers.6.controllers.mlp.gain_weight", 2, 102));
    CHECK(ink_trunk_wants_f32("model.llm.layers.6.controllers.v.forget_weight", 2, 103));
    CHECK(!ink_trunk_wants_f32("model.llm.embed.weight", 2, 102));
    CHECK(!ink_trunk_wants_f32("model.llm.unembed.weight", 2, 102));
    CHECK(!ink_trunk_wants_f32("model.llm.layers.0.attn.wq_du.weight", 2, 102));
    CHECK(!ink_trunk_wants_f32("model.llm.layers.0.attn.wk_dv.weight", 2, 102));
    CHECK(!ink_trunk_wants_f32("model.llm.layers.0.attn.wv_dv.weight", 2, 102));
    CHECK(!ink_trunk_wants_f32("model.llm.layers.0.attn.wr_du.weight", 2, 102));
    CHECK(!ink_trunk_wants_f32("model.llm.layers.0.attn.wo_ud.weight", 2, 102));
    CHECK(!ink_trunk_wants_f32("model.llm.layers.0.attn.gate_attn.weight", 2, 102));
    CHECK(!ink_trunk_wants_f32("model.llm.layers.0.attn.attn_sink_proj.weight", 2, 102));
    CHECK(!ink_trunk_wants_f32("model.llm.layers.5.attn.k_sconv.weight_proj", 2, 102));
    CHECK(!ink_trunk_wants_f32("model.llm.layers.5.attn.v_sconv.weight_proj", 2, 102));
    CHECK(!ink_trunk_wants_f32("model.llm.layers.0.mlp.gate.bias", 1, 101));
    CHECK(!ink_trunk_wants_f32("model.llm.attn_res.output_query", 1, 101));

    // BF16: the high half of the float
    CHECK(widen_one(0x3F80, 102) == 1.0f);
    CHECK(widen_one(0xBF80, 102) == -1.0f);
    CHECK(widen_one(0x0000, 102) == 0.0f);
    CHECK(widen_one(0x4049, 102) > 3.14f && widen_one(0x4049, 102) < 3.15f);
    // IEEE half: 1.0, -2.0, 0.5, the smallest subnormal 2^-24, the largest 65504, infinity
    CHECK(widen_one(0x3C00, 103) == 1.0f);
    CHECK(widen_one(0xC000, 103) == -2.0f);
    CHECK(widen_one(0x3800, 103) == 0.5f);
    CHECK(widen_one(0x0001, 103) == 5.9604644775390625e-08f);
    CHECK(widen_one(0x7BFF, 103) == 65504.0f);
    CHECK(widen_one(0x7C00, 103) > 3.4e38f);
    // a run of values
    std::vector<uint8_t> narrow = {0x80, 0x3F, 0x00, 0xC0, 0x00, 0x40};  // BF16 1.0, -2.0, 2.0
    std::vector<float> wide(3);
    ink_widen16_to_f32(narrow.data(), 102, 3, reinterpret_cast<uint8_t*>(wide.data()));
    CHECK(wide[0] == 1.0f && wide[1] == -2.0f && wide[2] == 2.0f);

    if (failed) { std::fprintf(stderr, "%d check(s) failed\n", failed); return 1; }
    std::printf("ink_widen_test: all checks passed\n");
    return 0;
}
