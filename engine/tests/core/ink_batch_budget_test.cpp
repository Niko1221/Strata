// Placement must leave room for the N-slot K/V and the batch scratch before the card is filled with experts (INKOTHER, boxia RX 7900 XT:
// `--bench --parallel 32` ended with "[ink-hip] scratch alloc: out of memory" because every expert was placed first). No GPU.
#include "ink_batch.h"

#include <cstdio>

using namespace ink2;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)
constexpr size_t GiB = 1ull << 30;

// the allocation list of the HIP backend's ensure_batch_scratch(B), one buffer per line, as the backend writes it
size_t listed(const BatchGeom& g, size_t B) {
    const size_t D = g.hidden, q = (size_t)g.n_head * g.head_dim, kv = (size_t)g.n_kv * g.head_dim, r = (size_t)g.n_head * g.d_rel;
    const size_t qkvrg = 2 * q + r + 2 * kv, inter = g.inter, ne = g.n_experts, tk = g.top_k;
    const size_t cap = std::max<size_t>(B, std::min<size_t>(B * tk, 512));
    const size_t xk = std::max<size_t>({2 * D, q, 4096});
    size_t t = 0;
    t += B * 4 * 7 + (B + 1) * 4;                    // slot ids, pos, tok ids, img idx, want idx, grp slot, grp tok, grp off
    t += std::max<size_t>(B, 1025) * xk * 2;         // x bf16
    t += B * qkvrg * 4;
    for (int i = 0; i < 5; ++i) t += B * D * 4;      // cur, res, in0, in1, in2
    t += B * 3 * D * 4 + B * 4 * D * 4;              // mx3, msgs
    t += B * D * 4 + B * 2 * D * 4 + 2 * (B * D * 4);// ctrl tmp, ctrl in, ctrl gain, ctrl forget
    t += B * q * 4;                                  // q
    t += 4 * (B * kv * 4);                           // k raw, v raw, k conv, v conv
    t += B * r * 4;                                  // r raw
    t += B * g.n_head * (size_t)g.rel_extent * 4;    // rel bias
    t += B * g.n_head * 4;                           // sinks
    t += B * q * 4;                                  // gate a
    t += 2 * (B * g.n_kv * 4 * 4);                   // k dyn, v dyn
    t += B * q * 4;                                  // attn out
    t += std::max<size_t>(B, 4) * g.n_head * 4 * 130 * 4;   // split-KV partials
    t += B * D * 4;                                  // o proj
    t += B * D * 4 + B * D * 4;                      // a sconv, mlp in
    t += cap * D * 2;                                // mlp f16
    t += B * ne * 4;                                 // r logits
    t += 2 * (B * tk * 4);                           // topk ids, topk w
    t += B * tk * inter * 2;                         // moe act
    t += cap * 2 * inter * 4 + cap * D * 4;          // moe gu, moe ye
    t += 129 * 4 + 128 * 4;                          // exp off, exp cur
    t += B * D + B * (D >> 4) * 4;                   // x q8, x d
    t += B * tk * inter + B * tk * (inter >> 4) * 4; // act q8, act d
    t += B * tk * 4 + B * tk * 4;                    // bucket tok, slotk bidx
    t += B * tk * D * 2 + B * tk * D * 4;            // bucket x f16, bucket y
    t += B * D * 4 + B * D * 4;                      // moe out, m sconv
    t += 2 * (cap * 4);                              // exp tok idx, exp tok w
    t += B * D * 4;                                  // h norm
    t += 2 * (B * D * 4);                            // prelude, bank out
    t += 8 * (B * D * 4) + 4 * (B * D * 4) + 4 * (B * D * 4);   // depth blocks, bank, taps
    return t;
}
}  // namespace

int main() {
    const BatchGeom g;
    for (int B : {1, 4, 32, 64, 512, 1024}) CHECK(estimate_batch_scratch_bytes(g, B) == listed(g, (size_t)B));
    CHECK(estimate_batch_scratch_bytes(g, 32) > (16u << 20) && estimate_batch_scratch_bytes(g, 32) < (512u << 20));
    CHECK(estimate_batch_scratch_bytes(g, 512) > estimate_batch_scratch_bytes(g, 32));
    CHECK(estimate_batch_scratch_bytes(g, 1024) > estimate_batch_scratch_bytes(g, 512));

    // the boxia case: a 20 GiB card, 32 slots at 4096 tokens of FP16 K/V, the trunk in the card
    const size_t card = 20 * GiB, reserve = 1 * GiB, trunk = 3 * GiB, margin = 768ull << 20;
    const size_t per_slot = estimate_slot_bytes(4096, KvDtype::FP16, false).total_bytes();
    const size_t kv32 = per_slot * 32;
    const size_t scratch = estimate_batch_scratch_bytes(g, 512);
    const size_t old_budget = card - (reserve + trunk + std::min<size_t>(kv32, card / 4) + margin);   // the earlier rule: K/V estimate capped at a quarter of the card
    const size_t new_budget = plan_expert_vram_budget(card, reserve, trunk, kv32, scratch, margin);
    CHECK(kv32 > card / 4);                                       // the cap really cut the estimate
    CHECK(new_budget < old_budget);
    CHECK(new_budget + reserve + trunk + kv32 + scratch + margin <= card);   // everything fits
    // with the old budget filled with experts the 32 slots (K/V + scratch) no longer fit: this was the failure
    CHECK(card - (old_budget + reserve + trunk) < kv32 + scratch);
    // with the new one they do
    CHECK(card - (new_budget + reserve + trunk) >= kv32 + scratch);
    // more slots, fewer experts; one slot: almost the whole card
    CHECK(plan_expert_vram_budget(card, reserve, trunk, per_slot * 4, estimate_batch_scratch_bytes(g, 512), margin) > new_budget);
    CHECK(plan_expert_vram_budget(card, reserve, trunk, per_slot, estimate_batch_scratch_bytes(g, 512), margin) > plan_expert_vram_budget(card, reserve, trunk, per_slot * 4, scratch, margin));
    // a card that cannot hold even the overhead gives 0 (the experts go to RAM), never a wrapped-around size
    CHECK(plan_expert_vram_budget(2 * GiB, reserve, trunk, kv32, scratch, margin) == 0);
    if (failed) { std::fprintf(stderr, "%d checks failed\n", failed); return 1; }
    std::printf("ink_batch_budget_test OK\n");
    return 0;
}
