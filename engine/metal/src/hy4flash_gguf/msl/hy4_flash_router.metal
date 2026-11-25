// engine/metal/src/hy4flash_gguf/msl/hy4_flash_router.metal - the MoE router of a hy4-flash trunk block (semantics.md §2).
//
//   probs  = sigmoid(ffn_gate_inp . cur)                 the FULL-WIDTH 3072 cur, 512 logits
//   biased = probs + exp_probs_b                         what the SELECTION ranks
//   sel    = the top `topk` (16) of biased, ties to the SMALLER expert id
//   w[r]   = probs[sel[r]] / max(sum_r probs[sel[r]], 2^-14) * 3.998
//
// Two things make this its own kernel rather than a reuse of moe.metal's moe_route, for the same reason the MoE
// GEMVs are:
//
//   * moe_route is softmax over the logits and top TEN by probability, both written as compile-time facts
//     (`route_tok<NL>`, `rank < 10`).  hy4-flash is sigmoid, top SIXTEEN, and the rank key is the BIASED probability
//     while the weight is the UNBIASED one - which of the two feeds the weight is not a detail (semantics.md §2);
//   * the sum of the selected probabilities is taken IN SELECTION ORDER here (rank 0 then 1 then ...).  moe_route
//     accumulates it on whichever lane won and then folds the lanes with an xor tree, which is a different order.
//     hy4-flash needs one order that holds for every window size.
//
// One thread group per token, so nothing depends on how many tokens the dispatch carries: the sigmoid pass is
// spread over HY4_FLASH_TG threads, and simdgroup 0 alone runs the rank loop and the weight tail.
//
// The floor is on the SUM, not on the weights (hy4_block.py latent_moe: w / maximum(w.sum(), 6.103515625e-5)).

#define HY4_FLASH_ROUTER_MAXE 1024   // experts a token can be routed over (512 here); one threadgroup array each
#define HY4_FLASH_ROUTER_MAXK 64     // top-k slots (16 here)

struct Hy4FlashRouterArgs {
    uint n_expert;
    uint topk;
    uint l_stride;     // floats between tokens of the logits
    uint id_stride;    // ints between tokens of ids
    uint w_stride;     // floats between tokens of w
    uint n_tok;
    float w_scale;     // expert_weights_scale (3.998)
    float floor;       // 2^-14 = 6.103515625e-5
};

// Test hook (the convention of gemv.metal's GEMV_PROBE): 1 = ties go to the LARGER expert id.  The tie test's
// negative control - with it the "all scores equal -> 0..topk-1" case must fail.  Unset in production.
constant int HY4_FLASH_ROUTER_PROBE [[function_constant(50)]];
constant int hy4_flash_router_probe = is_function_constant_defined(HY4_FLASH_ROUTER_PROBE) ? HY4_FLASH_ROUTER_PROBE : 0;

kernel void hy4_flash_router(device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]],
                       device int* ids [[buffer(2)]], device float* w [[buffer(3)]],
                       constant Hy4FlashRouterArgs& a [[buffer(4)]], uint3 tgp [[threadgroup_position_in_grid]],
                       ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float tb[HY4_FLASH_ROUTER_MAXE];   // probs + bias: the ranking key, overwritten with -inf once taken
    threadgroup float tp[HY4_FLASH_ROUTER_MAXE];   // the unbiased probs: the weights' source
    threadgroup float sp[HY4_FLASH_ROUTER_MAXK];   // the selected probability, by rank
    threadgroup int si[HY4_FLASH_ROUTER_MAXK];     // the selected expert id, by rank
    device const float* lr = logits + (ulong)tgp.x * a.l_stride;
    for (uint e = tid; e < a.n_expert; e += HY4_FLASH_TG) {
        const float p = hy4_flash_sigmoid(lr[e]);
        tp[e] = p;
        tb[e] = p + bias[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    const bool hi = hy4_flash_router_probe == 1;
    const int none = hi ? -1 : 0x7FFFFFFF;   // the id a lane with no candidate carries, so a tie never picks it
    for (uint r = 0; r < a.topk; ++r) {
        float best = -INFINITY;
        int bi = none;
        // ascending e, strict >: within a lane's stride the FIRST of equal values wins, i.e. the smaller id
        for (uint e = lane; e < a.n_expert; e += 32) {
            const float v = tb[e];
            if (v > best || (v == best && hi)) {
                best = v;
                bi = int(e);
            }
        }
        for (ushort o = 16; o > 0; o >>= 1) {
            const float ob = simd_shuffle_xor(best, o);
            const int oi = simd_shuffle_xor(bi, o);
            if (ob > best || (ob == best && (hi ? oi > bi : oi < bi))) {
                best = ob;
                bi = oi;
            }
        }
        if (lane == 0) {
            si[r] = bi;
            sp[r] = tp[bi];
            tb[bi] = -INFINITY;   // taken; the next rank must not see it
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        float s = 0.0f;
        for (uint r = 0; r < a.topk; ++r) s += sp[r];   // selection order
        s = max(s, a.floor);
        device int* ir = ids + (ulong)tgp.x * a.id_stride;
        device float* wr = w + (ulong)tgp.x * a.w_stride;
        for (uint r = 0; r < a.topk; ++r) {
            ir[r] = si[r];
            wr[r] = sp[r] / s * a.w_scale;   // the reference's two steps, in its order
        }
    }
}
