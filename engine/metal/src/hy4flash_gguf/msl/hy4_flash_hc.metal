// engine/metal/src/hy4flash_gguf/msl/hy4_flash_hc.metal - the hyper-connection gates and the two mixes, x4 streams (semantics.md §3).
//
// The inter-block state is HY4_FLASH_HC = 4 streams of n_embd, contiguous per token, so a block hands on 12288 values.
// Each block runs this twice (before attention, before the FFN) and the model once more for the output head.
//
//   flat      = x as [hc * n_embd]                     (the caller's hy4_flash_rmsnorm with HY4_FLASH_NORM_W = false)
//   flat_norm = rms_norm(flat, eps)                    over ALL hc * n_embd jointly, NOT per stream
//   mixes     = fn^T . flat_norm                       (the caller's dense GEMV; fn is Q4_K [12288, 2 * hc])
//   pre [i]   = sigmoid(mixes[i]      * scale[0] + base[i])      + hc_eps
//   post[i]   = sigmoid(mixes[hc + i] * scale[1] + base[hc + i]) * magnitude + hc_eps
//   cur       = sum_i x[i] * pre[i]                    the ORIGINAL x, not flat_norm
//   out[i]    = residual[i] + y * post[i]              each stream keeps its own residual
//
// Four traps semantics.md §3 names, all of them here: the reduce consumes the original x; the RMS is over the whole
// 12288 row; `pre` has no magnitude while `post` applies it BEFORE hc_eps; and `fn` is Q4_K.
// HY4_FLASH_HC_POST_GATE = false is the output head's form (`output_hc_fn` is [12288, hc], a single scale, no post gate).
//
// The gates are computed redundantly by every thread of the group from the same inputs - 4 or 8 sigmoids, and it
// saves a barrier.  The reduce sums the streams in index order 0..hc-1 with contraction off, so it does not depend
// on the window.

#define HY4_FLASH_HC 4

struct Hy4FlashHcArgs {
    uint n_embd;
    uint x_stride;     // floats between tokens of x (hc * n_embd when packed)
    uint mix_stride;   // floats between tokens of mixes (2 * hc, or hc for the head)
    uint cur_stride;
    float hc_eps;
    float hc_mag;      // hyper_connection.magnitude (2.0); unused when HY4_FLASH_HC_POST_GATE is false
};

constant bool HY4_FLASH_HC_POST_GATE [[function_constant(41)]];   // see hy4_flash_norm.metal on the index range
constant bool hy4_flash_hc_post_gate = is_function_constant_defined(HY4_FLASH_HC_POST_GATE) ? HY4_FLASH_HC_POST_GATE : true;

// grid (n_tok, 1, 1), HY4_FLASH_TG threads.  pre and post are written as [n_tok][hc] (post only with the gate on); they
// are needed later by hy4_flash_hc_post and are small enough to keep.
kernel void hy4_flash_hc_mix(device const float* x [[buffer(0)]], device const float* mixes [[buffer(1)]],
                       device const float* scale [[buffer(2)]], device const float* base [[buffer(3)]],
                       device float* pre_out [[buffer(4)]], device float* post_out [[buffer(5)]],
                       device float* cur [[buffer(6)]], constant Hy4FlashHcArgs& a [[buffer(7)]],
                       uint3 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    device const float* m = mixes + (ulong)tgp.x * a.mix_stride;
    float pre[HY4_FLASH_HC];
    for (uint i = 0; i < HY4_FLASH_HC; ++i) pre[i] = hy4_flash_sigmoid(m[i] * scale[0] + base[i]) + a.hc_eps;
    if (tid == 0)
        for (uint i = 0; i < HY4_FLASH_HC; ++i) {
            pre_out[tgp.x * HY4_FLASH_HC + i] = pre[i];
            if (hy4_flash_hc_post_gate)
                post_out[tgp.x * HY4_FLASH_HC + i] =
                    hy4_flash_sigmoid(m[HY4_FLASH_HC + i] * scale[1] + base[HY4_FLASH_HC + i]) * a.hc_mag + a.hc_eps;
        }
    device const float* xr = x + (ulong)tgp.x * a.x_stride;
    device float* cr = cur + (ulong)tgp.x * a.cur_stride;
    for (uint d = tid; d < a.n_embd; d += HY4_FLASH_TG) {
        float s = 0.0f;
        for (uint i = 0; i < HY4_FLASH_HC; ++i) s += xr[(ulong)i * a.n_embd + d] * pre[i];
        cr[d] = s;
    }
}

// out[t][i][d] = residual[t][i][d] + y[t][d] * post[t][i].  grid (ceil(n_embd) , hc, n_tok).
// `out` may alias `residual` (each thread touches one element).
kernel void hy4_flash_hc_post(device const float* y [[buffer(0)]], device const float* residual [[buffer(1)]],
                        device const float* post [[buffer(2)]], device float* out [[buffer(3)]],
                        constant Hy4FlashHcArgs& a [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.n_embd) return;
    const ulong o = (ulong)gid.z * a.x_stride + (ulong)gid.y * a.n_embd + gid.x;
    out[o] = residual[o] + y[(ulong)gid.z * a.cur_stride + gid.x] * post[gid.z * HY4_FLASH_HC + gid.y];
}

// ---- hy4_flash_hc_mix followed by the block's RMSNorm of `cur`, in ONE dispatch.  Bitwise the pair of dispatches
// (hy4_flash_hc_mix, then hy4_flash_rmsnorm with the weight): the mix is hy4_flash_hc_mix's loop, op for op, and each thread then
// norms exactly the elements it wrote, in the order hy4_flash_rmsnorm's thread `tid` takes them (d = tid, tid + HY4_FLASH_TG,
// ...: the same 256-thread group, so the same per-thread sums and the same hy4_flash_tg_sum tree), reading back its own
// stores.  `cur` is still written (the dump and nothing else reads it later).  grid (n_tok, 1, 1), HY4_FLASH_TG threads.
struct Hy4FlashHcNormArgs {
    uint y_stride;   // floats between tokens of y
    float eps;
};
kernel void hy4_flash_hc_mix_norm(device const float* x [[buffer(0)]], device const float* mixes [[buffer(1)]],
                            device const float* scale [[buffer(2)]], device const float* base [[buffer(3)]],
                            device float* pre_out [[buffer(4)]], device float* post_out [[buffer(5)]],
                            device float* cur [[buffer(6)]], constant Hy4FlashHcArgs& a [[buffer(7)]],
                            device const float* w [[buffer(8)]], device float* y [[buffer(9)]],
                            constant Hy4FlashHcNormArgs& na [[buffer(10)]], uint3 tgp [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    device const float* m = mixes + (ulong)tgp.x * a.mix_stride;
    float pre[HY4_FLASH_HC];
    for (uint i = 0; i < HY4_FLASH_HC; ++i) pre[i] = hy4_flash_sigmoid(m[i] * scale[0] + base[i]) + a.hc_eps;
    if (tid == 0)
        for (uint i = 0; i < HY4_FLASH_HC; ++i) {
            pre_out[tgp.x * HY4_FLASH_HC + i] = pre[i];
            if (hy4_flash_hc_post_gate)
                post_out[tgp.x * HY4_FLASH_HC + i] =
                    hy4_flash_sigmoid(m[HY4_FLASH_HC + i] * scale[1] + base[HY4_FLASH_HC + i]) * a.hc_mag + a.hc_eps;
        }
    device const float* xr = x + (ulong)tgp.x * a.x_stride;
    device float* cr = cur + (ulong)tgp.x * a.cur_stride;
    float ss = 0.0f;
    for (uint d = tid; d < a.n_embd; d += HY4_FLASH_TG) {
        float s = 0.0f;
        for (uint i = 0; i < HY4_FLASH_HC; ++i) s += xr[(ulong)i * a.n_embd + d] * pre[i];
        cr[d] = s;
        ss += s * s;   // hy4_flash_rmsnorm: `const float v = xr[i]; s += v * v;`
    }
    threadgroup float part[HY4_FLASH_TG_SG];
    const float sc = precise::rsqrt(hy4_flash_tg_sum(ss, part, sg, lane) / float(a.n_embd) + na.eps);
    device float* yr = y + (ulong)tgp.x * na.y_stride;
    for (uint d = tid; d < a.n_embd; d += HY4_FLASH_TG) {
        const float v = cr[d] * sc;   // this thread's own store above
        yr[d] = v * w[d];
    }
}
