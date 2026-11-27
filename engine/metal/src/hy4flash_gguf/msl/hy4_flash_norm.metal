// engine/metal/src/hy4flash_gguf/msl/hy4_flash_norm.metal - RMSNorm with an optional weight, f32 throughout.
//
//   y[t][i] = x[t][i] * rsqrt(mean_i(x[t][i]^2) + eps) * w[i]
//
// The spec is the sum of squares in f32 over a fixed reduction tree and rsqrt(mean + eps) - rsqrt, not 1/sqrt, so
// the kernel is written as that.  HY4_FLASH_NORM_W = false drops the weight: that is the iHC flat norm (over all
// hc * n_embd = 12288 values jointly, semantics.md §3) and the Engram row norm's bare form.
//
// One thread group per row, so nothing about the tree depends on how many tokens the dispatch carries.

struct Hy4FlashNormArgs {
    uint n;          // elements in a row
    uint x_stride;   // floats between rows of x
    uint y_stride;   // floats between rows of y
    float eps;
};

// function constant 40: the hy4-flash kernels share one library with quant/units/gemv.metal, which already use 0, 9, 23
// and 25, so hy4-flash's constants start at 40.
constant bool HY4_FLASH_NORM_W [[function_constant(40)]];
constant bool hy4_flash_norm_w = is_function_constant_defined(HY4_FLASH_NORM_W) ? HY4_FLASH_NORM_W : true;

kernel void hy4_flash_rmsnorm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                        device float* y [[buffer(2)]], constant Hy4FlashNormArgs& a [[buffer(3)]],
                        uint3 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    device const float* xr = x + (ulong)tgp.x * a.x_stride;
    device float* yr = y + (ulong)tgp.x * a.y_stride;
    threadgroup float part[HY4_FLASH_TG_SG];
    float s = 0.0f;
    for (uint i = tid; i < a.n; i += HY4_FLASH_TG) {
        const float v = xr[i];
        s += v * v;
    }
    const float scale = precise::rsqrt(hy4_flash_tg_sum(s, part, sg, lane) / float(a.n) + a.eps);
    for (uint i = tid; i < a.n; i += HY4_FLASH_TG) {
        const float v = xr[i] * scale;
        yr[i] = hy4_flash_norm_w ? v * w[i] : v;
    }
}

// ---- the same norm over more cores ----------------------------------------------------------------------------------
// hy4_flash_rmsnorm is one thread group per row, so a decode row (T <= 8 of them) runs on one or a few of the GPU's cores
// and pays the whole row's latency there twice (sum, then scale).  These two kernels compute the SAME value with the
// row spread out, by splitting hy4_flash_rmsnorm along its own reduction tree:
//   hy4_flash_rms_part    grid (HY4_FLASH_TG_SG, rows), 32 threads.  Thread group g is hy4_flash_rmsnorm's simdgroup g: thread `lane`
//                   is hy4_flash_rmsnorm's thread 32 g + lane, sums the same elements in the same order, and the group's
//                   butterfly (hy4_flash_simd_sum) gives that simdgroup's partial, written to part[row][g];
//   hy4_flash_rms_apply   grid (ceil(n / HY4_FLASH_TG), rows), HY4_FLASH_TG threads.  Every group adds the HY4_FLASH_TG_SG partials in
//                   index order - hy4_flash_tg_sum's loop - takes the same rsqrt(mean + eps) and scales its slice.
// So every output is x * scale (* w) with the scale of hy4_flash_rmsnorm, bit for bit.
kernel void hy4_flash_rms_part(device const float* x [[buffer(0)]], device float* part [[buffer(1)]],
                         constant Hy4FlashNormArgs& a [[buffer(2)]], uint3 tgp [[threadgroup_position_in_grid]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    device const float* xr = x + (ulong)tgp.y * a.x_stride;
    const uint tid = tgp.x * 32u + lane;
    float s = 0.0f;
    for (uint i = tid; i < a.n; i += HY4_FLASH_TG) {
        const float v = xr[i];
        s += v * v;
    }
    s = hy4_flash_simd_sum(s);
    if (lane == 0) part[tgp.y * HY4_FLASH_TG_SG + tgp.x] = s;
}

kernel void hy4_flash_rms_apply(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                          device float* y [[buffer(2)]], constant Hy4FlashNormArgs& a [[buffer(3)]],
                          device const float* part [[buffer(4)]], uint3 tgp [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]]) {
    device const float* pr = part + tgp.y * HY4_FLASH_TG_SG;
    float tot = 0.0f;
    for (ushort k = 0; k < HY4_FLASH_TG_SG; ++k) tot += pr[k];
    const float scale = precise::rsqrt(tot / float(a.n) + a.eps);
    const uint i = tgp.x * HY4_FLASH_TG + tid;
    if (i >= a.n) return;
    const float v = x[(ulong)tgp.y * a.x_stride + i] * scale;
    y[(ulong)tgp.y * a.y_stride + i] = hy4_flash_norm_w ? v * w[i] : v;
}
