// engine/metal/src/hy4flash_gguf/msl/hy4_flash_common.metal - the pieces every hy4-flash kernel shares.
//
// Why an own file instead of reusing quant.metal's helpers: hy4-flash must give BITWISE the same value for a position
// whether it is computed in a decode window of 1..8 tokens or in a prefill block of up to 1024 ("T invariance").
// Two rules follow, and both are structural here:
//
//   * no reduction may depend on the number of tokens in the dispatch.  Every kernel below owns one row (or one
//     element) per thread group, with a thread count that is a compile-time constant, so the summation tree is a
//     function of the row width alone;
//   * no sum may be reassociated.  The libraries are compiled with safe math locked (mtl_rt.h lock_safe_math(), so
//     STRATA_METAL_MATH=fast is an error, not a downgrade), and FMA contraction is off for the WHOLE library -
//     hy4_flash_fp.metal, first in hy4_flash_kernel_files(), does that; the pragma is repeated here so this file stays
//     self-contained if it is ever compiled on its own.
//
// simd_sum() is not used BY THE KERNELS IN THIS FILE OR BY hy4-flash'S OWN KERNELS: Metal does not specify its order, so
// a butterfly of simd_shuffle_xor() is written out instead (hy4_flash_simd_sum / hy4_flash_lane_sum below), which fixes the
// order on every device and in every window size.  Note what that does NOT cover: the dense GEMV reuses the shared
// gemv.metal, whose lane reduction is quant.metal's lane_sum<LPR>, and lane_sum<32> IS simd_sum - which the frozen
// wide tile {32, 1, 8, 2} selects, so the 12288-wide iHC matvec and the LM head go through it.  There, stability
// across window sizes is a property the T-invariance tests check on this device, not a specified order.
#pragma METAL fp contract(off)

#define HY4_FLASH_TG 256                 // threads per thread group for the row kernels
#define HY4_FLASH_TG_SG (HY4_FLASH_TG / 32)    // simdgroups in one of them

// Sum of v over the 32 lanes of a simdgroup: 5 fixed butterfly steps, the result valid in every lane.
static inline float hy4_flash_simd_sum(float v) {
    for (ushort o = 16; o > 0; o >>= 1) v += simd_shuffle_xor(v, o);
    return v;
}

// Sum of one value per thread over a HY4_FLASH_TG thread group: the butterfly, then the HY4_FLASH_TG_SG partials added in index
// order.  Needs `part` in threadgroup memory; the result is valid in every thread.
static inline float hy4_flash_tg_sum(float v, threadgroup float* part, ushort sg, ushort lane) {
    v = hy4_flash_simd_sum(v);
    if (lane == 0) part[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (ushort k = 0; k < HY4_FLASH_TG_SG; ++k) tot += part[k];
    return tot;
}

// The same sum over groups of LPR consecutive lanes (LPR = 1, 2, 4, 8, 16 or 32), the result valid in every lane.
// quant.metal's lane_sum<32> calls simd_sum(), whose order Metal does not specify, so the GEMV-shaped hy4-flash kernels
// use this instead: the butterfly is written out for every LPR, including 32.
// `rev` runs the butterfly from the smallest stride up - a DIFFERENT summation order, and the only use is the
// negative control of the T-invariance tests (HY4_FLASH_MOE_PROBE = 1); production always passes false.
template <int LPR>
static inline float hy4_flash_lane_sum(float v, bool rev) {
    if (rev) {
        for (ushort o = 1; o < LPR; o <<= 1) v += simd_shuffle_xor(v, o);
        return v;
    }
    for (ushort o = LPR / 2; o > 0; o >>= 1) v += simd_shuffle_xor(v, o);
    return v;
}

// Sum of one uint per thread over a HY4_FLASH_TG thread group, same shape as hy4_flash_tg_sum.  Integer, so the order does not
// matter for the value; the barriers are what make it safe to call in a loop (the indexer's top-k search does).
static inline uint hy4_flash_tg_sum_u(uint v, threadgroup uint* part, ushort sg, ushort lane) {
    for (ushort o = 16; o > 0; o >>= 1) v += simd_shuffle_xor(v, o);
    threadgroup_barrier(mem_flags::mem_threadgroup);   // every reader of the previous round is done
    if (lane == 0) part[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint tot = 0;
    for (ushort k = 0; k < HY4_FLASH_TG_SG; ++k) tot += part[k];
    return tot;
}

// Which of the three M-RoPE planes pair `k` takes.
//
// The authority INTERLEAVES the planes rather than blocking them, and that is not a detail: blocking would
// give pairs 0-11 the time plane, 12-21 height, 22-31 width, which is a different rotation for 20 of the 32
// pairs.  transformers PR #50903's `recomposition_frequencies` starts from the time plane everywhere and
// then overwrites `slice(1, 3 * section[1], 3)` with height and `slice(2, 3 * section[2], 3)` with width -
// so pair k takes height when k % 3 == 1 and k < 3 * section[1], width when k % 3 == 2 and
// k < 3 * section[2], and time otherwise.
//
// The counts are the check on that reading: with section = [12, 10, 10] and 32 pairs it selects height for
// {1, 4, ... 28} and width for {2, 5, ... 29} - ten each - leaving twelve for time, including pair 31,
// whose k % 3 == 1 falls outside the length.  Exactly the section the config declares.
static inline uint hy4_flash_mrope_axis(uint k, uint len1, uint len2) {
    const uint r = k % 3;
    if (r == 1 && k < len1) return 1;
    if (r == 2 && k < len2) return 2;
    return 0;
}

// sigmoid as the reference writes it (1 / (1 + exp(-x)); hy4_block.py sigmoid, semantics.md §3).
static inline float hy4_flash_sigmoid(float x) { return 1.0f / (1.0f + precise::exp(-x)); }

// SiLU as the reference writes it: x / (1 + exp(-x)), NOT x * sigmoid(x) (hy4_block.py silu).
static inline float hy4_flash_silu(float x) { return x / (1.0f + precise::exp(-x)); }
