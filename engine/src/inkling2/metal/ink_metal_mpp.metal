// ink_metal_mpp.metal - Metal 4 Neural Accelerator (mpp::tensor_ops::matmul2d) Kernels for Inkling-2-Mini
// Compiled dynamically on Apple Silicon GPUs with Metal 4 / MPP support (e.g., Apple M5 Max);
// older macOS / M1-M4 Macs gracefully fall back to simdgroup_matrix kernels in ink_metal_codecs.metal.
// Implements:
//   1. Native FP16 matmul2d Dense Trunk & Concatenated QKV/R/Gate/O/Controller/DSpark GEMMs
//   2. Hybrid Decompress-Once + Staged Grouped-Expert MoE Prefill/Batching (SF4, MX8, S2_K..S8_K + fused SITU)
//   3. Tiled matmul2d FlashAttention-2 Prefill (BQ=32, BK=64) with L1-hot RelPos bias + Deferred-Sink Online Softmax
// 100% MIT clean-room implementation.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// ---- Shared codec helpers (kept self-contained so ink_metal_mpp.metal compiles standalone!) ----
constant float kMppE2M1[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static inline float mpp_e4m3_to_f32(uint c) {
    const uint mag = c & 0x7Fu;
    if (mag == 0x7Fu) return 0.0f;
    const uint exp = mag >> 3;
    const uint man = mag & 7u;
    float v = (exp == 0u) ? (float(man) * (1.0f / 512.0f))
                          : as_type<float>(((exp + 120u) << 23) | (man << 20));
    return (c & 0x80u) ? -v : v;
}

static inline float mpp_ue8m0_to_f32(uint e) {
    if (e == 0u) return 5.877471754111438e-39f;
    if (e == 255u) return 0.0f;
    return as_type<float>(e << 23);
}

static inline float mpp_bf16_to_f32(ushort b) { return as_type<float>(uint(b) << 16); }

static inline ushort mpp_ld16(device const uchar* p) { return ushort(p[0]) | (ushort(p[1]) << 8); }

static inline constexpr int mpp_sk_sb_bytes(int F) {
    return (F == 2) ? 70 : (F == 3) ? 102 : (F == 4) ? 134 : (F == 5) ? 166 : (F == 6) ? 198 : 262;
}
static inline constexpr float mpp_sk_half(int F) {
    return (F == 2) ? 1.5f : (F == 3) ? 3.5f : (F == 4) ? 7.5f : (F == 5) ? 15.5f : (F == 6) ? 31.5f : 0.0f;
}

constant half4 kMppS2Lut[256] = {
    half4(-1.5h,-1.5h,-1.5h,-1.5h), half4(-0.5h,-1.5h,-1.5h,-1.5h), half4(0.5h,-1.5h,-1.5h,-1.5h), half4(1.5h,-1.5h,-1.5h,-1.5h),
    half4(-1.5h,-0.5h,-1.5h,-1.5h), half4(-0.5h,-0.5h,-1.5h,-1.5h), half4(0.5h,-0.5h,-1.5h,-1.5h), half4(1.5h,-0.5h,-1.5h,-1.5h),
    half4(-1.5h,0.5h,-1.5h,-1.5h), half4(-0.5h,0.5h,-1.5h,-1.5h), half4(0.5h,0.5h,-1.5h,-1.5h), half4(1.5h,0.5h,-1.5h,-1.5h),
    half4(-1.5h,1.5h,-1.5h,-1.5h), half4(-0.5h,1.5h,-1.5h,-1.5h), half4(0.5h,1.5h,-1.5h,-1.5h), half4(1.5h,1.5h,-1.5h,-1.5h),
    half4(-1.5h,-1.5h,-0.5h,-1.5h), half4(-0.5h,-1.5h,-0.5h,-1.5h), half4(0.5h,-1.5h,-0.5h,-1.5h), half4(1.5h,-1.5h,-0.5h,-1.5h),
    half4(-1.5h,-0.5h,-0.5h,-1.5h), half4(-0.5h,-0.5h,-0.5h,-1.5h), half4(0.5h,-0.5h,-0.5h,-1.5h), half4(1.5h,-0.5h,-0.5h,-1.5h),
    half4(-1.5h,0.5h,-0.5h,-1.5h), half4(-0.5h,0.5h,-0.5h,-1.5h), half4(0.5h,0.5h,-0.5h,-1.5h), half4(1.5h,0.5h,-0.5h,-1.5h),
    half4(-1.5h,1.5h,-0.5h,-1.5h), half4(-0.5h,1.5h,-0.5h,-1.5h), half4(0.5h,1.5h,-0.5h,-1.5h), half4(1.5h,1.5h,-0.5h,-1.5h),
    half4(-1.5h,-1.5h,0.5h,-1.5h), half4(-0.5h,-1.5h,0.5h,-1.5h), half4(0.5h,-1.5h,0.5h,-1.5h), half4(1.5h,-1.5h,0.5h,-1.5h),
    half4(-1.5h,-0.5h,0.5h,-1.5h), half4(-0.5h,-0.5h,0.5h,-1.5h), half4(0.5h,-0.5h,0.5h,-1.5h), half4(1.5h,-0.5h,0.5h,-1.5h),
    half4(-1.5h,0.5h,0.5h,-1.5h), half4(-0.5h,0.5h,0.5h,-1.5h), half4(0.5h,0.5h,0.5h,-1.5h), half4(1.5h,0.5h,0.5h,-1.5h),
    half4(-1.5h,1.5h,0.5h,-1.5h), half4(-0.5h,1.5h,0.5h,-1.5h), half4(0.5h,1.5h,0.5h,-1.5h), half4(1.5h,1.5h,0.5h,-1.5h),
    half4(-1.5h,-1.5h,1.5h,-1.5h), half4(-0.5h,-1.5h,1.5h,-1.5h), half4(0.5h,-1.5h,1.5h,-1.5h), half4(1.5h,-1.5h,1.5h,-1.5h),
    half4(-1.5h,-0.5h,1.5h,-1.5h), half4(-0.5h,-0.5h,1.5h,-1.5h), half4(0.5h,-0.5h,1.5h,-1.5h), half4(1.5h,-0.5h,1.5h,-1.5h),
    half4(-1.5h,0.5h,1.5h,-1.5h), half4(-0.5h,0.5h,1.5h,-1.5h), half4(0.5h,0.5h,1.5h,-1.5h), half4(1.5h,0.5h,1.5h,-1.5h),
    half4(-1.5h,1.5h,1.5h,-1.5h), half4(-0.5h,1.5h,1.5h,-1.5h), half4(0.5h,1.5h,1.5h,-1.5h), half4(1.5h,1.5h,1.5h,-1.5h),
    half4(-1.5h,-1.5h,-1.5h,-0.5h), half4(-0.5h,-1.5h,-1.5h,-0.5h), half4(0.5h,-1.5h,-1.5h,-0.5h), half4(1.5h,-1.5h,-1.5h,-0.5h),
    half4(-1.5h,-0.5h,-1.5h,-0.5h), half4(-0.5h,-0.5h,-1.5h,-0.5h), half4(0.5h,-0.5h,-1.5h,-0.5h), half4(1.5h,-0.5h,-1.5h,-0.5h),
    half4(-1.5h,0.5h,-1.5h,-0.5h), half4(-0.5h,0.5h,-1.5h,-0.5h), half4(0.5h,0.5h,-1.5h,-0.5h), half4(1.5h,0.5h,-1.5h,-0.5h),
    half4(-1.5h,1.5h,-1.5h,-0.5h), half4(-0.5h,1.5h,-1.5h,-0.5h), half4(0.5h,1.5h,-1.5h,-0.5h), half4(1.5h,1.5h,-1.5h,-0.5h),
    half4(-1.5h,-1.5h,-0.5h,-0.5h), half4(-0.5h,-1.5h,-0.5h,-0.5h), half4(0.5h,-1.5h,-0.5h,-0.5h), half4(1.5h,-1.5h,-0.5h,-0.5h),
    half4(-1.5h,-0.5h,-0.5h,-0.5h), half4(-0.5h,-0.5h,-0.5h,-0.5h), half4(0.5h,-0.5h,-0.5h,-0.5h), half4(1.5h,-0.5h,-0.5h,-0.5h),
    half4(-1.5h,0.5h,-0.5h,-0.5h), half4(-0.5h,0.5h,-0.5h,-0.5h), half4(0.5h,0.5h,-0.5h,-0.5h), half4(1.5h,0.5h,-0.5h,-0.5h),
    half4(-1.5h,1.5h,-0.5h,-0.5h), half4(-0.5h,1.5h,-0.5h,-0.5h), half4(0.5h,1.5h,-0.5h,-0.5h), half4(1.5h,1.5h,-0.5h,-0.5h),
    half4(-1.5h,-1.5h,0.5h,-0.5h), half4(-0.5h,-1.5h,0.5h,-0.5h), half4(0.5h,-1.5h,0.5h,-0.5h), half4(1.5h,-1.5h,0.5h,-0.5h),
    half4(-1.5h,-0.5h,0.5h,-0.5h), half4(-0.5h,-0.5h,0.5h,-0.5h), half4(0.5h,-0.5h,0.5h,-0.5h), half4(1.5h,-0.5h,0.5h,-0.5h),
    half4(-1.5h,0.5h,0.5h,-0.5h), half4(-0.5h,0.5h,0.5h,-0.5h), half4(0.5h,0.5h,0.5h,-0.5h), half4(1.5h,0.5h,0.5h,-0.5h),
    half4(-1.5h,1.5h,0.5h,-0.5h), half4(-0.5h,1.5h,0.5h,-0.5h), half4(0.5h,1.5h,0.5h,-0.5h), half4(1.5h,1.5h,0.5h,-0.5h),
    half4(-1.5h,-1.5h,1.5h,-0.5h), half4(-0.5h,-1.5h,1.5h,-0.5h), half4(0.5h,-1.5h,1.5h,-0.5h), half4(1.5h,-1.5h,1.5h,-0.5h),
    half4(-1.5h,-0.5h,1.5h,-0.5h), half4(-0.5h,-0.5h,1.5h,-0.5h), half4(0.5h,-0.5h,1.5h,-0.5h), half4(1.5h,-0.5h,1.5h,-0.5h),
    half4(-1.5h,0.5h,1.5h,-0.5h), half4(-0.5h,0.5h,1.5h,-0.5h), half4(0.5h,0.5h,1.5h,-0.5h), half4(1.5h,0.5h,1.5h,-0.5h),
    half4(-1.5h,1.5h,1.5h,-0.5h), half4(-0.5h,1.5h,1.5h,-0.5h), half4(0.5h,1.5h,1.5h,-0.5h), half4(1.5h,1.5h,1.5h,-0.5h),
    half4(-1.5h,-1.5h,-1.5h,0.5h), half4(-0.5h,-1.5h,-1.5h,0.5h), half4(0.5h,-1.5h,-1.5h,0.5h), half4(1.5h,-1.5h,-1.5h,0.5h),
    half4(-1.5h,-0.5h,-1.5h,0.5h), half4(-0.5h,-0.5h,-1.5h,0.5h), half4(0.5h,-0.5h,-1.5h,0.5h), half4(1.5h,-0.5h,-1.5h,0.5h),
    half4(-1.5h,0.5h,-1.5h,0.5h), half4(-0.5h,0.5h,-1.5h,0.5h), half4(0.5h,0.5h,-1.5h,0.5h), half4(1.5h,0.5h,-1.5h,0.5h),
    half4(-1.5h,1.5h,-1.5h,0.5h), half4(-0.5h,1.5h,-1.5h,0.5h), half4(0.5h,1.5h,-1.5h,0.5h), half4(1.5h,1.5h,-1.5h,0.5h),
    half4(-1.5h,-1.5h,-0.5h,0.5h), half4(-0.5h,-1.5h,-0.5h,0.5h), half4(0.5h,-1.5h,-0.5h,0.5h), half4(1.5h,-1.5h,-0.5h,0.5h),
    half4(-1.5h,-0.5h,-0.5h,0.5h), half4(-0.5h,-0.5h,-0.5h,0.5h), half4(0.5h,-0.5h,-0.5h,0.5h), half4(1.5h,-0.5h,-0.5h,0.5h),
    half4(-1.5h,0.5h,-0.5h,0.5h), half4(-0.5h,0.5h,-0.5h,0.5h), half4(0.5h,0.5h,-0.5h,0.5h), half4(1.5h,0.5h,-0.5h,0.5h),
    half4(-1.5h,1.5h,-0.5h,0.5h), half4(-0.5h,1.5h,-0.5h,0.5h), half4(0.5h,1.5h,-0.5h,0.5h), half4(1.5h,1.5h,-0.5h,0.5h),
    half4(-1.5h,-1.5h,0.5h,0.5h), half4(-0.5h,-1.5h,0.5h,0.5h), half4(0.5h,-1.5h,0.5h,0.5h), half4(1.5h,-1.5h,0.5h,0.5h),
    half4(-1.5h,-0.5h,0.5h,0.5h), half4(-0.5h,-0.5h,0.5h,0.5h), half4(0.5h,-0.5h,0.5h,0.5h), half4(1.5h,-0.5h,0.5h,0.5h),
    half4(-1.5h,0.5h,0.5h,0.5h), half4(-0.5h,0.5h,0.5h,0.5h), half4(0.5h,0.5h,0.5h,0.5h), half4(1.5h,0.5h,0.5h,0.5h),
    half4(-1.5h,1.5h,0.5h,0.5h), half4(-0.5h,1.5h,0.5h,0.5h), half4(0.5h,1.5h,0.5h,0.5h), half4(1.5h,1.5h,0.5h,0.5h),
    half4(-1.5h,-1.5h,1.5h,0.5h), half4(-0.5h,-1.5h,1.5h,0.5h), half4(0.5h,-1.5h,1.5h,0.5h), half4(1.5h,-1.5h,1.5h,0.5h),
    half4(-1.5h,-0.5h,1.5h,0.5h), half4(-0.5h,-0.5h,1.5h,0.5h), half4(0.5h,-0.5h,1.5h,0.5h), half4(1.5h,-0.5h,1.5h,0.5h),
    half4(-1.5h,0.5h,1.5h,0.5h), half4(-0.5h,0.5h,1.5h,0.5h), half4(0.5h,0.5h,1.5h,0.5h), half4(1.5h,0.5h,1.5h,0.5h),
    half4(-1.5h,1.5h,1.5h,0.5h), half4(-0.5h,1.5h,1.5h,0.5h), half4(0.5h,1.5h,1.5h,0.5h), half4(1.5h,1.5h,1.5h,0.5h),
    half4(-1.5h,-1.5h,-1.5h,1.5h), half4(-0.5h,-1.5h,-1.5h,1.5h), half4(0.5h,-1.5h,-1.5h,1.5h), half4(1.5h,-1.5h,-1.5h,1.5h),
    half4(-1.5h,-0.5h,-1.5h,1.5h), half4(-0.5h,-0.5h,-1.5h,1.5h), half4(0.5h,-0.5h,-1.5h,1.5h), half4(1.5h,-0.5h,-1.5h,1.5h),
    half4(-1.5h,0.5h,-1.5h,1.5h), half4(-0.5h,0.5h,-1.5h,1.5h), half4(0.5h,0.5h,-1.5h,1.5h), half4(1.5h,0.5h,-1.5h,1.5h),
    half4(-1.5h,1.5h,-1.5h,1.5h), half4(-0.5h,1.5h,-1.5h,1.5h), half4(0.5h,1.5h,-1.5h,1.5h), half4(1.5h,1.5h,-1.5h,1.5h),
    half4(-1.5h,-1.5h,-0.5h,1.5h), half4(-0.5h,-1.5h,-0.5h,1.5h), half4(0.5h,-1.5h,-0.5h,1.5h), half4(1.5h,-1.5h,-0.5h,1.5h),
    half4(-1.5h,-0.5h,-0.5h,1.5h), half4(-0.5h,-0.5h,-0.5h,1.5h), half4(0.5h,-0.5h,-0.5h,1.5h), half4(1.5h,-0.5h,-0.5h,1.5h),
    half4(-1.5h,0.5h,-0.5h,1.5h), half4(-0.5h,0.5h,-0.5h,1.5h), half4(0.5h,0.5h,-0.5h,1.5h), half4(1.5h,0.5h,-0.5h,1.5h),
    half4(-1.5h,1.5h,-0.5h,1.5h), half4(-0.5h,1.5h,-0.5h,1.5h), half4(0.5h,1.5h,-0.5h,1.5h), half4(1.5h,1.5h,-0.5h,1.5h),
    half4(-1.5h,-1.5h,0.5h,1.5h), half4(-0.5h,-1.5h,0.5h,1.5h), half4(0.5h,-1.5h,0.5h,1.5h), half4(1.5h,-1.5h,0.5h,1.5h),
    half4(-1.5h,-0.5h,0.5h,1.5h), half4(-0.5h,-0.5h,0.5h,1.5h), half4(0.5h,-0.5h,0.5h,1.5h), half4(1.5h,-0.5h,0.5h,1.5h),
    half4(-1.5h,0.5h,0.5h,1.5h), half4(-0.5h,0.5h,0.5h,1.5h), half4(0.5h,0.5h,0.5h,1.5h), half4(1.5h,0.5h,0.5h,1.5h),
    half4(-1.5h,1.5h,0.5h,1.5h), half4(-0.5h,1.5h,0.5h,1.5h), half4(0.5h,1.5h,0.5h,1.5h), half4(1.5h,1.5h,0.5h,1.5h),
    half4(-1.5h,-1.5h,1.5h,1.5h), half4(-0.5h,-1.5h,1.5h,1.5h), half4(0.5h,-1.5h,1.5h,1.5h), half4(1.5h,-1.5h,1.5h,1.5h),
    half4(-1.5h,-0.5h,1.5h,1.5h), half4(-0.5h,-0.5h,1.5h,1.5h), half4(0.5h,-0.5h,1.5h,1.5h), half4(1.5h,-0.5h,1.5h,1.5h),
    half4(-1.5h,0.5h,1.5h,1.5h), half4(-0.5h,0.5h,1.5h,1.5h), half4(0.5h,0.5h,1.5h,1.5h), half4(1.5h,0.5h,1.5h,1.5h),
    half4(-1.5h,1.5h,1.5h,1.5h), half4(-0.5h,1.5h,1.5h,1.5h), half4(0.5h,1.5h,1.5h,1.5h), half4(1.5h,1.5h,1.5h,1.5h),
};
constant half4 kMppS3Hi[16] = {
    half4(-2.0h,-2.0h,-2.0h,-2.0h), half4(2.0h,-2.0h,-2.0h,-2.0h), half4(-2.0h,2.0h,-2.0h,-2.0h), half4(2.0h,2.0h,-2.0h,-2.0h),
    half4(-2.0h,-2.0h,2.0h,-2.0h), half4(2.0h,-2.0h,2.0h,-2.0h), half4(-2.0h,2.0h,2.0h,-2.0h), half4(2.0h,2.0h,2.0h,-2.0h),
    half4(-2.0h,-2.0h,-2.0h,2.0h), half4(2.0h,-2.0h,-2.0h,2.0h), half4(-2.0h,2.0h,-2.0h,2.0h), half4(2.0h,2.0h,-2.0h,2.0h),
    half4(-2.0h,-2.0h,2.0h,2.0h), half4(2.0h,-2.0h,2.0h,2.0h), half4(-2.0h,2.0h,2.0h,2.0h), half4(2.0h,2.0h,2.0h,2.0h),
};
constant half2 kMppS4Lut[256] = {
    half2(-7.5h,-7.5h), half2(-6.5h,-7.5h), half2(-5.5h,-7.5h), half2(-4.5h,-7.5h), half2(-3.5h,-7.5h), half2(-2.5h,-7.5h), half2(-1.5h,-7.5h), half2(-0.5h,-7.5h),
    half2(0.5h,-7.5h), half2(1.5h,-7.5h), half2(2.5h,-7.5h), half2(3.5h,-7.5h), half2(4.5h,-7.5h), half2(5.5h,-7.5h), half2(6.5h,-7.5h), half2(7.5h,-7.5h),
    half2(-7.5h,-6.5h), half2(-6.5h,-6.5h), half2(-5.5h,-6.5h), half2(-4.5h,-6.5h), half2(-3.5h,-6.5h), half2(-2.5h,-6.5h), half2(-1.5h,-6.5h), half2(-0.5h,-6.5h),
    half2(0.5h,-6.5h), half2(1.5h,-6.5h), half2(2.5h,-6.5h), half2(3.5h,-6.5h), half2(4.5h,-6.5h), half2(5.5h,-6.5h), half2(6.5h,-6.5h), half2(7.5h,-6.5h),
    half2(-7.5h,-5.5h), half2(-6.5h,-5.5h), half2(-5.5h,-5.5h), half2(-4.5h,-5.5h), half2(-3.5h,-5.5h), half2(-2.5h,-5.5h), half2(-1.5h,-5.5h), half2(-0.5h,-5.5h),
    half2(0.5h,-5.5h), half2(1.5h,-5.5h), half2(2.5h,-5.5h), half2(3.5h,-5.5h), half2(4.5h,-5.5h), half2(5.5h,-5.5h), half2(6.5h,-5.5h), half2(7.5h,-5.5h),
    half2(-7.5h,-4.5h), half2(-6.5h,-4.5h), half2(-5.5h,-4.5h), half2(-4.5h,-4.5h), half2(-3.5h,-4.5h), half2(-2.5h,-4.5h), half2(-1.5h,-4.5h), half2(-0.5h,-4.5h),
    half2(0.5h,-4.5h), half2(1.5h,-4.5h), half2(2.5h,-4.5h), half2(3.5h,-4.5h), half2(4.5h,-4.5h), half2(5.5h,-4.5h), half2(6.5h,-4.5h), half2(7.5h,-4.5h),
    half2(-7.5h,-3.5h), half2(-6.5h,-3.5h), half2(-5.5h,-3.5h), half2(-4.5h,-3.5h), half2(-3.5h,-3.5h), half2(-2.5h,-3.5h), half2(-1.5h,-3.5h), half2(-0.5h,-3.5h),
    half2(0.5h,-3.5h), half2(1.5h,-3.5h), half2(2.5h,-3.5h), half2(3.5h,-3.5h), half2(4.5h,-3.5h), half2(5.5h,-3.5h), half2(6.5h,-3.5h), half2(7.5h,-3.5h),
    half2(-7.5h,-2.5h), half2(-6.5h,-2.5h), half2(-5.5h,-2.5h), half2(-4.5h,-2.5h), half2(-3.5h,-2.5h), half2(-2.5h,-2.5h), half2(-1.5h,-2.5h), half2(-0.5h,-2.5h),
    half2(0.5h,-2.5h), half2(1.5h,-2.5h), half2(2.5h,-2.5h), half2(3.5h,-2.5h), half2(4.5h,-2.5h), half2(5.5h,-2.5h), half2(6.5h,-2.5h), half2(7.5h,-2.5h),
    half2(-7.5h,-1.5h), half2(-6.5h,-1.5h), half2(-5.5h,-1.5h), half2(-4.5h,-1.5h), half2(-3.5h,-1.5h), half2(-2.5h,-1.5h), half2(-1.5h,-1.5h), half2(-0.5h,-1.5h),
    half2(0.5h,-1.5h), half2(1.5h,-1.5h), half2(2.5h,-1.5h), half2(3.5h,-1.5h), half2(4.5h,-1.5h), half2(5.5h,-1.5h), half2(6.5h,-1.5h), half2(7.5h,-1.5h),
    half2(-7.5h,-0.5h), half2(-6.5h,-0.5h), half2(-5.5h,-0.5h), half2(-4.5h,-0.5h), half2(-3.5h,-0.5h), half2(-2.5h,-0.5h), half2(-1.5h,-0.5h), half2(-0.5h,-0.5h),
    half2(0.5h,-0.5h), half2(1.5h,-0.5h), half2(2.5h,-0.5h), half2(3.5h,-0.5h), half2(4.5h,-0.5h), half2(5.5h,-0.5h), half2(6.5h,-0.5h), half2(7.5h,-0.5h),
    half2(-7.5h,0.5h), half2(-6.5h,0.5h), half2(-5.5h,0.5h), half2(-4.5h,0.5h), half2(-3.5h,0.5h), half2(-2.5h,0.5h), half2(-1.5h,0.5h), half2(-0.5h,0.5h),
    half2(0.5h,0.5h), half2(1.5h,0.5h), half2(2.5h,0.5h), half2(3.5h,0.5h), half2(4.5h,0.5h), half2(5.5h,0.5h), half2(6.5h,0.5h), half2(7.5h,0.5h),
    half2(-7.5h,1.5h), half2(-6.5h,1.5h), half2(-5.5h,1.5h), half2(-4.5h,1.5h), half2(-3.5h,1.5h), half2(-2.5h,1.5h), half2(-1.5h,1.5h), half2(-0.5h,1.5h),
    half2(0.5h,1.5h), half2(1.5h,1.5h), half2(2.5h,1.5h), half2(3.5h,1.5h), half2(4.5h,1.5h), half2(5.5h,1.5h), half2(6.5h,1.5h), half2(7.5h,1.5h),
    half2(-7.5h,2.5h), half2(-6.5h,2.5h), half2(-5.5h,2.5h), half2(-4.5h,2.5h), half2(-3.5h,2.5h), half2(-2.5h,2.5h), half2(-1.5h,2.5h), half2(-0.5h,2.5h),
    half2(0.5h,2.5h), half2(1.5h,2.5h), half2(2.5h,2.5h), half2(3.5h,2.5h), half2(4.5h,2.5h), half2(5.5h,2.5h), half2(6.5h,2.5h), half2(7.5h,2.5h),
    half2(-7.5h,3.5h), half2(-6.5h,3.5h), half2(-5.5h,3.5h), half2(-4.5h,3.5h), half2(-3.5h,3.5h), half2(-2.5h,3.5h), half2(-1.5h,3.5h), half2(-0.5h,3.5h),
    half2(0.5h,3.5h), half2(1.5h,3.5h), half2(2.5h,3.5h), half2(3.5h,3.5h), half2(4.5h,3.5h), half2(5.5h,3.5h), half2(6.5h,3.5h), half2(7.5h,3.5h),
    half2(-7.5h,4.5h), half2(-6.5h,4.5h), half2(-5.5h,4.5h), half2(-4.5h,4.5h), half2(-3.5h,4.5h), half2(-2.5h,4.5h), half2(-1.5h,4.5h), half2(-0.5h,4.5h),
    half2(0.5h,4.5h), half2(1.5h,4.5h), half2(2.5h,4.5h), half2(3.5h,4.5h), half2(4.5h,4.5h), half2(5.5h,4.5h), half2(6.5h,4.5h), half2(7.5h,4.5h),
    half2(-7.5h,5.5h), half2(-6.5h,5.5h), half2(-5.5h,5.5h), half2(-4.5h,5.5h), half2(-3.5h,5.5h), half2(-2.5h,5.5h), half2(-1.5h,5.5h), half2(-0.5h,5.5h),
    half2(0.5h,5.5h), half2(1.5h,5.5h), half2(2.5h,5.5h), half2(3.5h,5.5h), half2(4.5h,5.5h), half2(5.5h,5.5h), half2(6.5h,5.5h), half2(7.5h,5.5h),
    half2(-7.5h,6.5h), half2(-6.5h,6.5h), half2(-5.5h,6.5h), half2(-4.5h,6.5h), half2(-3.5h,6.5h), half2(-2.5h,6.5h), half2(-1.5h,6.5h), half2(-0.5h,6.5h),
    half2(0.5h,6.5h), half2(1.5h,6.5h), half2(2.5h,6.5h), half2(3.5h,6.5h), half2(4.5h,6.5h), half2(5.5h,6.5h), half2(6.5h,6.5h), half2(7.5h,6.5h),
    half2(-7.5h,7.5h), half2(-6.5h,7.5h), half2(-5.5h,7.5h), half2(-4.5h,7.5h), half2(-3.5h,7.5h), half2(-2.5h,7.5h), half2(-1.5h,7.5h), half2(-0.5h,7.5h),
    half2(0.5h,7.5h), half2(1.5h,7.5h), half2(2.5h,7.5h), half2(3.5h,7.5h), half2(4.5h,7.5h), half2(5.5h,7.5h), half2(6.5h,7.5h), half2(7.5h,7.5h),
};
constant half4 kMppS5Hi[16] = {
    half4(-8.0h,-8.0h,-8.0h,-8.0h), half4(8.0h,-8.0h,-8.0h,-8.0h), half4(-8.0h,8.0h,-8.0h,-8.0h), half4(8.0h,8.0h,-8.0h,-8.0h),
    half4(-8.0h,-8.0h,8.0h,-8.0h), half4(8.0h,-8.0h,8.0h,-8.0h), half4(-8.0h,8.0h,8.0h,-8.0h), half4(8.0h,8.0h,8.0h,-8.0h),
    half4(-8.0h,-8.0h,-8.0h,8.0h), half4(8.0h,-8.0h,-8.0h,8.0h), half4(-8.0h,8.0h,-8.0h,8.0h), half4(8.0h,8.0h,-8.0h,8.0h),
    half4(-8.0h,-8.0h,8.0h,8.0h), half4(8.0h,-8.0h,8.0h,8.0h), half4(-8.0h,8.0h,8.0h,8.0h), half4(8.0h,8.0h,8.0h,8.0h),
};

// Constant-cache LUT .strata S_K decoder (10-15x fewer ALU ops on S2_K..S6_K!)
template <int F>
static inline void mpp_sk_w8(device const uchar* pl, int j, thread float4& o0, thread float4& o1) {
    if constexpr (F == 2) {
        device const uchar* p = pl + (j >> 2);
        o0 = float4(kMppS2Lut[p[0]]);
        o1 = float4(kMppS2Lut[p[1]]);
    } else if constexpr (F == 3) {
        device const uchar* p = pl + (j >> 2);
        device const uchar* hp = pl + 64;
        const uint h = uint(hp[j >> 3]);
        o0 = float4(kMppS2Lut[p[0]] + kMppS3Hi[h & 15u]);
        o1 = float4(kMppS2Lut[p[1]] + kMppS3Hi[h >> 4]);
    } else if constexpr (F == 4) {
        device const uchar* p = pl + (j >> 1);
        o0 = float4(half4(kMppS4Lut[p[0]], kMppS4Lut[p[1]]));
        o1 = float4(half4(kMppS4Lut[p[2]], kMppS4Lut[p[3]]));
    } else if constexpr (F == 5) {
        device const uchar* p = pl + (j >> 1);
        device const uchar* hp = pl + 128;
        const uint h = uint(hp[j >> 3]);
        o0 = float4(half4(kMppS4Lut[p[0]], kMppS4Lut[p[1]]) + kMppS5Hi[h & 15u]);
        o1 = float4(half4(kMppS4Lut[p[2]], kMppS4Lut[p[3]]) + kMppS5Hi[h >> 4]);
    } else if constexpr (F == 6) {
        device const uchar* p = pl + (j >> 1);
        device const uchar* hp = pl + 128 + (j >> 2);
        o0 = float4(fma(kMppS2Lut[hp[0]], half4(16.0h), half4(kMppS4Lut[p[0]], kMppS4Lut[p[1]])));
        o1 = float4(fma(kMppS2Lut[hp[1]], half4(16.0h), half4(kMppS4Lut[p[2]], kMppS4Lut[p[3]])));
    } else {
        device const char* qp = (device const char*)(pl + j);
        o0 = float4(float(qp[0]), float(qp[1]), float(qp[2]), float(qp[3]));
        o1 = float4(float(qp[4]), float(qp[5]), float(qp[6]), float(qp[7]));
    }
}

// Exact SITU(g, u, beta=4.0, lb=25.0) — matches ink_metal_codecs.metal bit-for-bit!
static inline float mpp_situ_act(float g, float u) {
    const float sig = 1.0f / (1.0f + exp(-g));
    const float g_act = 4.0f * tanh(clamp(g * 0.25f, -15.0f, 15.0f)) * sig;
    const float u_act = 25.0f * tanh(clamp(u * 0.04f, -15.0f, 15.0f));
    return g_act * u_act;
}

struct InkExpertDevMpp {
    ulong body_off;
    uint dn_off;
    uchar gu_code;
    uchar dn_code;
    ushort chunk_idx;
};

static inline device const uchar* ink_mpp_sec4_ptr(
    ushort cidx,
    device const uchar* s0,
    device const uchar* s1,
    device const uchar* s2,
    device const uchar* s3) {
    if (cidx == 0u) return s0;
    if (cidx == 1u) return s1;
    if (cidx == 2u) return s2;
    return s3;
}

// ============================================================================
// 1. Cast / Init-Time Weight Conversion + Metal 4 matmul2d Dense Trunk GEMMs
// ============================================================================

kernel void ink_mpp_f32_to_f16(
    device const float* src [[buffer(0)]],
    device half* dst        [[buffer(1)]],
    constant int& N         [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid) * 4;
    if (i + 3 < N) {
        *((device half4*)(dst + i)) = half4(*((device const float4*)(src + i)));
    } else {
        for (int k = i; k < N; ++k) dst[k] = half(src[k]);
    }
}

kernel void ink_mpp_bf16_to_f16(
    device const ushort* src [[buffer(0)]],
    device half* dst         [[buffer(1)]],
    constant int& N          [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid) * 4;
    if (i + 3 < N) {
        const ushort4 v = *((device const ushort4*)(src + i));
        *((device half4*)(dst + i)) = half4(mpp_bf16_to_f32(v.x), mpp_bf16_to_f32(v.y),
                                            mpp_bf16_to_f32(v.z), mpp_bf16_to_f32(v.w));
    } else {
        for (int k = i; k < N; ++k) dst[k] = half(mpp_bf16_to_f32(src[k]));
    }
}

kernel void ink_mpp_q8row_to_f16(
    device const uchar* W [[buffer(0)]],
    device half* dst      [[buffer(1)]],
    constant int& N       [[buffer(2)]],
    constant int& K       [[buffer(3)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int r = int(tgp.y);
    if (r >= N) return;
    const float sc = ((device const float*)W)[r];
    device const char* row = (device const char*)(W + (size_t)N * 4) + (size_t)r * K;
    const int k = int(tgp.x) * 256 + int(tid);
    if (k < K) dst[(size_t)r * K + k] = half(float(row[k]) * sc);
}

kernel void ink_mpp_sk_to_f16(
    device const uchar* W [[buffer(0)]],
    device half* dst      [[buffer(1)]],
    constant int& N       [[buffer(2)]],
    constant int& K       [[buffer(3)]],
    constant int& F       [[buffer(4)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int r = int(tgp.y);
    if (r >= N) return;
    const int SB = mpp_sk_sb_bytes(F);
    const size_t rowb = (size_t)(K >> 8) * SB;
    const int col = (int(tgp.x) * 32 + int(tid)) * 8;
    if (col >= K) return;
    const int sbi = col >> 8, j = col & 255, g = j >> 6;
    device const uchar* blk = W + (size_t)r * rowb + (size_t)sbi * SB;
    const float eff = float(as_type<half>(mpp_ld16(blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
    float4 w0 = 0.0f, w1 = 0.0f;
    if (F == 2)      mpp_sk_w8<2>(blk + 6, j, w0, w1);
    else if (F == 3) mpp_sk_w8<3>(blk + 6, j, w0, w1);
    else if (F == 4) mpp_sk_w8<4>(blk + 6, j, w0, w1);
    else if (F == 5) mpp_sk_w8<5>(blk + 6, j, w0, w1);
    else if (F == 6) mpp_sk_w8<6>(blk + 6, j, w0, w1);
    else             mpp_sk_w8<8>(blk + 6, j, w0, w1);
    *((device half4*)(dst + (size_t)r * K + col + 0)) = half4(w0 * eff);
    *((device half4*)(dst + (size_t)r * K + col + 4)) = half4(w1 * eff);
}

struct InkMppMmArgs {
    int M, N, K, out_stride;
};

// Native Metal 4 matmul2d: Y[M, out_stride] = X_h[M, K] @ W_h[N, K]^T (FP16 operands -> FP32 output)
template <int TM, int TN, int SG>
kernel void ink_mpp_mm_f16(
    device half* W          [[buffer(0)]],
    device half* X          [[buffer(1)]],
    device float* Y         [[buffer(2)]],
    constant InkMppMmArgs& a [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]]) {
    tensor<device half,  dextents<int32_t, 2>, tensor_inline> tA(X, dextents<int32_t, 2>(a.K, a.M));
    tensor<device half,  dextents<int32_t, 2>, tensor_inline> tB(W, dextents<int32_t, 2>(a.K, a.N));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(Y, dextents<int32_t, 2>(a.out_stride, a.M));
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = tA.slice(0, int(tg.y) * TM);
    auto mB = tB.slice(0, int(tg.x) * TN);
    auto mY = tY.slice(int(tg.x) * TN, int(tg.y) * TM);
    op.run(mA, mB, mY);
}

#define INK_MPP_MM(TM, TN, SG) \
    template [[host_name("ink_mpp_mm_f16_" #TM "_" #TN "_" #SG)]] kernel decltype(ink_mpp_mm_f16<TM, TN, SG>) ink_mpp_mm_f16<TM, TN, SG>;
INK_MPP_MM(16, 64, 4)
INK_MPP_MM(32, 64, 4)
INK_MPP_MM(64, 64, 4)
INK_MPP_MM(64, 128, 4)
INK_MPP_MM(128, 64, 4)

// Split concatenated [Q(3072), R(384), Sinks(24), Gate(3072)] (stride=6552) into separate buffers
// grid (M, 26, 1), 256 threads
kernel void ink_mpp_split_qrsg(
    device const float* src [[buffer(0)]],
    device float* q_out     [[buffer(1)]],
    device float* r_out     [[buffer(2)]],
    device float* s_out     [[buffer(3)]],
    device float* g_out     [[buffer(4)]],
    constant int& stride    [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int m = int(tgp.x);
    const int idx = int(tgp.y) * 256 + int(tid);
    if (idx >= 6552) return;
    const float v = src[(size_t)m * stride + idx];
    if (idx < 3072) {
        q_out[(size_t)m * 3072 + idx] = v;
    } else if (idx < 3456) {
        r_out[(size_t)m * 384 + (idx - 3072)] = v;
    } else if (idx < 3480) {
        s_out[(size_t)m * 24 + (idx - 3456)] = v;
    } else {
        g_out[(size_t)m * 3072 + (idx - 3480)] = v;
    }
}

// Split concatenated [K(1024), k_dyn(32)] or [V(1024), v_dyn(32)] (stride=1056)
// grid (M, 5, 1), 256 threads
kernel void ink_mpp_split_kv_dyn(
    device const float* src [[buffer(0)]],
    device float* kv_out    [[buffer(1)]],
    device float* dyn_out   [[buffer(2)]],
    constant int& has_dyn   [[buffer(3)]],
    constant int& stride    [[buffer(4)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int m = int(tgp.x);
    const int idx = int(tgp.y) * 256 + int(tid);
    if (idx < 1024) {
        kv_out[(size_t)m * 1024 + idx] = src[(size_t)m * stride + idx];
    } else if (has_dyn && idx < 1056) {
        dyn_out[(size_t)m * 32 + (idx - 1024)] = src[(size_t)m * stride + idx];
    }
}

// Fused controller apply + RMSNorm reading concatenated [gain(2304), forget(2304)] (stride=4608)!
// grid (M, 1, 1), 256 threads
kernel void ink_mpp_ctrl_fuse_gf_norm(
    device const float* cur_buf [[buffer(0)]],
    device const float* msg_buf [[buffer(1)]],
    device const float* gf_buf  [[buffer(2)]],  // [M, 4608]
    device const float* norm_w  [[buffer(3)]],
    device float* out_buf       [[buffer(4)]],
    uint m   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    device const float* cp = cur_buf + (size_t)m * 2304;
    device const float* mp = msg_buf + (size_t)m * 2304;
    device const float* gp = gf_buf + (size_t)m * 4608;
    device const float* fp = gp + 2304;
    device float* dst = out_buf + (size_t)m * 2304;
    float ss = 0.0f;
    for (uint d = tid; d < 2304u; d += 256u) {
        const float v = (1.0f - tanh(clamp(fp[d], -15.0f, 15.0f))) * cp[d]
                      + tanh(clamp(gp[d], -15.0f, 15.0f)) * mp[d];
        dst[d] = v;
        ss += v * v;
    }
    float s = simd_sum(ss);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tid & 31u) == 0u) sh[tid >> 5] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += sh[i];
    const float inv = rsqrt(tot / 2304.0f + 1e-6f);
    for (uint d = tid; d < 2304u; d += 256u) dst[d] = dst[d] * inv * norm_w[d];
}

// ============================================================================
// 2. Zero-DRAM-Write Multi-Tile Direct Metal 4 matmul2d Grouped-Expert MoE
//    - ZERO FP16 DRAM weight writes (eliminates ~140 GB DRAM traffic at P=1024!)
//    - 128-row Super-Tiles (NTILE=2 x TM=64) decode 8 KB compressed weights ONCE into SLM
//      and run 2x matmul2d(64, 64, 64) in registers -> 2x weight reuse, 0 extra DRAM!
//    - Specialized SF4 (CODEC=15), MX8 (CODEC=16), and General (CODEC=0) pipelines
//      with 16-byte vectorized loads + L2-swizzled grid (n_tiles, 36, 1)!
// ============================================================================

kernel void ink_mpp_moe_plan(
    device const int* topk_idx [[buffer(0)]],
    constant int& P            [[buffer(1)]],
    constant int& allow_128    [[buffer(2)]],  // 1 for M >= 128, 0 for M < 128
    device int* srt_pair       [[buffer(3)]],  // [total_rows] (-1 for padded rows)
    device int* pair_pos       [[buffer(4)]],  // [P] -> row index in sorted layout
    device int4* tiles_128     [[buffer(5)]],  // [n128] = (e, row0, valid_rows, 0)
    device int4* tiles_64      [[buffer(6)]],  // [n64]  = (e, row0, valid_rows, 0)
    device int4* tiles_32      [[buffer(7)]],  // [n32]  = (e, row0, valid_rows, 0)
    device int4* tiles_16      [[buffer(8)]],  // [n16]  = (e, row0, valid_rows, 0)
    device uint* meta          [[buffer(9)]],  // indirect dispatch args (24 uints)
    uint e    [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup int sg_r[4];
    threadgroup int sg_128[4];
    threadgroup int sg_64[4];
    threadgroup int sg_32[4];
    threadgroup int sg_16[4];

    int c = 0;
    for (int i = 0; i < P; ++i) if (topk_idx[i] == int(e)) ++c;

    // Exact 4-bucket binary tile decomposition (128 / 64 / 32 / 16): <= 15 wasted rows/expert (<4.5% waste at P=1024!)
    int n128 = 0, n64 = 0, n32 = 0, n16 = 0;
    if (c > 0) {
        int rem = c;
        if (allow_128) {
            n128 = rem >> 7;
            rem &= 127;
            if (rem > 112) { ++n128; rem = 0; }
        }
        if (rem > 0) {
            n64 = rem >> 6;
            rem &= 63;
            if (rem > 48) { ++n64; rem = 0; }
        }
        if (rem > 0) {
            n32 = rem >> 5;
            rem &= 31;
            if (rem > 16) { ++n32; rem = 0; }
        }
        if (rem > 0) {
            n16 = 1;
        }
    }
    const int rows = n128 * 128 + n64 * 64 + n32 * 32 + n16 * 16;

    const int ex_r   = simd_prefix_exclusive_sum(rows);
    const int ex_128 = simd_prefix_exclusive_sum(n128);
    const int ex_64  = simd_prefix_exclusive_sum(n64);
    const int ex_32  = simd_prefix_exclusive_sum(n32);
    const int ex_16  = simd_prefix_exclusive_sum(n16);
    if (lane == 31) {
        sg_r[sg]   = ex_r + rows;
        sg_128[sg] = ex_128 + n128;
        sg_64[sg]  = ex_64 + n64;
        sg_32[sg]  = ex_32 + n32;
        sg_16[sg]  = ex_16 + n16;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int br = 0, b128 = 0, b64 = 0, b32 = 0, b16 = 0;
    for (uint w = 0; w < sg; ++w) {
        br += sg_r[w]; b128 += sg_128[w]; b64 += sg_64[w]; b32 += sg_32[w]; b16 += sg_16[w];
    }
    const int roff = br + ex_r;
    const int o128 = b128 + ex_128;
    const int o64  = b64 + ex_64;
    const int o32  = b32 + ex_32;
    const int o16  = b16 + ex_16;

    int cur_r = roff, rem_c = c;
    for (int j = 0; j < n128; ++j) {
        const int v = min(128, rem_c);
        tiles_128[o128 + j] = int4(int(e), cur_r, v, 0);
        cur_r += 128; rem_c -= v;
    }
    for (int j = 0; j < n64; ++j) {
        const int v = min(64, rem_c);
        tiles_64[o64 + j] = int4(int(e), cur_r, v, 0);
        cur_r += 64; rem_c -= v;
    }
    for (int j = 0; j < n32; ++j) {
        const int v = min(32, rem_c);
        tiles_32[o32 + j] = int4(int(e), cur_r, v, 0);
        cur_r += 32; rem_c -= v;
    }
    for (int j = 0; j < n16; ++j) {
        const int v = min(16, rem_c);
        tiles_16[o16 + j] = int4(int(e), cur_r, v, 0);
        cur_r += 16; rem_c -= v;
    }

    for (int r = 0; r < rows; ++r) srt_pair[roff + r] = -1;
    if (c > 0) {
        int k = 0;
        for (int i = 0; i < P; ++i) {
            if (topk_idx[i] == int(e)) {
                const int row = roff + k++;
                srt_pair[row] = i;
                pair_pos[i] = row;
            }
        }
    }
    if (e == 127) {
        const uint t128 = uint(o128 + n128);
        const uint t64  = uint(o64 + n64);
        const uint t32  = uint(o32 + n32);
        const uint t16  = uint(o16 + n16);
        const uint tot_rows = uint(roff + rows);
        meta[0] = t128; meta[1] = t64; meta[2] = t32; meta[3] = tot_rows;
        // [4..6] gather: (tot_rows, 1, 1)
        meta[4] = tot_rows; meta[5] = 1u; meta[6] = 1u;
        // [7..9] dir_128 (L2-swizzled!): (t128, 36, 1)
        meta[7] = t128; meta[8] = 36u; meta[9] = 1u;
        // [10..12] dir_64 (L2-swizzled!): (t64, 36, 1)
        meta[10] = t64; meta[11] = 36u; meta[12] = 1u;
        // [13..15] dir_32 (L2-swizzled!): (t32, 36, 1)
        meta[13] = t32; meta[14] = 36u; meta[15] = 1u;
        // [16..18] dir_16 (L2-swizzled!): (t16, 36, 1)
        meta[16] = t16; meta[17] = 36u; meta[18] = 1u;
        meta[19] = t16;
    }
}

// Gather F32 x_in [M, 2304] into tile-aligned contiguous half XS [tot_rows, 2304]
// indirect grid (tot_rows, 1, 1), 256 threads
kernel void ink_mpp_moe_gather(
    device const float* x_in   [[buffer(0)]],
    device const int* srt_pair [[buffer(1)]],
    device half* XS            [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const int p = srt_pair[row];
    device half* dst = XS + (size_t)row * 2304;
    if (p >= 0) {
        device const float* src = x_in + (size_t)(p >> 3) * 2304;
        for (uint k4 = tid; k4 < 2304u / 4u; k4 += 256u) {
            *((device half4*)(dst + k4 * 4u)) = half4(*((device const float4*)(src + k4 * 4u)));
        }
    } else {
        for (uint k4 = tid; k4 < 2304u / 4u; k4 += 256u) {
            *((device half4*)(dst + k4 * 4u)) = half4(0.0h);
        }
    }
}

// Fast 32-weight vectorized decoders (FP32 scale multiply before half4 cast -> avoids FP16 scale underflow on MX8/SF4!)
static inline void mpp_dec_sf4_32(device const uchar* p_base, device const uchar* s_base, float outer,
                                  int row_pbytes, int row_sbytes, int gn, int k0, int half_idx, threadgroup half4* dst) {
    const int blk = (k0 >> 4) + half_idx * 2;
    const float sc0 = mpp_e4m3_to_f32(s_base[(size_t)gn * row_sbytes + blk + 0]) * outer;
    const float sc1 = mpp_e4m3_to_f32(s_base[(size_t)gn * row_sbytes + blk + 1]) * outer;
    const uint4 pk = *((device const uint4*)(p_base + (size_t)gn * row_pbytes + (size_t)blk * 8));
    dst[0] = half4(float4(kMppE2M1[pk.x & 15u], kMppE2M1[(pk.x >> 4) & 15u], kMppE2M1[(pk.x >> 8) & 15u], kMppE2M1[(pk.x >> 12) & 15u]) * sc0);
    dst[1] = half4(float4(kMppE2M1[(pk.x >> 16) & 15u], kMppE2M1[(pk.x >> 20) & 15u], kMppE2M1[(pk.x >> 24) & 15u], kMppE2M1[pk.x >> 28]) * sc0);
    dst[2] = half4(float4(kMppE2M1[pk.y & 15u], kMppE2M1[(pk.y >> 4) & 15u], kMppE2M1[(pk.y >> 8) & 15u], kMppE2M1[(pk.y >> 12) & 15u]) * sc0);
    dst[3] = half4(float4(kMppE2M1[(pk.y >> 16) & 15u], kMppE2M1[(pk.y >> 20) & 15u], kMppE2M1[(pk.y >> 24) & 15u], kMppE2M1[pk.y >> 28]) * sc0);
    dst[4] = half4(float4(kMppE2M1[pk.z & 15u], kMppE2M1[(pk.z >> 4) & 15u], kMppE2M1[(pk.z >> 8) & 15u], kMppE2M1[(pk.z >> 12) & 15u]) * sc1);
    dst[5] = half4(float4(kMppE2M1[(pk.z >> 16) & 15u], kMppE2M1[(pk.z >> 20) & 15u], kMppE2M1[(pk.z >> 24) & 15u], kMppE2M1[pk.z >> 28]) * sc1);
    dst[6] = half4(float4(kMppE2M1[pk.w & 15u], kMppE2M1[(pk.w >> 4) & 15u], kMppE2M1[(pk.w >> 8) & 15u], kMppE2M1[(pk.w >> 12) & 15u]) * sc1);
    dst[7] = half4(float4(kMppE2M1[(pk.w >> 16) & 15u], kMppE2M1[(pk.w >> 20) & 15u], kMppE2M1[(pk.w >> 24) & 15u], kMppE2M1[pk.w >> 28]) * sc1);
}

static inline void mpp_dec_mx8_32(device const uchar* w_base, device const uchar* s_base,
                                  int row_wbytes, int row_sbytes, int gn, int k0, int half_idx, threadgroup half4* dst) {
    const int col = k0 + half_idx * 32;
    const float sc = mpp_ue8m0_to_f32(s_base[(size_t)gn * row_sbytes + (col >> 5)]);
    device const uint4* wp = (device const uint4*)(w_base + (size_t)gn * row_wbytes + col);
    const uint4 a = wp[0], b = wp[1];
    dst[0] = half4(float4(mpp_e4m3_to_f32(a.x & 255u), mpp_e4m3_to_f32((a.x >> 8) & 255u), mpp_e4m3_to_f32((a.x >> 16) & 255u), mpp_e4m3_to_f32(a.x >> 24)) * sc);
    dst[1] = half4(float4(mpp_e4m3_to_f32(a.y & 255u), mpp_e4m3_to_f32((a.y >> 8) & 255u), mpp_e4m3_to_f32((a.y >> 16) & 255u), mpp_e4m3_to_f32(a.y >> 24)) * sc);
    dst[2] = half4(float4(mpp_e4m3_to_f32(a.z & 255u), mpp_e4m3_to_f32((a.z >> 8) & 255u), mpp_e4m3_to_f32((a.z >> 16) & 255u), mpp_e4m3_to_f32(a.z >> 24)) * sc);
    dst[3] = half4(float4(mpp_e4m3_to_f32(a.w & 255u), mpp_e4m3_to_f32((a.w >> 8) & 255u), mpp_e4m3_to_f32((a.w >> 16) & 255u), mpp_e4m3_to_f32(a.w >> 24)) * sc);
    dst[4] = half4(float4(mpp_e4m3_to_f32(b.x & 255u), mpp_e4m3_to_f32((b.x >> 8) & 255u), mpp_e4m3_to_f32((b.x >> 16) & 255u), mpp_e4m3_to_f32(b.x >> 24)) * sc);
    dst[5] = half4(float4(mpp_e4m3_to_f32(b.y & 255u), mpp_e4m3_to_f32((b.y >> 8) & 255u), mpp_e4m3_to_f32((b.y >> 16) & 255u), mpp_e4m3_to_f32(b.y >> 24)) * sc);
    dst[6] = half4(float4(mpp_e4m3_to_f32(b.z & 255u), mpp_e4m3_to_f32((b.z >> 8) & 255u), mpp_e4m3_to_f32((b.z >> 16) & 255u), mpp_e4m3_to_f32(b.z >> 24)) * sc);
    dst[7] = half4(float4(mpp_e4m3_to_f32(b.w & 255u), mpp_e4m3_to_f32((b.w >> 8) & 255u), mpp_e4m3_to_f32((b.w >> 16) & 255u), mpp_e4m3_to_f32(b.w >> 24)) * sc);
}

template <int F>
static inline void mpp_dec_sk_32_t(device const uchar* row_ptr, int col, threadgroup half4* dst) {
    constexpr int SB = mpp_sk_sb_bytes(F);
    const int sbi = col >> 8, j = col & 255, g = j >> 6;
    device const uchar* blk = row_ptr + (size_t)sbi * SB;
    const float eff = float(as_type<half>(mpp_ld16(blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
    for (int q = 0; q < 4; ++q) {
        float4 w0, w1;
        mpp_sk_w8<F>(blk + 6, j + q * 8, w0, w1);
        dst[q * 2 + 0] = half4(w0 * eff);
        dst[q * 2 + 1] = half4(w1 * eff);
    }
}

static inline void mpp_dec_sk_32(device const uchar* row_ptr, int code, int col, threadgroup half4* dst) {
    if (code == 3)      mpp_dec_sk_32_t<3>(row_ptr, col, dst);
    else if (code == 2) mpp_dec_sk_32_t<2>(row_ptr, col, dst);
    else if (code == 4) mpp_dec_sk_32_t<4>(row_ptr, col, dst);
    else if (code == 5) mpp_dec_sk_32_t<5>(row_ptr, col, dst);
    else if (code == 6) mpp_dec_sk_32_t<6>(row_ptr, col, dst);
    else if (code == 8) mpp_dec_sk_32_t<8>(row_ptr, col, dst);
}

// Direct Zero-DRAM-Write Grouped-Expert w13 + fused exact SITU (TK=64 -> 8 KB sB, max occupancy!)
// CODEC: 15=SF4, 16=MX8, 0=Generic | NTILE: 1 (TM rows) or 2 (2*TM=128 rows, 2x weight reuse!)
// indirect grid (n_tiles, 36, 1) [L2-swizzled!], 128 threads (4 SGs)
template <int CODEC, int TM, int NTILE>
kernel void ink_mpp_moe_gu_dir(
    device const uchar* sec4           [[buffer(0)]],
    device const InkExpertDevMpp* ex   [[buffer(1)]],
    device half* XS                    [[buffer(2)]],
    device half* HS                    [[buffer(3)]],
    device const int4* tiles           [[buffer(4)]],
    device const uchar* sec4_1         [[buffer(5)]],
    device const uchar* sec4_2         [[buffer(6)]],
    device const uchar* sec4_3         [[buffer(7)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int TN = 64, TK = 64, SG = 4, NTH = 128;
    constexpr int W_HALVES = TN * TK;       // 64*64 = 8 KB
    constexpr int O_HALVES = 2 * TM * TN;   // 16 KB (TM=64) / 8 KB (TM=32) / 4 KB (TM=16)
    constexpr int S_HALVES = W_HALVES > O_HALVES ? W_HALVES : O_HALVES;
    threadgroup half sMem[S_HALVES];

    const int4 tl = tiles[tg.x];
    const int e = tl.x, row0 = tl.y;
    const int n0 = int(tg.y) * TN;
    const InkExpertDevMpp ed = ex[e];
    device const uchar* body = ink_mpp_sec4_ptr(ed.chunk_idx, sec4, sec4_1, sec4_2, sec4_3) + ed.body_off;
    const bool is_sk_dn = (ed.dn_code != 15 && ed.dn_code != 16 && ed.dn_code != 20);
    const int gcode = (CODEC == 0) ? int(ed.gu_code) : CODEC;
    const float sf4_outer = (gcode == 15) ? (((device const float*)body)[0] / ((device const float*)body)[1]) : 0.0f;
    const size_t sk_rowb = (CODEC == 0 && gcode != 15 && gcode != 16) ? (9ul * size_t(mpp_sk_sb_bytes(gcode))) : 0ul;
    const int br = int(tid >> 1), hi = int(tid & 1u);
    const int gn = n0 + br;

    tensor<device half, dextents<int32_t, 2>, tensor_inline> tX(XS, dextents<int32_t, 2>(2304, 1 << 24));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB(sMem, dextents<int32_t, 2>(TK, TN));
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto tA0 = tX.template slice<TK, TM>(0, row0);
    auto acc0 = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB), float>();
    auto acc1 = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB), float>();
    for (uint16_t i = 0; i < acc0.get_capacity(); ++i) {
        if (acc0.is_valid_element(i)) {
            acc0[i] = 0.0f;
            if constexpr (NTILE == 2) acc1[i] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k0 = 0; k0 < 2304; k0 += TK) {
        threadgroup half4* d = (threadgroup half4*)(sMem + br * TK + hi * 32);
        if constexpr (CODEC == 15) {
            mpp_dec_sf4_32(body + 16, body + 16 + 2654208, sf4_outer, 1152, 144, gn, k0, hi, d);
        } else if constexpr (CODEC == 16) {
            mpp_dec_mx8_32(body, body + 5308416, 2304, 72, gn, k0, hi, d);
        } else {
            if (gcode == 15) {
                mpp_dec_sf4_32(body + 16, body + 16 + 2654208, sf4_outer, 1152, 144, gn, k0, hi, d);
            } else if (gcode == 16) {
                mpp_dec_mx8_32(body, body + 5308416, 2304, 72, gn, k0, hi, d);
            } else {
                mpp_dec_sk_32(body + 2304 + (size_t)gn * sk_rowb, gcode, k0 + hi * 32, d);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA0 = tX.template slice<TK, TM>(k0, row0);
        op.run(mA0, tB, acc0);
        if constexpr (NTILE == 2) {
            auto mA1 = tX.template slice<TK, TM>(k0, row0 + TM);
            op.run(mA1, tB, acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* ot = reinterpret_cast<threadgroup float*>(sMem);
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(ot, dextents<int32_t, 2>(TN, TM));
    constexpr int PAIRS = TM * 32;
    const int fbase = n0 >> 1;

    acc0.store(tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = int(tid); idx < PAIRS; idx += NTH) {
        const int m = idx >> 5, f = idx & 31;
        const float g = ot[m * TN + 2 * f], u = ot[m * TN + 2 * f + 1];
        const float inv_sq = is_sk_dn ? (1.0f / float(as_type<half>(mpp_ld16(body + 2 * (fbase + f))))) : 1.0f;
        HS[(size_t)(row0 + m) * 1152 + fbase + f] = half(mpp_situ_act(g, u) * inv_sq);
    }
    if constexpr (NTILE == 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc1.store(tO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = int(tid); idx < PAIRS; idx += NTH) {
            const int m = idx >> 5, f = idx & 31;
            const float g = ot[m * TN + 2 * f], u = ot[m * TN + 2 * f + 1];
            const float inv_sq = is_sk_dn ? (1.0f / float(as_type<half>(mpp_ld16(body + 2 * (fbase + f))))) : 1.0f;
            HS[(size_t)(row0 + TM + m) * 1152 + fbase + f] = half(mpp_situ_act(g, u) * inv_sq);
        }
    }
}

// Direct Zero-DRAM-Write Grouped-Expert w2 (TK=64 -> 8 KB SLM, max occupancy!)
// indirect grid (n_tiles, 36, 1) [L2-swizzled!], 128 threads (4 SGs)
template <int CODEC, int TM, int NTILE>
kernel void ink_mpp_moe_dn_dir(
    device const uchar* sec4           [[buffer(0)]],
    device const InkExpertDevMpp* ex   [[buffer(1)]],
    device half* HS                    [[buffer(2)]],
    device float* YE                   [[buffer(3)]],
    device const int4* tiles           [[buffer(4)]],
    device const uchar* sec4_1         [[buffer(5)]],
    device const uchar* sec4_2         [[buffer(6)]],
    device const uchar* sec4_3         [[buffer(7)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int TN = 64, TK = 64, SG = 4;
    threadgroup half sB[TN * TK];

    const int4 tl = tiles[tg.x];
    const int e = tl.x, row0 = tl.y;
    const int d0 = int(tg.y) * TN;
    const InkExpertDevMpp ed = ex[e];
    device const uchar* body = ink_mpp_sec4_ptr(ed.chunk_idx, sec4, sec4_1, sec4_2, sec4_3) + ed.body_off;
    const int dcode = (CODEC == 0) ? int(ed.dn_code) : CODEC;
    const float sf4_outer = (dcode == 15) ? (((device const float*)body)[2] / ((device const float*)body)[3]) : 0.0f;
    const size_t sk_pairb = (CODEC == 0 && dcode != 15 && dcode != 16) ? (9ul * size_t(mpp_sk_sb_bytes(dcode))) : 0ul;
    const int br = int(tid >> 1), hi = int(tid & 1u);
    const int gn = d0 + br;

    tensor<device half,  dextents<int32_t, 2>, tensor_inline> tH(HS, dextents<int32_t, 2>(1152, 1 << 24));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(YE, dextents<int32_t, 2>(2304, 1 << 24));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB(sB, dextents<int32_t, 2>(TK, TN));
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto tA0 = tH.template slice<TK, TM>(0, row0);
    auto acc0 = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB), float>();
    auto acc1 = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB), float>();
    for (uint16_t i = 0; i < acc0.get_capacity(); ++i) {
        if (acc0.is_valid_element(i)) {
            acc0[i] = 0.0f;
            if constexpr (NTILE == 2) acc1[i] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k0 = 0; k0 < 1152; k0 += TK) {
        threadgroup half4* d = (threadgroup half4*)(sB + br * TK + hi * 32);
        if constexpr (CODEC == 15) {
            mpp_dec_sf4_32(body + ed.dn_off, body + ed.dn_off + 1327104, sf4_outer, 576, 72, gn, k0, hi, d);
        } else if constexpr (CODEC == 16) {
            mpp_dec_mx8_32(body + ed.dn_off, body + ed.dn_off + 2654208, 1152, 36, gn, k0, hi, d);
        } else {
            if (dcode == 15) {
                mpp_dec_sf4_32(body + ed.dn_off, body + ed.dn_off + 1327104, sf4_outer, 576, 72, gn, k0, hi, d);
            } else if (dcode == 16) {
                mpp_dec_mx8_32(body + ed.dn_off, body + ed.dn_off + 2654208, 1152, 36, gn, k0, hi, d);
            } else {
                const int rp = gn >> 1;
                const int col_p = ((gn & 1) ? 1152 : 0) + k0 + hi * 32;
                mpp_dec_sk_32(body + ed.dn_off + (size_t)rp * sk_pairb, dcode, col_p, d);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA0 = tH.template slice<TK, TM>(k0, row0);
        op.run(mA0, tB, acc0);
        if constexpr (NTILE == 2) {
            auto mA1 = tH.template slice<TK, TM>(k0, row0 + TM);
            op.run(mA1, tB, acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    acc0.store(tY.slice(d0, row0));
    if constexpr (NTILE == 2) acc1.store(tY.slice(d0, row0 + TM));
}

#define INK_MPP_DIR_SET(CNAME, CODE) \
    template [[host_name("ink_mpp_moe_gu_" CNAME "_16")]]  kernel decltype(ink_mpp_moe_gu_dir<CODE, 16, 1>) ink_mpp_moe_gu_dir<CODE, 16, 1>; \
    template [[host_name("ink_mpp_moe_dn_" CNAME "_16")]]  kernel decltype(ink_mpp_moe_dn_dir<CODE, 16, 1>) ink_mpp_moe_dn_dir<CODE, 16, 1>; \
    template [[host_name("ink_mpp_moe_gu_" CNAME "_32")]]  kernel decltype(ink_mpp_moe_gu_dir<CODE, 32, 1>) ink_mpp_moe_gu_dir<CODE, 32, 1>; \
    template [[host_name("ink_mpp_moe_dn_" CNAME "_32")]]  kernel decltype(ink_mpp_moe_dn_dir<CODE, 32, 1>) ink_mpp_moe_dn_dir<CODE, 32, 1>; \
    template [[host_name("ink_mpp_moe_gu_" CNAME "_64")]]  kernel decltype(ink_mpp_moe_gu_dir<CODE, 64, 1>) ink_mpp_moe_gu_dir<CODE, 64, 1>; \
    template [[host_name("ink_mpp_moe_dn_" CNAME "_64")]]  kernel decltype(ink_mpp_moe_dn_dir<CODE, 64, 1>) ink_mpp_moe_dn_dir<CODE, 64, 1>; \
    template [[host_name("ink_mpp_moe_gu_" CNAME "_128")]] kernel decltype(ink_mpp_moe_gu_dir<CODE, 64, 2>) ink_mpp_moe_gu_dir<CODE, 64, 2>; \
    template [[host_name("ink_mpp_moe_dn_" CNAME "_128")]] kernel decltype(ink_mpp_moe_dn_dir<CODE, 64, 2>) ink_mpp_moe_dn_dir<CODE, 64, 2>;
INK_MPP_DIR_SET("sf4", 15)
INK_MPP_DIR_SET("mx8", 16)
INK_MPP_DIR_SET("gen", 0)

// Weighted reduce-8 from sorted YE rows -> moe_raw [M, 2304]
// grid (M, 9, 1), 256 threads
kernel void ink_mpp_moe_combine(
    device const float* YE       [[buffer(0)]],
    device const int* pair_pos   [[buffer(1)]],
    device const float* topk_w   [[buffer(2)]],
    device float* moe_raw        [[buffer(3)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int d = int(tgp.y) * 256 + int(tid);
    if (d >= 2304) return;
    const int m = int(tgp.x);
    float acc = 0.0f;
    for (int k = 0; k < 8; ++k) {
        const int row = pair_pos[m * 8 + k];
        acc = fma(YE[(size_t)row * 2304 + d], topk_w[m * 8 + k], acc);
    }
    moe_raw[(size_t)m * 2304 + d] = acc;
}

// ============================================================================
// 3. Metal 4 matmul2d FlashAttention-2 Prefill (BQ=32, BK=64) with RelPos & Sinks
// ============================================================================

struct InkMppAttnArgs {
    int M;
    int M_pad;
    int slot;
    int pos0;
    int is_swa;
    int swa_win;
    int rel_extent;
    int kv_stride;
    float scale;
    int pad;
};

// Pre-transpose Q [M, 24, 128] F32 -> Q_h [24, M_pad, 128] half (pre-scaled by 1/sqrt(128))
// and R [M, 24, 16] F32 -> R_h [24, M_pad, 16] half
// grid (M_pad, 24, 1), 128 threads
kernel void ink_mpp_prep_qr(
    device const float* q_in [[buffer(0)]],
    device const float* r_in [[buffer(1)]],
    device half* q_h         [[buffer(2)]],
    device half* r_h         [[buffer(3)]],
    constant int& M          [[buffer(4)]],
    constant int& M_pad      [[buffer(5)]],
    constant float& scale    [[buffer(6)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    const int m = int(tgp.x), h = int(tgp.y);
    const float qv = (m < M) ? (q_in[((size_t)m * 24 + h) * 128 + tid] * scale) : 0.0f;
    q_h[((size_t)h * M_pad + m) * 128 + tid] = half(qv);
    if (tid < 16u) {
        const float rv = (m < M) ? r_in[((size_t)m * 24 + h) * 16 + tid] : 0.0f;
        r_h[((size_t)h * M_pad + m) * 16 + tid] = half(rv);
    }
}

// Metal 4 FlashAttention-2 Prefill: BQ=32 queries x BK=64 keys, 4 SGs = 128 threads, 21.5 KB SLM!
// grid (M_pad / 32, 24, 1), 128 threads
kernel void ink_mpp_relpos_gqa_pf(
    device half* q_h                [[buffer(0)]],  // [24, M_pad, 128] pre-scaled
    device const half* r_h          [[buffer(1)]],  // [24, M_pad, 16]
    device const half* rel_proj_t   [[buffer(2)]],  // [1025, 16] transposed half (L1-hot!)
    device const float* sinks_buf   [[buffer(3)]],  // [M, 24]
    device const float* gate_buf    [[buffer(4)]],  // [M, 24, 128]
    device const half* k_cache      [[buffer(5)]],  // [max_slots, kv_stride, 8, 128]
    device const half* v_cache      [[buffer(6)]],  // [max_slots, kv_stride, 8, 128]
    device float* attn_out          [[buffer(7)]],  // [M, 24, 128]
    constant InkMppAttnArgs& A      [[buffer(8)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup half sKV[64 * 128];   // 16 KB: holds K[64,128], then sS[32,64] (8KB float), then V[64,128], then sO[32,128] (16KB float)
    threadgroup half sP[32 * 64];     // 4 KB
    threadgroup half sR[32 * 16];     // 1 KB
    threadgroup float sm[32], sl[32], sc[32];
    threadgroup float* sF = reinterpret_cast<threadgroup float*>(sKV);

    const int q0 = int(tgp.x) * 32;
    const int h = int(tgp.y);
    const int kh = h / 3;
    const int qpos0 = A.pos0 + q0;
    const int q_valid = min(32, A.M - q0);
    if (q_valid <= 0) return;

    if (tid < 32u) {
        sm[tid] = -INFINITY;
        sl[tid] = 0.0f;
        sc[tid] = 0.0f;
    }
    // Load R[32, 16] into sR (128 threads x 4 halfs = 512 halfs)
    {
        const int r = int(tid >> 2), c4 = int((tid & 3u) * 4u);
        *((threadgroup half4*)(sR + r * 16 + c4)) =
            *((device const half4*)(r_h + ((size_t)h * A.M_pad + q0 + r) * 16 + c4));
    }

    device half* qbase = q_h + ((size_t)h * A.M_pad + q0) * 128;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ(qbase, dextents<int32_t, 2>(128, 32));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tKV(sKV, dextents<int32_t, 2>(128, 64));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(sP, dextents<int32_t, 2>(64, 32));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tS(sF, dextents<int32_t, 2>(64, 32));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(sF, dextents<int32_t, 2>(128, 32));

    constexpr auto dQK = matmul2d_descriptor(32, 64, 128, false, true, false);
    constexpr auto dPV = matmul2d_descriptor(32, 128, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dQK, execution_simdgroups<4>> opQK;
    matmul2d<dPV, execution_simdgroups<4>> opPV;

    auto accO = opPV.template get_destination_cooperative_tensor<decltype(tP), decltype(tKV), float>();
    for (uint16_t i = 0; i < accO.get_capacity(); ++i)
        if (accO.is_valid_element(i)) accO[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int k_min = A.is_swa ? max(0, qpos0 - A.swa_win + 1) : 0;
    const int k_max = qpos0 + q_valid;  // exclusive
    const size_t slot_kv_base = (size_t)A.slot * A.kv_stride;

    for (int k0 = k_min; k0 < k_max; k0 += 64) {
        // 1. Stage K[64, 128] into sKV (128 threads x 8 uint4 = 64*128 halfs)
        for (int p = 0; p < 8; ++p) {
            const uint idx = uint(p * 128) + tid;
            const uint kr = idx >> 4;
            const uint kc = (idx & 15u) << 3;
            const int kpos = k0 + int(kr);
            uint4 kv = uint4(0u);
            if (kpos < k_max) {
                const int kidx = A.is_swa ? (kpos & 1023) : kpos;
                const size_t off = ((slot_kv_base + kidx) * 8 + kh) * 128 + kc;
                kv = *((device const uint4*)(k_cache + off));
            }
            *((threadgroup uint4*)(sKV + kr * 128u + kc)) = kv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. S[32, 64] = Q[32, 128] @ K[64, 128]^T in cooperative tensor registers!
        auto accS = opQK.template get_destination_cooperative_tensor<decltype(tQ), decltype(tKV), float>();
        opQK.run(tQ, tKV, accS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        accS.store(tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Add RelPos bias + Causal/SWA mask + Deferred-Sink Online Softmax -> sP[32, 64]
        // 128 threads = 32 query rows x 4 threads/row (each thread handles 16 keys!)
        {
            const int r = int(tid >> 2);
            const int lane4 = int(tid & 3u);
            const int qpos = qpos0 + r;
            const bool r_ok = (r < q_valid);
            const half4 r0 = *((threadgroup half4*)(sR + r * 16 + 0));
            const half4 r1 = *((threadgroup half4*)(sR + r * 16 + 4));
            const half4 r2 = *((threadgroup half4*)(sR + r * 16 + 8));
            const half4 r3 = *((threadgroup half4*)(sR + r * 16 + 12));
            const float4 rf0 = float4(r0), rf1 = float4(r1), rf2 = float4(r2), rf3 = float4(r3);

            float vals[16];
            float mx = -INFINITY;
            for (int e = 0; e < 16; ++e) {
                const int j = lane4 * 16 + e;
                const int kpos = k0 + j;
                float x = -INFINITY;
                if (r_ok && kpos <= qpos && (!A.is_swa || (qpos - kpos) < A.swa_win)) {
                    const int dist = min(qpos - kpos, A.rel_extent);
                    device const half4* rp = (device const half4*)(rel_proj_t + (size_t)dist * 16);
                    const float rb = dot(rf0, float4(rp[0])) + dot(rf1, float4(rp[1]))
                                   + dot(rf2, float4(rp[2])) + dot(rf3, float4(rp[3]));
                    x = sF[r * 64 + j] + rb;
                }
                vals[e] = x;
                mx = max(mx, x);
            }
            mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
            mx = max(mx, simd_shuffle_xor(mx, ushort(2)));

            const float m_old = sm[r];
            const float mn = max(m_old, mx);
            const float corr = (m_old == -INFINITY) ? 0.0f : exp(m_old - mn);
            float ps = 0.0f;
            for (int e = 0; e < 16; ++e) {
                const int j = lane4 * 16 + e;
                const float p = (vals[e] > -1e29f) ? exp(vals[e] - mn) : 0.0f;
                ps += p;
                sP[r * 64 + j] = half(p);
            }
            ps += simd_shuffle_xor(ps, ushort(1));
            ps += simd_shuffle_xor(ps, ushort(2));
            if (lane4 == 0) {
                sm[r] = mn;
                sl[r] = sl[r] * corr + ps;
                sc[r] = corr;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4. Rescale accO if not first block
        if (k0 > k_min) {
            for (uint16_t i = 0; i < accO.get_capacity(); ++i) {
                if (accO.is_valid_element(i)) {
                    const int row = accO.get_multidimensional_index(i)[1];
                    accO[i] *= sc[row];
                }
            }
        }

        // 5. Stage V[64, 128] into sKV and accumulate O += P @ V
        for (int p = 0; p < 8; ++p) {
            const uint idx = uint(p * 128) + tid;
            const uint vr = idx >> 4;
            const uint vc = (idx & 15u) << 3;
            const int kpos = k0 + int(vr);
            uint4 vv = uint4(0u);
            if (kpos < k_max) {
                const int kidx = A.is_swa ? (kpos & 1023) : kpos;
                const size_t off = ((slot_kv_base + kidx) * 8 + kh) * 128 + vc;
                vv = *((device const uint4*)(v_cache + off));
            }
            *((threadgroup uint4*)(sKV + vr * 128u + vc)) = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        opPV.run(tP, tKV, accO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 6. Final deferred-sink normalization + fused sigmoid(gate) -> attn_out[q0+r, h, 0..127]
    if (int(tid) < q_valid) {
        const float sk = sinks_buf[(size_t)(q0 + int(tid)) * 24 + h];
        const float m_k = sm[tid];
        if (m_k == -INFINITY) {
            sc[tid] = 0.0f;
        } else {
            const float m_fin = max(m_k, sk);
            const float so = exp(m_k - m_fin);
            sc[tid] = so / (sl[tid] * so + exp(sk - m_fin));
        }
    }
    accO.store(tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int r = 0; r < q_valid; ++r) {
        const float inv = sc[r];
        const size_t goff = ((size_t)(q0 + r) * 24 + h) * 128 + tid;
        const float g = gate_buf[goff];
        const float g_sig = 1.0f / (1.0f + exp(-g));
        attn_out[goff] = sF[r * 128 + tid] * inv * g_sig;
    }
}

// ============================================================================
// 4. M5 Max Opt-In: INT8 (char) Metal 4 NAX Q8_ROW Dense Trunk Kernels (FAST)
//    - Halves trunk weight memory (skips +2.73 GiB duplicate FP16 trunk!) & 2x GEMV BW
//    - Native Metal 4 matmul2d<char, char -> int> NAX tensor ops + in-register float rescale!
// ============================================================================

kernel void ink_mpp_bf16_to_q8row(
    device const ushort* src [[buffer(0)]],
    device char* q_dst       [[buffer(1)]],
    device float* sc_dst     [[buffer(2)]],
    constant int& N          [[buffer(3)]],
    constant int& K          [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (int(row) >= N) return;
    threadgroup float sh[8];
    device const ushort* rp = src + (size_t)row * K;
    float mx = 0.0f;
    for (int k = int(tid); k < K; k += 256) mx = max(mx, fabs(mpp_bf16_to_f32(rp[k])));
    mx = simd_max(mx);
    if ((tid & 31u) == 0u) sh[tid >> 5] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float g = 0.0f;
        for (int s = 0; s < 8; ++s) g = max(g, sh[s]);
        sh[0] = (g > 1e-12f) ? (g / 127.0f) : 1.0f;
        sc_dst[row] = sh[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / sh[0];
    device char* dst = q_dst + (size_t)row * K;
    for (int k = int(tid); k < K; k += 256) {
        dst[k] = char(clamp(rint(mpp_bf16_to_f32(rp[k]) * inv), -127.0f, 127.0f));
    }
}

kernel void ink_mpp_f32_to_q8row(
    device const float* src [[buffer(0)]],
    device char* q_dst      [[buffer(1)]],
    device float* sc_dst    [[buffer(2)]],
    constant int& N         [[buffer(3)]],
    constant int& K         [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (int(row) >= N) return;
    threadgroup float sh[8];
    device const float* rp = src + (size_t)row * K;
    float mx = 0.0f;
    for (int k = int(tid); k < K; k += 256) mx = max(mx, fabs(rp[k]));
    mx = simd_max(mx);
    if ((tid & 31u) == 0u) sh[tid >> 5] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float g = 0.0f;
        for (int s = 0; s < 8; ++s) g = max(g, sh[s]);
        sh[0] = (g > 1e-12f) ? (g / 127.0f) : 1.0f;
        sc_dst[row] = sh[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / sh[0];
    device char* dst = q_dst + (size_t)row * K;
    for (int k = int(tid); k < K; k += 256) {
        dst[k] = char(clamp(rint(rp[k] * inv), -127.0f, 127.0f));
    }
}

kernel void ink_mpp_q8row_copy(
    device const uchar* W [[buffer(0)]],
    device char* q_dst    [[buffer(1)]],
    device float* sc_dst  [[buffer(2)]],
    constant int& N       [[buffer(3)]],
    constant int& K       [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (int(row) >= N) return;
    if (tid == 0u) sc_dst[row] = ((device const float*)W)[row];
    device const char* src = (device const char*)(W + (size_t)N * 4) + (size_t)row * K;
    device char* dst = q_dst + (size_t)row * K;
    for (int k = int(tid); k < K; k += 256) dst[k] = src[k];
}

kernel void ink_mpp_sk_to_q8row(
    device const uchar* W [[buffer(0)]],
    device char* q_dst    [[buffer(1)]],
    device float* sc_dst  [[buffer(2)]],
    constant int& N       [[buffer(3)]],
    constant int& K       [[buffer(4)]],
    constant int& F       [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (int(row) >= N) return;
    threadgroup float sh[8];
    const int SB = mpp_sk_sb_bytes(F);
    const size_t rowb = (size_t)(K >> 8) * SB;
    device const uchar* rp = W + (size_t)row * rowb;
    const int n8 = K >> 3;
    float mx = 0.0f;
    for (int u = int(tid); u < n8; u += 256) {
        const int col = u << 3, sbi = col >> 8, j = col & 255, g = j >> 6;
        device const uchar* blk = rp + (size_t)sbi * SB;
        const float eff = float(as_type<half>(mpp_ld16(blk))) * (float(blk[2 + g]) * (1.0f / 255.0f));
        float4 w0 = 0.0f, w1 = 0.0f;
        if (F == 2) mpp_sk_w8<2>(blk + 6, j, w0, w1);
        else if (F == 3) mpp_sk_w8<3>(blk + 6, j, w0, w1);
        else if (F == 4) mpp_sk_w8<4>(blk + 6, j, w0, w1);
        else if (F == 5) mpp_sk_w8<5>(blk + 6, j, w0, w1);
        else if (F == 6) mpp_sk_w8<6>(blk + 6, j, w0, w1);
        else mpp_sk_w8<8>(blk + 6, j, w0, w1);
        const float4 a0 = fabs(w0 * eff), a1 = fabs(w1 * eff);
        mx = max(mx, max(max(a0.x, a0.y), max(a0.z, a0.w)));
        mx = max(mx, max(max(a1.x, a1.y), max(a1.z, a1.w)));
    }
    mx = simd_max(mx);
    if ((tid & 31u) == 0u) sh[tid >> 5] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float g = 0.0f;
        for (int s = 0; s < 8; ++s) g = max(g, sh[s]);
        sh[0] = (g > 1e-12f) ? (g / 127.0f) : 1.0f;
        sc_dst[row] = sh[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / sh[0];
    device char* dst = q_dst + (size_t)row * K;
    for (int u = int(tid); u < n8; u += 256) {
        const int col = u << 3, sbi = col >> 8, j = col & 255, g = j >> 6;
        device const uchar* blk = rp + (size_t)sbi * SB;
        const float eff = float(as_type<half>(mpp_ld16(blk))) * (float(blk[2 + g]) * (1.0f / 255.0f)) * inv;
        float4 w0 = 0.0f, w1 = 0.0f;
        if (F == 2) mpp_sk_w8<2>(blk + 6, j, w0, w1);
        else if (F == 3) mpp_sk_w8<3>(blk + 6, j, w0, w1);
        else if (F == 4) mpp_sk_w8<4>(blk + 6, j, w0, w1);
        else if (F == 5) mpp_sk_w8<5>(blk + 6, j, w0, w1);
        else if (F == 6) mpp_sk_w8<6>(blk + 6, j, w0, w1);
        else mpp_sk_w8<8>(blk + 6, j, w0, w1);
        dst[col + 0] = char(clamp(rint(w0.x * eff), -127.0f, 127.0f));
        dst[col + 1] = char(clamp(rint(w0.y * eff), -127.0f, 127.0f));
        dst[col + 2] = char(clamp(rint(w0.z * eff), -127.0f, 127.0f));
        dst[col + 3] = char(clamp(rint(w0.w * eff), -127.0f, 127.0f));
        dst[col + 4] = char(clamp(rint(w1.x * eff), -127.0f, 127.0f));
        dst[col + 5] = char(clamp(rint(w1.y * eff), -127.0f, 127.0f));
        dst[col + 6] = char(clamp(rint(w1.z * eff), -127.0f, 127.0f));
        dst[col + 7] = char(clamp(rint(w1.w * eff), -127.0f, 127.0f));
    }
}

// Dynamic per-row activation quantizer F32 [M, K] -> INT8 [M, K] + float scale [M]
// grid (M, 1, 1), 256 threads
kernel void ink_mpp_quant_x_i8(
    device const float* src [[buffer(0)]],
    device char* q_dst      [[buffer(1)]],
    device float* sc_dst    [[buffer(2)]],
    constant int& K         [[buffer(3)]],
    uint m   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[8];
    device const float* rp = src + (size_t)m * K;
    float mx = 0.0f;
    for (int k = int(tid); k < K; k += 256) mx = max(mx, fabs(rp[k]));
    mx = simd_max(mx);
    if ((tid & 31u) == 0u) sh[tid >> 5] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float g = 0.0f;
        for (int s = 0; s < 8; ++s) g = max(g, sh[s]);
        sh[0] = (g > 1e-12f) ? (g / 127.0f) : 1.0f;
        sc_dst[m] = sh[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / sh[0];
    device char* dst = q_dst + (size_t)m * K;
    for (int k = int(tid); k < K; k += 256) {
        dst[k] = char(clamp(rint(rp[k] * inv), -127.0f, 127.0f));
    }
}

// Native Metal 4 INT8 matmul2d: Y[M, out_stride] = (X_i8[M, K] @ W_i8[N, K]^T) * Xs[m] * Ws[n]
// SLM-cached row/col scales + coalesced threadgroup float4 epilogue (eliminates 16K DRAM scale loads/tile!)
template <int TM, int TN, int SG>
kernel void ink_mpp_mm_i8(
    device int8_t* W         [[buffer(0)]],
    device const float* Ws   [[buffer(1)]],
    device int8_t* X         [[buffer(2)]],
    device const float* Xs   [[buffer(3)]],
    device float* Y          [[buffer(4)]],
    constant InkMppMmArgs& a [[buffer(5)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup float sXs[TM];
    threadgroup float sWs[TN];
    const int m0 = int(tg.y) * TM;
    const int n0 = int(tg.x) * TN;
    if (int(tid) < TM) sXs[tid] = (m0 + int(tid) < a.M) ? Xs[m0 + int(tid)] : 0.0f;
    if (int(tid) < TN) sWs[tid] = (n0 + int(tid) < a.N) ? Ws[n0 + int(tid)] : 0.0f;

    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tA(X, dextents<int32_t, 2>(a.K, a.M));
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tB(W, dextents<int32_t, 2>(a.K, a.N));
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = tA.slice(0, m0);
    auto mB = tB.slice(0, n0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), int>();
    op.run(mA, mB, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            const auto idx = acc.get_multidimensional_index(i);
            const int ln = int(idx[0]);
            const int lm = int(idx[1]);
            const int gn = n0 + ln;
            const int gm = m0 + lm;
            if (gm < a.M && gn < a.N) {
                Y[(size_t)gm * a.out_stride + gn] = float(acc[i]) * sXs[lm] * sWs[ln];
            }
        }
    }
}

#define INK_MPP_MM_I8(TM, TN, SG) \
    template [[host_name("ink_mpp_mm_i8_" #TM "_" #TN "_" #SG)]] kernel decltype(ink_mpp_mm_i8<TM, TN, SG>) ink_mpp_mm_i8<TM, TN, SG>;
INK_MPP_MM_I8(8, 64, 4)
INK_MPP_MM_I8(16, 64, 4)
INK_MPP_MM_I8(32, 64, 4)
INK_MPP_MM_I8(64, 64, 4)
INK_MPP_MM_I8(64, 128, 4)

// Ultra-fast INT8 GEMV for M==1 (32 rows/TG = 4 rows/SG, vectorized char4x4 loads -> 2x trunk BW vs BF16!)
// grid (ceil(N / 32), 1, 1), 256 threads
kernel void ink_mpp_mv1_i8(
    device const char* W    [[buffer(0)]],
    device const float* Ws  [[buffer(1)]],
    device const float* X   [[buffer(2)]],
    device float* Y         [[buffer(3)]],
    constant int& N         [[buffer(4)]],
    constant int& K         [[buffer(5)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint sg = tid >> 5, lane = tid & 31u;
    const int row0 = int(tg) * 32 + int(sg) * 4;
    if (row0 >= N) return;
    const bool h1 = (row0 + 1) < N, h2 = (row0 + 2) < N, h3 = (row0 + 3) < N;
    device const char4* w0p = (device const char4*)(W + (size_t)row0 * K);
    device const char4* w1p = (device const char4*)(W + (size_t)(row0 + (h1 ? 1 : 0)) * K);
    device const char4* w2p = (device const char4*)(W + (size_t)(row0 + (h2 ? 2 : 0)) * K);
    device const char4* w3p = (device const char4*)(W + (size_t)(row0 + (h3 ? 3 : 0)) * K);
    device const float4* xp = (device const float4*)X;
    const int K16 = K >> 4;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int k = int(lane); k < K16; k += 32) {
        const int k4 = k * 4;
        const float4 x0 = xp[k4], x1 = xp[k4 + 1], x2 = xp[k4 + 2], x3 = xp[k4 + 3];
        a0 += dot(float4(w0p[k4]), x0) + dot(float4(w0p[k4 + 1]), x1) + dot(float4(w0p[k4 + 2]), x2) + dot(float4(w0p[k4 + 3]), x3);
        if (h1) a1 += dot(float4(w1p[k4]), x0) + dot(float4(w1p[k4 + 1]), x1) + dot(float4(w1p[k4 + 2]), x2) + dot(float4(w1p[k4 + 3]), x3);
        if (h2) a2 += dot(float4(w2p[k4]), x0) + dot(float4(w2p[k4 + 1]), x1) + dot(float4(w2p[k4 + 2]), x2) + dot(float4(w2p[k4 + 3]), x3);
        if (h3) a3 += dot(float4(w3p[k4]), x0) + dot(float4(w3p[k4 + 1]), x1) + dot(float4(w3p[k4 + 2]), x2) + dot(float4(w3p[k4 + 3]), x3);
    }
    const float v0 = simd_sum(a0) * Ws[row0]; if (lane == 0u) Y[row0] = v0;
    if (h1) { const float v1 = simd_sum(a1) * Ws[row0 + 1]; if (lane == 0u) Y[row0 + 1] = v1; }
    if (h2) { const float v2 = simd_sum(a2) * Ws[row0 + 2]; if (lane == 0u) Y[row0 + 2] = v2; }
    if (h3) { const float v3 = simd_sum(a3) * Ws[row0 + 3]; if (lane == 0u) Y[row0 + 3] = v3; }
}

// Single-pass INT8 GEMV for M=2..8 (16 rows/TG = 2 rows/SG, zero weight re-reads!)
// grid (ceil(N / 16), 1, 1), 256 threads
kernel void ink_mpp_mv_i8(
    device const char* W    [[buffer(0)]],
    device const float* Ws  [[buffer(1)]],
    device const float* X   [[buffer(2)]],
    device float* Y         [[buffer(3)]],
    constant int& N         [[buffer(4)]],
    constant int& K         [[buffer(5)]],
    constant int& M         [[buffer(6)]],
    constant int& out_stride [[buffer(7)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint sg = tid >> 5, lane = tid & 31u;
    const int row0 = int(tg) * 16 + int(sg) * 2;
    if (row0 >= N) return;
    const bool h1 = (row0 + 1) < N;
    device const char4* w0p = (device const char4*)(W + (size_t)row0 * K);
    device const char4* w1p = (device const char4*)(W + (size_t)(row0 + (h1 ? 1 : 0)) * K);
    const int K4 = K >> 2;
    float a0[8] = {0,0,0,0,0,0,0,0};
    float a1[8] = {0,0,0,0,0,0,0,0};
    for (int k = int(lane); k < K4; k += 32) {
        const float4 w0 = float4(w0p[k]);
        const float4 w1 = h1 ? float4(w1p[k]) : float4(0.0f);
        for (int i = 0; i < M; ++i) {
            const float4 xv = ((device const float4*)(X + (size_t)i * K))[k];
            a0[i] += dot(w0, xv);
            if (h1) a1[i] += dot(w1, xv);
        }
    }
    const float sc0 = Ws[row0], sc1 = h1 ? Ws[row0 + 1] : 0.0f;
    for (int i = 0; i < M; ++i) {
        const float v0 = simd_sum(a0[i]) * sc0;
        if (lane == 0u) Y[(size_t)i * out_stride + row0] = v0;
        if (h1) {
            const float v1 = simd_sum(a1[i]) * sc1;
            if (lane == 0u) Y[(size_t)i * out_stride + row0 + 1] = v1;
        }
    }
}

// ============================================================================
// 5. M5 Max Opt-In: 16 KB Double-Buffered SLM + simd_shuffle SF4 + Full 20-Type GGUF_RAW + TM=8 NAX MoE
// ============================================================================

struct InkMppGgufHdr {
    uint magic;      // 0x47475546 ('GGUF')
    uint gu_type;    // ggml_type for gate/up
    uint dn_type;    // ggml_type for down
    uint flags;      // bit 0: 1=interleaved, 0=split
    float gate_scale;
    float up_scale;
    float dn_scale;
    uint reserved;
};

constant uchar kMppIq3XxsSigns[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255
};

constant uint kMppIq3XxsGrid[256] = {
    0x04040404u, 0x04040414u, 0x04040424u, 0x04040c0cu, 0x04040c1cu, 0x04040c3eu, 0x04041404u, 0x04041414u,
    0x04041c0cu, 0x04042414u, 0x04043e1cu, 0x04043e2cu, 0x040c040cu, 0x040c041cu, 0x040c0c04u, 0x040c0c14u,
    0x040c140cu, 0x040c142cu, 0x040c1c04u, 0x040c1c14u, 0x040c240cu, 0x040c2c24u, 0x040c3e04u, 0x04140404u,
    0x04140414u, 0x04140424u, 0x04140c0cu, 0x04141404u, 0x04141414u, 0x04141c0cu, 0x04141c1cu, 0x04141c3eu,
    0x04142c0cu, 0x04142c3eu, 0x04143e2cu, 0x041c040cu, 0x041c043eu, 0x041c0c04u, 0x041c0c14u, 0x041c142cu,
    0x041c3e04u, 0x04240c1cu, 0x04241c3eu, 0x04242424u, 0x04242c3eu, 0x04243e1cu, 0x04243e2cu, 0x042c040cu,
    0x042c043eu, 0x042c1c14u, 0x042c2c14u, 0x04341c2cu, 0x04343424u, 0x043e0c04u, 0x043e0c24u, 0x043e0c34u,
    0x043e241cu, 0x043e340cu, 0x0c04040cu, 0x0c04041cu, 0x0c040c04u, 0x0c040c14u, 0x0c04140cu, 0x0c04141cu,
    0x0c041c04u, 0x0c041c14u, 0x0c041c24u, 0x0c04243eu, 0x0c042c04u, 0x0c0c0404u, 0x0c0c0414u, 0x0c0c0c0cu,
    0x0c0c1404u, 0x0c0c1414u, 0x0c14040cu, 0x0c14041cu, 0x0c140c04u, 0x0c140c14u, 0x0c14140cu, 0x0c141c04u,
    0x0c143e14u, 0x0c1c0404u, 0x0c1c0414u, 0x0c1c1404u, 0x0c1c1c0cu, 0x0c1c2434u, 0x0c1c3434u, 0x0c24040cu,
    0x0c24042cu, 0x0c242c04u, 0x0c2c1404u, 0x0c2c1424u, 0x0c2c2434u, 0x0c2c3e0cu, 0x0c34042cu, 0x0c3e1414u,
    0x0c3e2404u, 0x14040404u, 0x14040414u, 0x14040c0cu, 0x14040c1cu, 0x14041404u, 0x14041414u, 0x14041434u,
    0x14041c0cu, 0x14042414u, 0x140c040cu, 0x140c041cu, 0x140c042cu, 0x140c0c04u, 0x140c0c14u, 0x140c140cu,
    0x140c1c04u, 0x140c341cu, 0x140c343eu, 0x140c3e04u, 0x14140404u, 0x14140414u, 0x14140c0cu, 0x14140c3eu,
    0x14141404u, 0x14141414u, 0x14141c3eu, 0x14142404u, 0x14142c2cu, 0x141c040cu, 0x141c0c04u, 0x141c0c24u,
    0x141c3e04u, 0x141c3e24u, 0x14241c2cu, 0x14242c1cu, 0x142c041cu, 0x142c143eu, 0x142c240cu, 0x142c3e24u,
    0x143e040cu, 0x143e041cu, 0x143e0c34u, 0x143e242cu, 0x1c04040cu, 0x1c040c04u, 0x1c040c14u, 0x1c04140cu,
    0x1c04141cu, 0x1c042c04u, 0x1c04342cu, 0x1c043e14u, 0x1c0c0404u, 0x1c0c0414u, 0x1c0c1404u, 0x1c0c1c0cu,
    0x1c0c2424u, 0x1c0c2434u, 0x1c14040cu, 0x1c14041cu, 0x1c140c04u, 0x1c14142cu, 0x1c142c14u, 0x1c143e14u,
    0x1c1c0c0cu, 0x1c1c1c1cu, 0x1c241c04u, 0x1c24243eu, 0x1c243e14u, 0x1c2c0404u, 0x1c2c0434u, 0x1c2c1414u,
    0x1c2c2c2cu, 0x1c340c24u, 0x1c341c34u, 0x1c34341cu, 0x1c3e1c1cu, 0x1c3e3404u, 0x24040424u, 0x24040c3eu,
    0x24041c2cu, 0x24041c3eu, 0x24042c1cu, 0x24042c3eu, 0x240c3e24u, 0x24141404u, 0x24141c3eu, 0x24142404u,
    0x24143404u, 0x24143434u, 0x241c043eu, 0x241c242cu, 0x24240424u, 0x24242c0cu, 0x24243424u, 0x242c142cu,
    0x242c241cu, 0x242c3e04u, 0x243e042cu, 0x243e0c04u, 0x243e0c14u, 0x243e1c04u, 0x2c040c14u, 0x2c04240cu,
    0x2c043e04u, 0x2c0c0404u, 0x2c0c0434u, 0x2c0c1434u, 0x2c0c2c2cu, 0x2c140c24u, 0x2c141c14u, 0x2c143e14u,
    0x2c1c0414u, 0x2c1c2c1cu, 0x2c240c04u, 0x2c24141cu, 0x2c24143eu, 0x2c243e14u, 0x2c2c0414u, 0x2c2c1c0cu,
    0x2c342c04u, 0x2c3e1424u, 0x2c3e2414u, 0x34041424u, 0x34042424u, 0x34042434u, 0x34043424u, 0x340c140cu,
    0x340c340cu, 0x34140c3eu, 0x34143424u, 0x341c1c04u, 0x341c1c34u, 0x34242424u, 0x342c042cu, 0x342c2c14u,
    0x34341c1cu, 0x343e041cu, 0x343e140cu, 0x3e04041cu, 0x3e04042cu, 0x3e04043eu, 0x3e040c04u, 0x3e041c14u,
    0x3e042c14u, 0x3e0c1434u, 0x3e0c2404u, 0x3e140c14u, 0x3e14242cu, 0x3e142c14u, 0x3e1c0404u, 0x3e1c0c2cu,
    0x3e1c1c1cu, 0x3e1c3404u, 0x3e24140cu, 0x3e24240cu, 0x3e2c0404u, 0x3e2c0414u, 0x3e2c1424u, 0x3e341c04u
};

constant float kMppIq4NlF32[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
       1.0f,   13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f
};

static inline size_t mpp_gguf_row_bytes(uint t, int K) {
    if (t == 12u) return (size_t)(K >> 8) * 144ul;
    if (t == 13u) return (size_t)(K >> 8) * 176ul;
    if (t == 14u) return (size_t)(K >> 8) * 210ul;
    if (t == 8u)  return (size_t)(K >> 5) * 34ul;
    if (t == 23u) return (size_t)(K >> 8) * 136ul;
    if (t == 18u) return (size_t)(K >> 8) * 98ul;
    if (t == 20u || t == 2u) return (size_t)(K >> 5) * 18ul;
    if (t == 11u) return (size_t)(K >> 8) * 110ul;
    if (t == 10u) return (size_t)(K >> 8) * 84ul;
    if (t == 39u) return (size_t)(K >> 5) * 17ul;
    if (t == 16u) return (size_t)(K >> 8) * 66ul;
    if (t == 17u) return (size_t)(K >> 8) * 74ul;
    if (t == 19u) return (size_t)(K >> 8) * 50ul;
    if (t == 21u) return (size_t)(K >> 8) * 110ul;
    if (t == 22u) return (size_t)(K >> 8) * 82ul;
    if (t == 42u) return (size_t)(K >> 6) * 18ul;
    if (t == 1u || t == 30u) return (size_t)K * 2ul;
    if (t == 0u)  return (size_t)K * 4ul;
    return 0ul;
}

static inline float mpp_gguf_q3k_scale(device const uchar* sc, int is) {
    int s = 0;
    if (is < 4)       s = int(sc[is] & 0xFu) | (int((sc[8 + is] >> 0) & 3u) << 4);
    else if (is < 8)  s = int(sc[is] & 0xFu) | (int((sc[8 + (is - 4)] >> 2) & 3u) << 4);
    else if (is < 12) s = int(sc[is - 8] >> 4) | (int((sc[8 + (is - 8)] >> 4) & 3u) << 4);
    else              s = int(sc[is - 8] >> 4) | (int((sc[8 + (is - 12)] >> 6) & 3u) << 4);
    return float(s - 32);
}

// Decodes 8 weights for ALL 20 native GGUF types (uses gguf_decode_8_rco from prepended ink_metal_iq_grids.metal!)
static inline void mpp_gguf_decode_8(uint t, device const uchar* row, int col, thread float4& w0, thread float4& w1) {
    if (t == 16u || t == 17u || t == 19u || t == 21u || t == 22u || t == 42u) {
        gguf_decode_8_rco(t, row, col, w0, w1);
        return;
    }
    if (t == 12u) {  // Q4_K
        const int b = col >> 8, idx = col & 255;
        device const uchar* blk = row + (size_t)b * 144ul;
        const float d = float(as_type<half>(mpp_ld16(blk)));
        const float dmin = float(as_type<half>(mpp_ld16(blk + 2)));
        device const uchar* sc = blk + 4;
        device const uchar* qs = blk + 16;
        const int j64 = idx >> 6, in64 = idx & 63, sub = j64 * 2 + (in64 >> 5);
        uchar sd, sm;
        if (sub < 4) { sd = sc[sub] & 63u; sm = sc[sub + 4] & 63u; }
        else { sd = (sc[sub + 4] & 0xFu) | ((sc[sub - 4] >> 6) << 4); sm = (sc[sub + 4] >> 4) | ((sc[sub] >> 6) << 4); }
        const float dl = d * float(sd), ml = -dmin * float(sm);
        device const uchar* qp = qs + j64 * 32 + (in64 & 31);
        if (in64 < 32) {
            w0 = float4(float(qp[0] & 15u), float(qp[1] & 15u), float(qp[2] & 15u), float(qp[3] & 15u)) * dl + ml;
            w1 = float4(float(qp[4] & 15u), float(qp[5] & 15u), float(qp[6] & 15u), float(qp[7] & 15u)) * dl + ml;
        } else {
            w0 = float4(float(qp[0] >> 4), float(qp[1] >> 4), float(qp[2] >> 4), float(qp[3] >> 4)) * dl + ml;
            w1 = float4(float(qp[4] >> 4), float(qp[5] >> 4), float(qp[6] >> 4), float(qp[7] >> 4)) * dl + ml;
        }
    } else if (t == 13u) {  // Q5_K
        const int b = col >> 8, idx = col & 255;
        device const uchar* blk = row + (size_t)b * 176ul;
        const float d = float(as_type<half>(mpp_ld16(blk)));
        const float dmin = float(as_type<half>(mpp_ld16(blk + 2)));
        device const uchar* sc = blk + 4;
        device const uchar* qh = blk + 16;
        device const uchar* qs = blk + 48;
        const int j64 = idx >> 6, in64 = idx & 63, sub = j64 * 2 + (in64 >> 5), l = in64 & 31;
        uchar sd, sm;
        if (sub < 4) { sd = sc[sub] & 63u; sm = sc[sub + 4] & 63u; }
        else { sd = (sc[sub + 4] & 0xFu) | ((sc[sub - 4] >> 6) << 4); sm = (sc[sub + 4] >> 4) | ((sc[sub] >> 6) << 4); }
        const float dl = d * float(sd), ml = -dmin * float(sm);
        device const uchar* qp = qs + j64 * 32 + l;
        device const uchar* hp = qh + l;
        const uint sh = (in64 < 32) ? 0u : 4u;
        float qv[8];
        for (int k = 0; k < 8; ++k) {
            const uint ql = (uint(qp[k]) >> sh) & 15u;
            const uint hb = (uint(hp[k]) >> uint(sub)) & 1u;
            qv[k] = float(ql | (hb << 4)) * dl + ml;
        }
        w0 = float4(qv[0], qv[1], qv[2], qv[3]);
        w1 = float4(qv[4], qv[5], qv[6], qv[7]);
    } else if (t == 14u) {  // Q6_K
        const int b = col >> 8, idx = col & 255;
        device const uchar* blk = row + (size_t)b * 210ul;
        const float d = float(as_type<half>(mpp_ld16(blk + 208)));
        const int hf = idx >> 7, r = idx & 127, qg = r >> 5, l = r & 31, is = l >> 4;
        device const uchar* qlh = blk + hf * 64 + ((qg & 1) ? 32 : 0) + l;
        device const uchar* qhh = blk + 128 + hf * 32 + l;
        device const char* sch = (device const char*)(blk + 192 + hf * 8);
        const float dl = d * float(sch[is + qg * 2]);
        const uint qsh = (qg >= 2) ? 4u : 0u;
        const uint hsh = uint(qg * 2);
        float qv[8];
        for (int k = 0; k < 8; ++k) {
            const int q = int(((uint(qlh[k]) >> qsh) & 15u) | (((uint(qhh[k]) >> hsh) & 3u) << 4)) - 32;
            qv[k] = dl * float(q);
        }
        w0 = float4(qv[0], qv[1], qv[2], qv[3]);
        w1 = float4(qv[4], qv[5], qv[6], qv[7]);
    } else if (t == 8u) {  // Q8_0
        const int b = col >> 5, in32 = col & 31;
        device const uchar* blk = row + (size_t)b * 34ul;
        const float d = float(as_type<half>(mpp_ld16(blk)));
        device const char* qs = (device const char*)(blk + 2 + in32);
        w0 = float4(float(qs[0]), float(qs[1]), float(qs[2]), float(qs[3])) * d;
        w1 = float4(float(qs[4]), float(qs[5]), float(qs[6]), float(qs[7])) * d;
    } else if (t == 23u) {  // IQ4_XS
        const int b = col >> 8, idx = col & 255;
        device const uchar* blk = row + (size_t)b * 136ul;
        const float d = float(as_type<half>(mpp_ld16(blk)));
        const ushort sh = mpp_ld16(blk + 2);
        device const uchar* sl = blk + 4;
        device const uchar* qs = blk + 8;
        const int ib = idx >> 5, in32 = idx & 31;
        const int ls = int((sl[ib >> 1] >> (4 * (ib & 1))) & 15u) | (int((sh >> (2 * ib)) & 3u) << 4);
        const float dl = d * float(ls - 32);
        device const uchar* qp = qs + ib * 16 + (in32 & 15);
        if (in32 < 16) {
            w0 = float4(kMppIq4NlF32[qp[0] & 15u], kMppIq4NlF32[qp[1] & 15u], kMppIq4NlF32[qp[2] & 15u], kMppIq4NlF32[qp[3] & 15u]) * dl;
            w1 = float4(kMppIq4NlF32[qp[4] & 15u], kMppIq4NlF32[qp[5] & 15u], kMppIq4NlF32[qp[6] & 15u], kMppIq4NlF32[qp[7] & 15u]) * dl;
        } else {
            w0 = float4(kMppIq4NlF32[qp[0] >> 4], kMppIq4NlF32[qp[1] >> 4], kMppIq4NlF32[qp[2] >> 4], kMppIq4NlF32[qp[3] >> 4]) * dl;
            w1 = float4(kMppIq4NlF32[qp[4] >> 4], kMppIq4NlF32[qp[5] >> 4], kMppIq4NlF32[qp[6] >> 4], kMppIq4NlF32[qp[7] >> 4]) * dl;
        }
    } else if (t == 18u) {  // IQ3_XXS
        const int b = col >> 8, idx = col & 255;
        device const uchar* blk = row + (size_t)b * 98ul;
        const float d = float(as_type<half>(mpp_ld16(blk)));
        device const uchar* qs = blk + 2;
        device const uchar* ss = blk + 66;
        const int ib32 = idx >> 5, l = (idx & 31) >> 3;
        const uint aux32 = uint(ss[4 * ib32 + 0]) | (uint(ss[4 * ib32 + 1]) << 8) |
                           (uint(ss[4 * ib32 + 2]) << 16) | (uint(ss[4 * ib32 + 3]) << 24);
        const float db = d * (0.5f + float(aux32 >> 28)) * 0.5f;
        const uchar signs = kMppIq3XxsSigns[(aux32 >> uint(7 * l)) & 127u];
        const uint g1 = kMppIq3XxsGrid[qs[ib32 * 8 + 2 * l + 0]];
        const uint g2 = kMppIq3XxsGrid[qs[ib32 * 8 + 2 * l + 1]];
        w0 = float4(
            float((g1 >>  0) & 255u) * ((signs &   1u) ? -db : db),
            float((g1 >>  8) & 255u) * ((signs &   2u) ? -db : db),
            float((g1 >> 16) & 255u) * ((signs &   4u) ? -db : db),
            float((g1 >> 24) & 255u) * ((signs &   8u) ? -db : db));
        w1 = float4(
            float((g2 >>  0) & 255u) * ((signs &  16u) ? -db : db),
            float((g2 >>  8) & 255u) * ((signs &  32u) ? -db : db),
            float((g2 >> 16) & 255u) * ((signs &  64u) ? -db : db),
            float((g2 >> 24) & 255u) * ((signs & 128u) ? -db : db));
    } else if (t == 20u) {  // IQ4_NL
        const int b = col >> 5, in32 = col & 31;
        device const uchar* blk = row + (size_t)b * 18ul;
        const float d = float(as_type<half>(mpp_ld16(blk)));
        device const uchar* qp = blk + 2 + (in32 & 15);
        if (in32 < 16) {
            w0 = float4(kMppIq4NlF32[qp[0] & 15u], kMppIq4NlF32[qp[1] & 15u], kMppIq4NlF32[qp[2] & 15u], kMppIq4NlF32[qp[3] & 15u]) * d;
            w1 = float4(kMppIq4NlF32[qp[4] & 15u], kMppIq4NlF32[qp[5] & 15u], kMppIq4NlF32[qp[6] & 15u], kMppIq4NlF32[qp[7] & 15u]) * d;
        } else {
            w0 = float4(kMppIq4NlF32[qp[0] >> 4], kMppIq4NlF32[qp[1] >> 4], kMppIq4NlF32[qp[2] >> 4], kMppIq4NlF32[qp[3] >> 4]) * d;
            w1 = float4(kMppIq4NlF32[qp[4] >> 4], kMppIq4NlF32[qp[5] >> 4], kMppIq4NlF32[qp[6] >> 4], kMppIq4NlF32[qp[7] >> 4]) * d;
        }
    } else if (t == 2u) {  // Q4_0
        const int b = col >> 5, in32 = col & 31;
        device const uchar* blk = row + (size_t)b * 18ul;
        const float d = float(as_type<half>(mpp_ld16(blk)));
        device const uchar* qp = blk + 2 + (in32 & 15);
        if (in32 < 16) {
            w0 = (float4(float(qp[0] & 15u), float(qp[1] & 15u), float(qp[2] & 15u), float(qp[3] & 15u)) - 8.0f) * d;
            w1 = (float4(float(qp[4] & 15u), float(qp[5] & 15u), float(qp[6] & 15u), float(qp[7] & 15u)) - 8.0f) * d;
        } else {
            w0 = (float4(float(qp[0] >> 4), float(qp[1] >> 4), float(qp[2] >> 4), float(qp[3] >> 4)) - 8.0f) * d;
            w1 = (float4(float(qp[4] >> 4), float(qp[5] >> 4), float(qp[6] >> 4), float(qp[7] >> 4)) - 8.0f) * d;
        }
    } else if (t == 11u) {  // Q3_K
        const int b = col >> 8, idx = col & 255;
        device const uchar* blk = row + (size_t)b * 110ul;
        device const uchar* hm = blk;
        device const uchar* qs = blk + 32;
        device const uchar* sc = blk + 96;
        const float d_all = float(as_type<half>(mpp_ld16(blk + 108)));
        const int n128 = idx >> 7, in128 = idx & 127, j = in128 >> 5, l32 = in128 & 31;
        const int is = n128 * 8 + j * 2 + (l32 >> 4);
        const float dl = d_all * mpp_gguf_q3k_scale(sc, is);
        const uint shift = uint(j * 2);
        const uchar mask = uchar(1u << uint(n128 * 4 + j));
        device const uchar* qp = qs + n128 * 32 + l32;
        device const uchar* hp = hm + l32;
        float qv[8];
        for (int k = 0; k < 8; ++k) {
            const int q2 = int((qp[k] >> shift) & 3u);
            const int hbit = (hp[k] & mask) ? 0 : 4;
            qv[k] = dl * float(q2 - hbit);
        }
        w0 = float4(qv[0], qv[1], qv[2], qv[3]);
        w1 = float4(qv[4], qv[5], qv[6], qv[7]);
    } else if (t == 10u) {  // Q2_K
        const int b = col >> 8, idx = col & 255;
        device const uchar* blk = row + (size_t)b * 84ul;
        device const uchar* sc = blk;
        device const uchar* qs = blk + 16;
        const float d = float(as_type<half>(mpp_ld16(blk + 80)));
        const float dmin = float(as_type<half>(mpp_ld16(blk + 82)));
        const int n128 = idx >> 7, in128 = idx & 127, j = in128 >> 5, l32 = in128 & 31;
        const int is = n128 * 8 + j * 2 + (l32 >> 4);
        const uchar scb = sc[is];
        const float dl = d * float(scb & 15u), ml = -dmin * float(scb >> 4);
        const uint shift = uint(j * 2);
        device const uchar* qp = qs + n128 * 32 + l32;
        w0 = float4(float((qp[0] >> shift) & 3u), float((qp[1] >> shift) & 3u), float((qp[2] >> shift) & 3u), float((qp[3] >> shift) & 3u)) * dl + ml;
        w1 = float4(float((qp[4] >> shift) & 3u), float((qp[5] >> shift) & 3u), float((qp[6] >> shift) & 3u), float((qp[7] >> shift) & 3u)) * dl + ml;
    } else if (t == 39u) {  // MXFP4
        const int b = col >> 5, in32 = col & 31;
        device const uchar* blk = row + (size_t)b * 17ul;
        const uint e = blk[0];
        const float d = (e < 2u) ? as_type<float>(0x00200000u << e) : as_type<float>((e - 1u) << 23);
        const float d2 = 2.0f * d;
        device const uchar* qp = blk + 1 + (in32 & 15);
        if (in32 < 16) {
            w0 = float4(kMppE2M1[qp[0] & 15u], kMppE2M1[qp[1] & 15u], kMppE2M1[qp[2] & 15u], kMppE2M1[qp[3] & 15u]) * d2;
            w1 = float4(kMppE2M1[qp[4] & 15u], kMppE2M1[qp[5] & 15u], kMppE2M1[qp[6] & 15u], kMppE2M1[qp[7] & 15u]) * d2;
        } else {
            w0 = float4(kMppE2M1[qp[0] >> 4], kMppE2M1[qp[1] >> 4], kMppE2M1[qp[2] >> 4], kMppE2M1[qp[3] >> 4]) * d2;
            w1 = float4(kMppE2M1[qp[4] >> 4], kMppE2M1[qp[5] >> 4], kMppE2M1[qp[6] >> 4], kMppE2M1[qp[7] >> 4]) * d2;
        }
    } else if (t == 30u) {  // BF16
        device const ushort* wp = (device const ushort*)(row + (size_t)col * 2ul);
        w0 = float4(mpp_bf16_to_f32(wp[0]), mpp_bf16_to_f32(wp[1]), mpp_bf16_to_f32(wp[2]), mpp_bf16_to_f32(wp[3]));
        w1 = float4(mpp_bf16_to_f32(wp[4]), mpp_bf16_to_f32(wp[5]), mpp_bf16_to_f32(wp[6]), mpp_bf16_to_f32(wp[7]));
    } else if (t == 1u) {   // F16
        device const half* wp = (device const half*)(row + (size_t)col * 2ul);
        w0 = float4(wp[0], wp[1], wp[2], wp[3]);
        w1 = float4(wp[4], wp[5], wp[6], wp[7]);
    } else if (t == 0u) {   // F32
        device const float4* wp = (device const float4*)(row + (size_t)col * 4ul);
        w0 = wp[0]; w1 = wp[1];
    } else {
        w0 = float4(0.0f); w1 = float4(0.0f);
    }
}

static inline void mpp_dec_gguf_32(device const uchar* row_ptr, uint t, float sc, int col, threadgroup half4* dst) {
    for (int q = 0; q < 4; ++q) {
        float4 w0, w1;
        mpp_gguf_decode_8(t, row_ptr, col + q * 8, w0, w1);
        dst[q * 2 + 0] = half4(w0 * sc);
        dst[q * 2 + 1] = half4(w1 * sc);
    }
}

static inline float4 mpp_sf4_shfl_4(float e2m1_lane, uint nibs16) {
    return float4(
        simd_shuffle(e2m1_lane, ushort(nibs16 & 15u)),
        simd_shuffle(e2m1_lane, ushort((nibs16 >> 4) & 15u)),
        simd_shuffle(e2m1_lane, ushort((nibs16 >> 8) & 15u)),
        simd_shuffle(e2m1_lane, ushort((nibs16 >> 12) & 15u)));
}

static inline void mpp_dec_sf4_32_shfl(device const uchar* p_base, device const uchar* s_base, float outer,
                                       int row_pbytes, int row_sbytes, int gn, int k0, int half_idx,
                                       float e2m1_lane, threadgroup half4* dst) {
    const int blk = (k0 >> 4) + half_idx * 2;
    const float sc0 = mpp_e4m3_to_f32(s_base[(size_t)gn * row_sbytes + blk + 0]) * outer;
    const float sc1 = mpp_e4m3_to_f32(s_base[(size_t)gn * row_sbytes + blk + 1]) * outer;
    const uint4 pk = *((device const uint4*)(p_base + (size_t)gn * row_pbytes + (size_t)blk * 8));
    dst[0] = half4(mpp_sf4_shfl_4(e2m1_lane, pk.x) * sc0);
    dst[1] = half4(mpp_sf4_shfl_4(e2m1_lane, pk.x >> 16) * sc0);
    dst[2] = half4(mpp_sf4_shfl_4(e2m1_lane, pk.y) * sc0);
    dst[3] = half4(mpp_sf4_shfl_4(e2m1_lane, pk.y >> 16) * sc0);
    dst[4] = half4(mpp_sf4_shfl_4(e2m1_lane, pk.z) * sc1);
    dst[5] = half4(mpp_sf4_shfl_4(e2m1_lane, pk.z >> 16) * sc1);
    dst[6] = half4(mpp_sf4_shfl_4(e2m1_lane, pk.w) * sc1);
    dst[7] = half4(mpp_sf4_shfl_4(e2m1_lane, pk.w >> 16) * sc1);
}

// 5-bucket binary MoE tile planner (128 / 64 / 32 / 16 / 8): adds TM=8 for small-M (M=6..16) with <=7 wasted rows!
kernel void ink_mpp_moe_plan_v2(
    device const int* topk_idx [[buffer(0)]],
    constant int& P            [[buffer(1)]],
    constant int& allow_128    [[buffer(2)]],
    constant int& allow_8      [[buffer(3)]],
    device int* srt_pair       [[buffer(4)]],
    device int* pair_pos       [[buffer(5)]],
    device int4* tiles_128     [[buffer(6)]],
    device int4* tiles_64      [[buffer(7)]],
    device int4* tiles_32      [[buffer(8)]],
    device int4* tiles_16      [[buffer(9)]],
    device int4* tiles_8       [[buffer(10)]],
    device uint* meta          [[buffer(11)]],  // 28 uints
    uint e    [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup int sg_r[4], sg_128[4], sg_64[4], sg_32[4], sg_16[4], sg_8[4];

    int c = 0;
    for (int i = 0; i < P; ++i) if (topk_idx[i] == int(e)) ++c;

    int n128 = 0, n64 = 0, n32 = 0, n16 = 0, n8 = 0;
    if (c > 0) {
        int rem = c;
        if (allow_128) {
            n128 = rem >> 7; rem &= 127;
            if (rem > 112) { ++n128; rem = 0; }
        }
        if (rem > 0) {
            n64 = rem >> 6; rem &= 63;
            if (rem > 48) { ++n64; rem = 0; }
        }
        if (rem > 0) {
            n32 = rem >> 5; rem &= 31;
            if (rem > 16) { ++n32; rem = 0; }
        }
        if (rem > 0) {
            if (allow_8 && rem <= 8) { n8 = 1; }
            else { n16 = 1; }
        }
    }
    const int rows = n128 * 128 + n64 * 64 + n32 * 32 + n16 * 16 + n8 * 8;
    const int ex_r   = simd_prefix_exclusive_sum(rows);
    const int ex_128 = simd_prefix_exclusive_sum(n128);
    const int ex_64  = simd_prefix_exclusive_sum(n64);
    const int ex_32  = simd_prefix_exclusive_sum(n32);
    const int ex_16  = simd_prefix_exclusive_sum(n16);
    const int ex_8   = simd_prefix_exclusive_sum(n8);
    if (lane == 31u) {
        sg_r[sg] = ex_r + rows; sg_128[sg] = ex_128 + n128; sg_64[sg] = ex_64 + n64;
        sg_32[sg] = ex_32 + n32; sg_16[sg] = ex_16 + n16; sg_8[sg] = ex_8 + n8;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int br = 0, b128 = 0, b64 = 0, b32 = 0, b16 = 0, b8 = 0;
    for (uint w = 0; w < sg; ++w) {
        br += sg_r[w]; b128 += sg_128[w]; b64 += sg_64[w]; b32 += sg_32[w]; b16 += sg_16[w]; b8 += sg_8[w];
    }
    const int roff = br + ex_r, o128 = b128 + ex_128, o64 = b64 + ex_64, o32 = b32 + ex_32, o16 = b16 + ex_16, o8 = b8 + ex_8;
    int cur_r = roff, rem_c = c;
    for (int j = 0; j < n128; ++j) { const int v = min(128, rem_c); tiles_128[o128 + j] = int4(int(e), cur_r, v, 0); cur_r += 128; rem_c -= v; }
    for (int j = 0; j < n64;  ++j) { const int v = min(64, rem_c);  tiles_64[o64 + j]   = int4(int(e), cur_r, v, 0); cur_r += 64;  rem_c -= v; }
    for (int j = 0; j < n32;  ++j) { const int v = min(32, rem_c);  tiles_32[o32 + j]   = int4(int(e), cur_r, v, 0); cur_r += 32;  rem_c -= v; }
    for (int j = 0; j < n16;  ++j) { const int v = min(16, rem_c);  tiles_16[o16 + j]   = int4(int(e), cur_r, v, 0); cur_r += 16;  rem_c -= v; }
    for (int j = 0; j < n8;   ++j) { const int v = min(8, rem_c);   tiles_8[o8 + j]     = int4(int(e), cur_r, v, 0); cur_r += 8;   rem_c -= v; }

    for (int r = 0; r < rows; ++r) srt_pair[roff + r] = -1;
    if (c > 0) {
        int k = 0;
        for (int i = 0; i < P; ++i) if (topk_idx[i] == int(e)) { const int row = roff + k++; srt_pair[row] = i; pair_pos[i] = row; }
    }
    if (e == 127u) {
        const uint t128 = uint(o128 + n128), t64 = uint(o64 + n64), t32 = uint(o32 + n32);
        const uint t16  = uint(o16 + n16),   t8  = uint(o8 + n8),   tot_rows = uint(roff + rows);
        meta[0] = t128; meta[1] = t64; meta[2] = t32; meta[3] = tot_rows;
        meta[4] = tot_rows; meta[5] = 1u; meta[6] = 1u;
        meta[7] = t128; meta[8] = 36u; meta[9] = 1u;
        meta[10] = t64; meta[11] = 36u; meta[12] = 1u;
        meta[13] = t32; meta[14] = 36u; meta[15] = 1u;
        meta[16] = t16; meta[17] = 36u; meta[18] = 1u;
        meta[19] = t16;
        meta[20] = t8;  meta[21] = 36u; meta[22] = 1u;
        meta[23] = t8;
    }
}

// 16 KB Double-Buffered SLM Grouped-Expert w13 + fused SITU (overlaps decode(k0+64) with NAX matmul2d(k0)!)
template <int CODEC, int TM, int NTILE>
kernel void ink_mpp_moe_gu_dbuf(
    device const uchar* sec4           [[buffer(0)]],
    device const InkExpertDevMpp* ex   [[buffer(1)]],
    device half* XS                    [[buffer(2)]],
    device half* HS                    [[buffer(3)]],
    device const int4* tiles           [[buffer(4)]],
    device const uchar* sec4_1         [[buffer(5)]],
    device const uchar* sec4_2         [[buffer(6)]],
    device const uchar* sec4_3         [[buffer(7)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int TN = 64, TK = 64, SG = 4, NTH = 128;
    constexpr int DBUF_HALVES = 2 * TN * TK;   // 16 KB double-buffered sB[2][64*64]
    constexpr int O_HALVES = 2 * TM * TN;      // <= 16 KB
    constexpr int S_HALVES = DBUF_HALVES > O_HALVES ? DBUF_HALVES : O_HALVES;
    threadgroup half sMem[S_HALVES];

    const float e2m1_lane = kMppE2M1[tid & 15u];
    const int4 tl = tiles[tg.x];
    const int e = tl.x, row0 = tl.y;
    const int n0 = int(tg.y) * TN;
    const InkExpertDevMpp ed = ex[e];
    device const uchar* body = ink_mpp_sec4_ptr(ed.chunk_idx, sec4, sec4_1, sec4_2, sec4_3) + ed.body_off;
    const bool is_sk_dn = (ed.dn_code != 15 && ed.dn_code != 16 && ed.dn_code != 20);
    const int gcode = (CODEC == 0) ? int(ed.gu_code) : CODEC;
    const float sf4_outer = (gcode == 15) ? (((device const float*)body)[0] / ((device const float*)body)[1]) : 0.0f;
    const size_t sk_rowb = (CODEC == 0 && gcode != 15 && gcode != 16 && gcode != 20) ? (9ul * size_t(mpp_sk_sb_bytes(gcode))) : 0ul;
    const uint gguf_t = (CODEC == 0 && gcode == 20) ? ((device const InkMppGgufHdr*)body)->gu_type : 0u;
    const size_t gguf_rb = (CODEC == 0 && gcode == 20) ? mpp_gguf_row_bytes(gguf_t, 2304) : 0ul;
    const uint gguf_fl = (CODEC == 0 && gcode == 20) ? ((device const InkMppGgufHdr*)body)->flags : 0u;
    const float gguf_gs = (CODEC == 0 && gcode == 20) ? ((device const InkMppGgufHdr*)body)->gate_scale : 1.0f;
    const float gguf_us = (CODEC == 0 && gcode == 20) ? ((device const InkMppGgufHdr*)body)->up_scale : 1.0f;

    const int br = int(tid >> 1), hi = int(tid & 1u);
    const int gn = n0 + br;

    auto decode_tile = [&](int k0, int buf_idx) {
        threadgroup half4* d = (threadgroup half4*)(sMem + buf_idx * (TN * TK) + br * TK + hi * 32);
        if constexpr (CODEC == 15) {
            mpp_dec_sf4_32_shfl(body + 16, body + 16 + 2654208, sf4_outer, 1152, 144, gn, k0, hi, e2m1_lane, d);
        } else if constexpr (CODEC == 16) {
            mpp_dec_mx8_32(body, body + 5308416, 2304, 72, gn, k0, hi, d);
        } else {
            if (gcode == 15) {
                mpp_dec_sf4_32_shfl(body + 16, body + 16 + 2654208, sf4_outer, 1152, 144, gn, k0, hi, e2m1_lane, d);
            } else if (gcode == 16) {
                mpp_dec_mx8_32(body, body + 5308416, 2304, 72, gn, k0, hi, d);
            } else if (gcode == 20) {
                const int src_row = (gguf_fl & 1u) ? gn : (((gn & 1) ? 1152 : 0) + (gn >> 1));
                const float sc = (gn & 1) ? gguf_us : gguf_gs;
                mpp_dec_gguf_32(body + 32 + (size_t)src_row * gguf_rb, gguf_t, sc, k0 + hi * 32, d);
            } else {
                mpp_dec_sk_32(body + 2304 + (size_t)gn * sk_rowb, gcode, k0 + hi * 32, d);
            }
        }
    };

    tensor<device half, dextents<int32_t, 2>, tensor_inline> tX(XS, dextents<int32_t, 2>(2304, 1 << 24));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB0(sMem, dextents<int32_t, 2>(TK, TN));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB1(sMem + TN * TK, dextents<int32_t, 2>(TK, TN));
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto tA0 = tX.template slice<TK, TM>(0, row0);
    auto acc0 = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), float>();
    auto acc1 = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), float>();
    for (uint16_t i = 0; i < acc0.get_capacity(); ++i) {
        if (acc0.is_valid_element(i)) {
            acc0[i] = 0.0f;
            if constexpr (NTILE == 2) acc1[i] = 0.0f;
        }
    }

    // Prologue: decode k0=0 into buffer 0
    decode_tile(0, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int cur = 0;
    for (int k0 = 0; k0 < 2304; k0 += TK) {
        const int nxt = 1 - cur;
        if (k0 + TK < 2304) decode_tile(k0 + TK, nxt);
        auto mA0 = tX.template slice<TK, TM>(k0, row0);
        if (cur == 0) {
            op.run(mA0, tB0, acc0);
            if constexpr (NTILE == 2) { auto mA1 = tX.template slice<TK, TM>(k0, row0 + TM); op.run(mA1, tB0, acc1); }
        } else {
            op.run(mA0, tB1, acc0);
            if constexpr (NTILE == 2) { auto mA1 = tX.template slice<TK, TM>(k0, row0 + TM); op.run(mA1, tB1, acc1); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        cur = nxt;
    }

    threadgroup float* ot = reinterpret_cast<threadgroup float*>(sMem);
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(ot, dextents<int32_t, 2>(TN, TM));
    constexpr int PAIRS = TM * 32;
    const int fbase = n0 >> 1;

    acc0.store(tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = int(tid); idx < PAIRS; idx += NTH) {
        const int m = idx >> 5, f = idx & 31;
        const float g = ot[m * TN + 2 * f], u = ot[m * TN + 2 * f + 1];
        const float inv_sq = is_sk_dn ? (1.0f / float(as_type<half>(mpp_ld16(body + 2 * (fbase + f))))) : 1.0f;
        HS[(size_t)(row0 + m) * 1152 + fbase + f] = half(mpp_situ_act(g, u) * inv_sq);
    }
    if constexpr (NTILE == 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc1.store(tO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = int(tid); idx < PAIRS; idx += NTH) {
            const int m = idx >> 5, f = idx & 31;
            const float g = ot[m * TN + 2 * f], u = ot[m * TN + 2 * f + 1];
            const float inv_sq = is_sk_dn ? (1.0f / float(as_type<half>(mpp_ld16(body + 2 * (fbase + f))))) : 1.0f;
            HS[(size_t)(row0 + TM + m) * 1152 + fbase + f] = half(mpp_situ_act(g, u) * inv_sq);
        }
    }
}

template <int CODEC, int TM, int NTILE>
kernel void ink_mpp_moe_dn_dbuf(
    device const uchar* sec4           [[buffer(0)]],
    device const InkExpertDevMpp* ex   [[buffer(1)]],
    device half* HS                    [[buffer(2)]],
    device float* YE                   [[buffer(3)]],
    device const int4* tiles           [[buffer(4)]],
    device const uchar* sec4_1         [[buffer(5)]],
    device const uchar* sec4_2         [[buffer(6)]],
    device const uchar* sec4_3         [[buffer(7)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int TN = 64, TK = 64, SG = 4;
    threadgroup half sB[2 * TN * TK];

    const float e2m1_lane = kMppE2M1[tid & 15u];
    const int4 tl = tiles[tg.x];
    const int e = tl.x, row0 = tl.y;
    const int d0 = int(tg.y) * TN;
    const InkExpertDevMpp ed = ex[e];
    device const uchar* body = ink_mpp_sec4_ptr(ed.chunk_idx, sec4, sec4_1, sec4_2, sec4_3) + ed.body_off;
    const int dcode = (CODEC == 0) ? int(ed.dn_code) : CODEC;
    const float sf4_outer = (dcode == 15) ? (((device const float*)body)[2] / ((device const float*)body)[3]) : 0.0f;
    const size_t sk_pairb = (CODEC == 0 && dcode != 15 && dcode != 16 && dcode != 20) ? (9ul * size_t(mpp_sk_sb_bytes(dcode))) : 0ul;
    const uint gguf_t = (CODEC == 0 && dcode == 20) ? ((device const InkMppGgufHdr*)body)->dn_type : 0u;
    const size_t gguf_rb = (CODEC == 0 && dcode == 20) ? mpp_gguf_row_bytes(gguf_t, 1152) : 0ul;
    const float gguf_ds = (CODEC == 0 && dcode == 20) ? ((device const InkMppGgufHdr*)body)->dn_scale : 1.0f;
    const int br = int(tid >> 1), hi = int(tid & 1u);
    const int gn = d0 + br;

    auto decode_tile = [&](int k0, int buf_idx) {
        threadgroup half4* d = (threadgroup half4*)(sB + buf_idx * (TN * TK) + br * TK + hi * 32);
        if constexpr (CODEC == 15) {
            mpp_dec_sf4_32_shfl(body + ed.dn_off, body + ed.dn_off + 1327104, sf4_outer, 576, 72, gn, k0, hi, e2m1_lane, d);
        } else if constexpr (CODEC == 16) {
            mpp_dec_mx8_32(body + ed.dn_off, body + ed.dn_off + 2654208, 1152, 36, gn, k0, hi, d);
        } else {
            if (dcode == 15) {
                mpp_dec_sf4_32_shfl(body + ed.dn_off, body + ed.dn_off + 1327104, sf4_outer, 576, 72, gn, k0, hi, e2m1_lane, d);
            } else if (dcode == 16) {
                mpp_dec_mx8_32(body + ed.dn_off, body + ed.dn_off + 2654208, 1152, 36, gn, k0, hi, d);
            } else if (dcode == 20) {
                mpp_dec_gguf_32(body + ed.dn_off + (size_t)gn * gguf_rb, gguf_t, gguf_ds, k0 + hi * 32, d);
            } else {
                const int rp = gn >> 1;
                const int col_p = ((gn & 1) ? 1152 : 0) + k0 + hi * 32;
                mpp_dec_sk_32(body + ed.dn_off + (size_t)rp * sk_pairb, dcode, col_p, d);
            }
        }
    };

    tensor<device half,  dextents<int32_t, 2>, tensor_inline> tH(HS, dextents<int32_t, 2>(1152, 1 << 24));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(YE, dextents<int32_t, 2>(2304, 1 << 24));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB0(sB, dextents<int32_t, 2>(TK, TN));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB1(sB + TN * TK, dextents<int32_t, 2>(TK, TN));
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto tA0 = tH.template slice<TK, TM>(0, row0);
    auto acc0 = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), float>();
    auto acc1 = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), float>();
    for (uint16_t i = 0; i < acc0.get_capacity(); ++i) {
        if (acc0.is_valid_element(i)) {
            acc0[i] = 0.0f;
            if constexpr (NTILE == 2) acc1[i] = 0.0f;
        }
    }

    decode_tile(0, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int cur = 0;
    for (int k0 = 0; k0 < 1152; k0 += TK) {
        const int nxt = 1 - cur;
        if (k0 + TK < 1152) decode_tile(k0 + TK, nxt);
        auto mA0 = tH.template slice<TK, TM>(k0, row0);
        if (cur == 0) {
            op.run(mA0, tB0, acc0);
            if constexpr (NTILE == 2) { auto mA1 = tH.template slice<TK, TM>(k0, row0 + TM); op.run(mA1, tB0, acc1); }
        } else {
            op.run(mA0, tB1, acc0);
            if constexpr (NTILE == 2) { auto mA1 = tH.template slice<TK, TM>(k0, row0 + TM); op.run(mA1, tB1, acc1); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        cur = nxt;
    }
    acc0.store(tY.slice(d0, row0));
    if constexpr (NTILE == 2) acc1.store(tY.slice(d0, row0 + TM));
}

#define INK_MPP_DBUF_SET(CNAME, CODE) \
    template [[host_name("ink_mpp_moe_gu_dbuf_" CNAME "_8")]]   kernel decltype(ink_mpp_moe_gu_dbuf<CODE, 8, 1>)  ink_mpp_moe_gu_dbuf<CODE, 8, 1>; \
    template [[host_name("ink_mpp_moe_dn_dbuf_" CNAME "_8")]]   kernel decltype(ink_mpp_moe_dn_dbuf<CODE, 8, 1>)  ink_mpp_moe_dn_dbuf<CODE, 8, 1>; \
    template [[host_name("ink_mpp_moe_gu_dbuf_" CNAME "_16")]]  kernel decltype(ink_mpp_moe_gu_dbuf<CODE, 16, 1>) ink_mpp_moe_gu_dbuf<CODE, 16, 1>; \
    template [[host_name("ink_mpp_moe_dn_dbuf_" CNAME "_16")]]  kernel decltype(ink_mpp_moe_dn_dbuf<CODE, 16, 1>) ink_mpp_moe_dn_dbuf<CODE, 16, 1>; \
    template [[host_name("ink_mpp_moe_gu_dbuf_" CNAME "_32")]]  kernel decltype(ink_mpp_moe_gu_dbuf<CODE, 32, 1>) ink_mpp_moe_gu_dbuf<CODE, 32, 1>; \
    template [[host_name("ink_mpp_moe_dn_dbuf_" CNAME "_32")]]  kernel decltype(ink_mpp_moe_dn_dbuf<CODE, 32, 1>) ink_mpp_moe_dn_dbuf<CODE, 32, 1>; \
    template [[host_name("ink_mpp_moe_gu_dbuf_" CNAME "_64")]]  kernel decltype(ink_mpp_moe_gu_dbuf<CODE, 64, 1>) ink_mpp_moe_gu_dbuf<CODE, 64, 1>; \
    template [[host_name("ink_mpp_moe_dn_dbuf_" CNAME "_64")]]  kernel decltype(ink_mpp_moe_dn_dbuf<CODE, 64, 1>) ink_mpp_moe_dn_dbuf<CODE, 64, 1>; \
    template [[host_name("ink_mpp_moe_gu_dbuf_" CNAME "_128")]] kernel decltype(ink_mpp_moe_gu_dbuf<CODE, 64, 2>) ink_mpp_moe_gu_dbuf<CODE, 64, 2>; \
    template [[host_name("ink_mpp_moe_dn_dbuf_" CNAME "_128")]] kernel decltype(ink_mpp_moe_dn_dbuf<CODE, 64, 2>) ink_mpp_moe_dn_dbuf<CODE, 64, 2>;
INK_MPP_DBUF_SET("sf4", 15)
INK_MPP_DBUF_SET("mx8", 16)
INK_MPP_DBUF_SET("gen", 0)

// ============================================================================
// 6. M5 Max Opt-In: Intra-Chunk SWA (chunk=8192) + SLM-Staged RelPos + FP8 KV FlashAttention-2 Prefill
// ============================================================================
struct InkMppAttnArgsV2 {
    int M;
    int M_pad;
    int slot;
    int pos0;
    int is_swa;
    int swa_win;
    int rel_extent;
    int kv_stride;
    float scale;
    int kv_fp8;
    int use_pf_kv;
    int pad;
};

kernel void ink_mpp_relpos_gqa_pf_v2(
    device half* q_h                [[buffer(0)]],
    device const half* r_h          [[buffer(1)]],
    device const half* rel_proj_t   [[buffer(2)]],
    device const float* sinks_buf   [[buffer(3)]],
    device const float* gate_buf    [[buffer(4)]],
    device const half* k_cache      [[buffer(5)]],
    device const half* v_cache      [[buffer(6)]],
    device float* attn_out          [[buffer(7)]],
    constant InkMppAttnArgsV2& A    [[buffer(8)]],
    device const half* pf_k_f16     [[buffer(9)]],
    device const half* pf_v_f16     [[buffer(10)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]) {
    threadgroup half sKV[64 * 128];   // 16 KB
    threadgroup half sP[32 * 64];     // 4 KB
    threadgroup half sR[32 * 16];     // 1 KB
    threadgroup half sRel[96 * 16];   // 3 KB (distances [d_base .. d_base + 95])
    threadgroup float sm[32], sl[32], sc[32];
    threadgroup float* sF = reinterpret_cast<threadgroup float*>(sKV);

    const int q0 = int(tgp.x) * 32;
    const int h = int(tgp.y);
    const int kh = h / 3;
    const int qpos0 = A.pos0 + q0;
    const int q_valid = min(32, A.M - q0);
    if (q_valid <= 0) return;

    if (tid < 32u) {
        sm[tid] = -INFINITY;
        sl[tid] = 0.0f;
        sc[tid] = 0.0f;
    }
    {
        const int r = int(tid >> 2), c4 = int((tid & 3u) * 4u);
        *((threadgroup half4*)(sR + r * 16 + c4)) =
            *((device const half4*)(r_h + ((size_t)h * A.M_pad + q0 + r) * 16 + c4));
    }

    device half* qbase = q_h + ((size_t)h * A.M_pad + q0) * 128;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ(qbase, dextents<int32_t, 2>(128, 32));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tKV(sKV, dextents<int32_t, 2>(128, 64));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(sP, dextents<int32_t, 2>(64, 32));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tS(sF, dextents<int32_t, 2>(64, 32));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(sF, dextents<int32_t, 2>(128, 32));

    constexpr auto dQK = matmul2d_descriptor(32, 64, 128, false, true, false);
    constexpr auto dPV = matmul2d_descriptor(32, 128, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dQK, execution_simdgroups<4>> opQK;
    matmul2d<dPV, execution_simdgroups<4>> opPV;

    auto accO = opPV.template get_destination_cooperative_tensor<decltype(tP), decltype(tKV), float>();
    for (uint16_t i = 0; i < accO.get_capacity(); ++i)
        if (accO.is_valid_element(i)) accO[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int k_min = A.is_swa ? max(0, qpos0 - A.swa_win + 1) : 0;
    const int k_max = qpos0 + q_valid;
    const size_t slot_kv_base = (size_t)A.slot * A.kv_stride;

    auto stage_kv = [&](device const half* cache_h, device const half* pf_h, int k0) {
        for (int p = 0; p < 8; ++p) {
            const uint idx = uint(p * 128) + tid;
            const uint kr = idx >> 4;
            const uint kc = (idx & 15u) << 3;
            const int kpos = k0 + int(kr);
            if (kpos < k_max) {
                if (A.use_pf_kv != 0 && kpos >= A.pos0) {
                    const size_t poff = ((size_t)(kpos - A.pos0) * 8 + kh) * 128 + kc;
                    *((threadgroup uint4*)(sKV + kr * 128u + kc)) = *((device const uint4*)(pf_h + poff));
                } else if (A.kv_fp8 == 0) {
                    const int kidx = A.is_swa ? (kpos & 1023) : kpos;
                    const size_t off = ((slot_kv_base + kidx) * 8 + kh) * 128 + kc;
                    *((threadgroup uint4*)(sKV + kr * 128u + kc)) = *((device const uint4*)(cache_h + off));
                } else {
                    const int kidx = A.is_swa ? (kpos & 1023) : kpos;
                    const size_t off = ((slot_kv_base + kidx) * 8 + kh) * 128 + kc;
                    device const uchar* cp = ((device const uchar*)cache_h) + off;
                    *((threadgroup half4*)(sKV + kr * 128u + kc + 0)) = half4(
                        mpp_e4m3_to_f32(cp[0]), mpp_e4m3_to_f32(cp[1]), mpp_e4m3_to_f32(cp[2]), mpp_e4m3_to_f32(cp[3]));
                    *((threadgroup half4*)(sKV + kr * 128u + kc + 4)) = half4(
                        mpp_e4m3_to_f32(cp[4]), mpp_e4m3_to_f32(cp[5]), mpp_e4m3_to_f32(cp[6]), mpp_e4m3_to_f32(cp[7]));
                }
            } else {
                *((threadgroup uint4*)(sKV + kr * 128u + kc)) = uint4(0u);
            }
        }
    };

    for (int k0 = k_min; k0 < k_max; k0 += 64) {
        // 1. Stage K[64, 128] into sKV AND stage RelPos window [d_base..d_base+95] into SLM sRel!
        stage_kv(k_cache, pf_k_f16, k0);
        const int d_base = max(0, qpos0 - (k0 + 63));
        for (uint idx = tid; idx < 96u * 4u; idx += 128u) {
            const int dr = int(idx >> 2);
            const int dc4 = int((idx & 3u) << 2);
            const int dist = min(d_base + dr, A.rel_extent);
            *((threadgroup half4*)(sRel + dr * 16 + dc4)) = *((device const half4*)(rel_proj_t + (size_t)dist * 16 + dc4));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. S[32, 64] = Q[32, 128] @ K[64, 128]^T
        auto accS = opQK.template get_destination_cooperative_tensor<decltype(tQ), decltype(tKV), float>();
        opQK.run(tQ, tKV, accS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        accS.store(tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. RelPos bias (100% from SLM sRel!) + Causal/SWA mask + Deferred-Sink Online Softmax -> sP
        {
            const int r = int(tid >> 2);
            const int lane4 = int(tid & 3u);
            const int qpos = qpos0 + r;
            const bool r_ok = (r < q_valid);
            const float4 rf0 = float4(*((threadgroup half4*)(sR + r * 16 + 0)));
            const float4 rf1 = float4(*((threadgroup half4*)(sR + r * 16 + 4)));
            const float4 rf2 = float4(*((threadgroup half4*)(sR + r * 16 + 8)));
            const float4 rf3 = float4(*((threadgroup half4*)(sR + r * 16 + 12)));

            float vals[16];
            float mx = -INFINITY;
            for (int e = 0; e < 16; ++e) {
                const int j = lane4 * 16 + e;
                const int kpos = k0 + j;
                float x = -INFINITY;
                if (r_ok && kpos <= qpos && (!A.is_swa || (qpos - kpos) < A.swa_win)) {
                    const int dist = min(qpos - kpos, A.rel_extent);
                    const int rel_i = clamp(dist - d_base, 0, 95);
                    threadgroup const half4* rp = (threadgroup const half4*)(sRel + rel_i * 16);
                    const float rb = dot(rf0, float4(rp[0])) + dot(rf1, float4(rp[1]))
                                   + dot(rf2, float4(rp[2])) + dot(rf3, float4(rp[3]));
                    x = sF[r * 64 + j] + rb;
                }
                vals[e] = x;
                mx = max(mx, x);
            }
            mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
            mx = max(mx, simd_shuffle_xor(mx, ushort(2)));

            const float m_old = sm[r];
            const float mn = max(m_old, mx);
            const float corr = (m_old == -INFINITY) ? 0.0f : exp(m_old - mn);
            float ps = 0.0f;
            for (int e = 0; e < 16; ++e) {
                const int j = lane4 * 16 + e;
                const float p = (vals[e] > -1e29f) ? exp(vals[e] - mn) : 0.0f;
                ps += p;
                sP[r * 64 + j] = half(p);
            }
            ps += simd_shuffle_xor(ps, ushort(1));
            ps += simd_shuffle_xor(ps, ushort(2));
            if (lane4 == 0) {
                sm[r] = mn;
                sl[r] = sl[r] * corr + ps;
                sc[r] = corr;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (k0 > k_min) {
            for (uint16_t i = 0; i < accO.get_capacity(); ++i) {
                if (accO.is_valid_element(i)) {
                    const int row = accO.get_multidimensional_index(i)[1];
                    accO[i] *= sc[row];
                }
            }
        }

        stage_kv(v_cache, pf_v_f16, k0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        opPV.run(tP, tKV, accO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (int(tid) < q_valid) {
        const float sk = sinks_buf[(size_t)(q0 + int(tid)) * 24 + h];
        const float m_k = sm[tid];
        if (m_k == -INFINITY) {
            sc[tid] = 0.0f;
        } else {
            const float m_fin = max(m_k, sk);
            const float so = exp(m_k - m_fin);
            sc[tid] = so / (sl[tid] * so + exp(sk - m_fin));
        }
    }
    accO.store(tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int r = 0; r < q_valid; ++r) {
        const float inv = sc[r];
        const size_t goff = ((size_t)(q0 + r) * 24 + h) * 128 + tid;
        const float g = gate_buf[goff];
        const float g_sig = 1.0f / (1.0f + exp(-g));
        attn_out[goff] = sF[r * 128 + tid] * inv * g_sig;
    }
}
