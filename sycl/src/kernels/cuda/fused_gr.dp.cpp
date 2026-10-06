// src/kernels/cuda/fused_gr.cu - see include/strata/kernels/fused_gr.hpp.
#define DPCT_PROFILING_ENABLED
#include <mutex>
#include <unordered_map>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/core/emulate.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "s26_tsum.dp.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace strata::kernels {
namespace {

constexpr int N = 2560;         // n_embd
constexpr int HC = 4;           // streams
constexpr int D = N * HC;       // 10240
constexpr int LR = 320;         // hc_lr
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
#if defined(__HIPCC__)
constexpr int TILE = 1280;             // eight-token tile fits gfx1100's 64 KiB LDS limit
#else
constexpr int TILE = 2560;             // xn floats per token staged at a time: 320 chunks of 8, 10 per lane
#endif
constexpr int TQ = TILE / 8 / 32;      // uint4 weight chunks per lane per tile
// SYCL port: the down kernel's 41 work-groups filled 15% of the B70's threads (metrics 2026-09-30); each row's
// D = 4 tiles are split over DOWN_SPLIT work-groups that write partial sums, summed in a fixed order by the up
// kernel (deterministic, unlike atomics) before the row's nonlinearity.
#ifndef STRATA_GR_DOWN_SPLIT
#define STRATA_GR_DOWN_SPLIT (D / TILE)   // 4: one tile per split (1: the unsplit order, for A/B)
#endif
constexpr int DOWN_SPLIT = STRATA_GR_DOWN_SPLIT;
constexpr int DOWN_BLOCKS = LR / WARPS;          // 40 blocks of 8 rows; one more for the inject rows
constexpr int UP_COLS = 32;                      // columns d per `up` block (x 4 streams = 128 rows)
constexpr int UP_BLOCKS = N / UP_COLS;           // 80

__dpct_inline__ float warp_sum(float v) {
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) v +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v,
            o);
    return v;
}
__dpct_inline__ float sigmoidf_(float x) {
    return 1.0f / (1.0f + sycl::native::exp(-x));
}

// 8 bf16 packed in a uint4, unpacked to two float4 once when reused across tokens.
struct Bf16x8 { sycl::float4 a, b; };
__dpct_inline__ Bf16x8 unpack8(const sycl::uint4 w) {
    return {sycl::float4(sycl::bit_cast<float>(w.x() << 16),
                         sycl::bit_cast<float>(w.x() & 0xffff0000u),
                         sycl::bit_cast<float>(w.y() << 16),
                         sycl::bit_cast<float>(w.y() & 0xffff0000u)),
            sycl::float4(sycl::bit_cast<float>(w.z() << 16),
                         sycl::bit_cast<float>(w.z() & 0xffff0000u),
                         sycl::bit_cast<float>(w.w() << 16),
                         sycl::bit_cast<float>(w.w() & 0xffff0000u))};
}
__dpct_inline__ float dot8u(const Bf16x8 &w, const sycl::float4 x0,
                            const sycl::float4 x1) {
    float acc = 0.0f;
    acc = sycl::fma((float)(w.a.x()), (float)(x0.x()), acc);
    acc = sycl::fma((float)(w.a.y()), (float)(x0.y()), acc);
    acc = sycl::fma((float)(w.a.z()), (float)(x0.z()), acc);
    acc = sycl::fma((float)(w.a.w()), (float)(x0.w()), acc);
    acc = sycl::fma((float)(w.b.x()), (float)(x1.x()), acc);
    acc = sycl::fma((float)(w.b.y()), (float)(x1.y()), acc);
    acc = sycl::fma((float)(w.b.z()), (float)(x1.z()), acc);
    acc = sycl::fma((float)(w.b.w()), (float)(x1.w()), acc);
    return acc;
}
__dpct_inline__ float dot8u_ptr(const Bf16x8 &w, const float *x) {
    const sycl::float4 *x4 = reinterpret_cast<const sycl::float4 *>(x);
    return dot8u(w, x4[0], x4[1]);
}

// 8 bf16 packed in a uint4 against 8 16-byte-aligned floats (same 8-FMA order).
__dpct_inline__ float dot8(const sycl::uint4 w, const float *x) {
    return dot8u_ptr(unpack8(w), x);
}

/*
DPCT1110: The total declared local variable size in device function
gr_down_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gr_down_kernel(FusedGrArgs a) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &xn = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[D]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &part =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARPS][HC]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_rs =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[HC]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    // 1. R' * w_norm into shared memory, and the per-stream sums of squares of R'.
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        sycl::float4 r = *reinterpret_cast<const sycl::float4 *>(a.R + i);
        if (a.apply) {
            const sycl::float4 b =
                *reinterpret_cast<const sycl::float4 *>(a.bo_prev + d);
            r.x() = sycl::fma((float)(b.x()), gw[c], r.x());
                r.y() = sycl::fma((float)(b.y()), gw[c], r.y());
            r.z() = sycl::fma((float)(b.z()), gw[c], r.z());
                r.w() = sycl::fma((float)(b.w()), gw[c], r.w());
        }
        const sycl::float4 g =
            *reinterpret_cast<const sycl::float4 *>(a.w_norm + i);
        float sq =
            r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<sycl::float4 *>(xn + i) = sycl::float4(
            r.x() * g.x(), r.y() * g.y(), r.z() * g.z(), r.w() * g.w());
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t < HC) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = sycl::rsqrt(s / (float)N + a.eps);
        if (item_ct1.get_group(2) == 0) a.rs[t] = s_rs[t];
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#pragma unroll
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // 2. one warp per output row: 10240 bf16 = 1280 chunks of 8, 40 per lane.
    const bool inject_block = item_ct1.get_group(2) == DOWN_BLOCKS;
    const int row = inject_block ? warp : item_ct1.get_group(2) * WARPS + warp;
    if (inject_block && (a.w_inject == nullptr || warp >= HC)) return;
    const uint16_t* wrow = (inject_block ? a.w_inject : a.w_down) + (size_t) row * D;
    const sycl::uint4 *w4 = reinterpret_cast<const sycl::uint4 *>(wrow);
    // SYCL port: the multi kernel's order - per-tile partial sums for the split rows (gr_parity: bit-identical),
    // the inject rows still one accumulation over the whole row (their block is not split there)
    float acc = 0.0f;
    if (inject_block) {
#pragma unroll 4
        for (int j = lane; j < D / 8; j += 32) acc += dot8(*(w4 + j), xn + j * 8);
        acc = warp_sum(acc);
    } else {
        for (int sp = 0; sp < DOWN_SPLIT; ++sp) {
            float part = 0.0f;
#pragma unroll 5
            for (int j = lane + sp * (TILE / 8); j < (sp + 1) * (TILE / 8); j += 32) part += dot8(*(w4 + j), xn + j * 8);
            acc += warp_sum(part);
        }
    }
    if (lane != 0) return;
    if (inject_block) {
        a.inject_out[row] = acc;
    } else {
        const float x = acc / (float) HC;
        a.lo[row] = x / (1.0f + sycl::native::exp(-x));
    }
}

__dpct_inline__ void gr_up_kernel(FusedGrArgs a) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &lo = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[LR]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &g = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[HC][UP_COLS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int d0 = item_ct1.get_group(2) * UP_COLS;
#pragma unroll
    for (int k = t; k < LR; k += THREADS) lo[k] = a.lo[k];
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // 128 rows (4 streams x 32 columns), 16 per warp: 320 bf16 = 40 chunks of 8.
    for (int r = warp; r < HC * UP_COLS; r += WARPS) {
        const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
        const sycl::uint4 *w4 =
            reinterpret_cast<const sycl::uint4 *>(a.w_up + (size_t)i * LR);
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        float acc = dot8(*(w4 + lane), lo + lane * 8);
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        if (lane < LR / 8 - 32) acc +=
            dot8(*(w4 + 32 + lane), lo + (32 + lane) * 8);
        acc = warp_sum(acc);
        if (lane == 0) {
            float rv = a.R[i];
            if (a.apply) {
                rv = sycl::fma((float)(a.bo_prev[d0 + dd]),
                               2.0f * sigmoidf_(a.inj_prev[c] / (float)HC), rv);
                a.R_out[i] = rv;                       // this block owns column d0+dd of every stream
            }
            const float x = rv * a.w_norm[i] * a.rs[c];
            g[c][dd] = x * sigmoidf_(acc);
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t < UP_COLS) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[c][t];
        a.mixed[d0 + t] = s / (float) HC;
    }
}

// ================================ plan v0.3 P6: T tokens, one weight read ================================
struct GrMulti {
    FusedGrArgs a[kFusedGrMaxT];
    float* xn;
    int T;
    float* part;   ///< [DOWN_SPLIT][kFusedGrMaxT][LR] partial row sums of the down kernel
    int nsplit = DOWN_SPLIT;   ///< SYCL port: splits the up kernel sums (1 after the sliced down kernel's reduce)
    float* part2 = nullptr;    ///< SYCL port: [GRS_SLICES][kFusedGrMaxT][LR + HC] slice partials of the sliced kernel
};


// S26 STRATA_QFUSE=1: the q8_1 image of `mixed`, written by the up kernels below. A q8_1 block is 32 columns and an
// up block owns UPM_COLS = 16, so the second of the two blocks that own a 32-column group (a per-group counter,
// incremented after the block's writes are fenced, reset by that block for the next launch) reads the 32 values back
// and quantizes them with native_quantize_q8_1_kernel's quantizer: one warp per token, the same XOR-tree max and sum,
// d = amax / 127, roundf(x / d), ds = (d, sum) - the bytes the separate launch writes.
struct GrQ81 { sycl::half2 ds; int8_t qs[32]; };
static_assert(2 * 16 == 32, "two up blocks per q8_1 group");
__dpct_inline__ void gr_q8_tail(const GrMulti &m, int d0) {
    /*
    DPCT1115: The sycl::ext::oneapi::group_local_memory_for_overwrite is
    used to allocate group-local memory at the none kernel functor scope of a
    work-group data parallel kernel. You may need to adjust the code.
    */
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &last = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0) {
        unsigned* c = m.a[0].q8_cnt + d0 / 32;
        last =
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                c, 1u) == 1u;
        if (last)
            dpct::atomic_exchange<sycl::access::address_space::generic_space>(
                c, 0u);
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (!last) return;
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    if (warp >= m.T) return;
    const int c0 = (d0 / 32) * 32;
    const float xi = *reinterpret_cast<const volatile float*>(m.a[warp].mixed + c0 + lane);
    float amax = sycl::fabs(xi), sum = xi;
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) amax = sycl::fmax(
        amax, dpct::experimental::permute_sub_group_by_xor(
                  0xffffffffu,
                  sycl::ext::oneapi::this_work_item::get_sub_group(), amax, o));
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) sum +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            sum, o);
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : sycl::round(xi / d);
    GrQ81* y = reinterpret_cast<GrQ81*>(m.a[warp].q8_mixed) + c0 / 32;
    y->qs[lane] = q;
    if (lane == 0) y->ds = sycl::half2(d, sum);
}
// Step 1 of `gr_down_kernel`, one block per token, same threads and reduction order: rs[t] and xn[t] to global.
/*
DPCT1110: The total declared local variable size in device function
gr_norm_multi_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gr_norm_multi_kernel(GrMulti m) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARPS][HC]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_rs =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[HC]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const FusedGrArgs &a = m.a[item_ct1.get_group(2)];
    float *xn = m.xn + (size_t)item_ct1.get_group(2) * D;
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        sycl::float4 r = *reinterpret_cast<const sycl::float4 *>(a.R + i);
        if (a.apply) {
            const sycl::float4 b =
                *reinterpret_cast<const sycl::float4 *>(a.bo_prev + d);
            r.x() = sycl::fma((float)(b.x()), gw[c], r.x());
                r.y() = sycl::fma((float)(b.y()), gw[c], r.y());
            r.z() = sycl::fma((float)(b.z()), gw[c], r.z());
                r.w() = sycl::fma((float)(b.w()), gw[c], r.w());
        }
        const sycl::float4 g =
            *reinterpret_cast<const sycl::float4 *>(a.w_norm + i);
        float sq =
            r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<sycl::float4 *>(xn + i) = sycl::float4(
            r.x() * g.x(), r.y() * g.y(), r.z() * g.z(), r.w() * g.w());
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t < HC) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = sycl::rsqrt(s / (float)N + a.eps);
        a.rs[t] = s_rs[t];
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#pragma unroll
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
}


// SYCL port: gr_norm_multi_kernel with one work-group per (token, stream) instead of per token: T groups left the
// B70 nearly idle (16 us a call). rs is per stream, so a group owns its stream outright. Thread t visits the same
// elements of stream c in the same order as in the per-token kernel (i = 4t + 1024k), and the warp and cross-warp
// sums run in the same order, so rs and xn are bitwise the old kernel's. The scaled values stay in registers
// (at most 3 float4 per thread) instead of a second pass over global memory.
__dpct_inline__ void gr_norm_split_port_kernel(GrMulti m) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto& part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARPS]>(item.get_group());
    auto& s_rs = *sycl::ext::oneapi::group_local_memory_for_overwrite<float>(item.get_group());
    const int tok = (int) item.get_group(2) / HC, c = (int) item.get_group(2) % HC;
    const FusedGrArgs& a = m.a[tok];
    float* xn = m.xn + (size_t) tok * D;
    const int t = (int) item.get_local_id(2), lane = t & 31, warp = t >> 5;
    const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    constexpr int MAXK = (N + THREADS * 4 - 1) / (THREADS * 4) + 1;   // 3: float4s of one stream per thread
    sycl::float4 keep[MAXK];
    int kept = 0;
    float ss = 0.0f;
    for (int i = t * 4; i < D; i += THREADS * 4) {
        if (i / N != c) continue;
        const int d = i - c * N;
        sycl::float4 r = *reinterpret_cast<const sycl::float4*>(a.R + i);
        if (a.apply) {
            const sycl::float4 b = *reinterpret_cast<const sycl::float4*>(a.bo_prev + d);
            r.x() = sycl::fma((float) (b.x()), gw, r.x());
            r.y() = sycl::fma((float) (b.y()), gw, r.y());
            r.z() = sycl::fma((float) (b.z()), gw, r.z());
            r.w() = sycl::fma((float) (b.w()), gw, r.w());
        }
        const sycl::float4 g = *reinterpret_cast<const sycl::float4*>(a.w_norm + i);
        ss += r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
        if (kept < MAXK) keep[kept++] = sycl::float4(r.x() * g.x(), r.y() * g.y(), r.z() * g.z(), r.w() * g.w());
    }
    const float v = warp_sum(ss);
    if (lane == 0) part[warp] = v;
    item.barrier(sycl::access::fence_space::local_space);
    if (t == 0) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) sum += part[w];
        s_rs = sycl::rsqrt(sum / (float) N + a.eps);
        a.rs[c] = s_rs;
    }
    item.barrier(sycl::access::fence_space::local_space);
    const float rs = s_rs;
    int k = 0;
    for (int i = t * 4; i < D && k < kept; i += THREADS * 4) {
        if (i / N != c) continue;
        const sycl::float4 x = keep[k++];
        *reinterpret_cast<sycl::float4*>(xn + i) = sycl::float4(x.x() * rs, x.y() * rs, x.z() * rs, x.w() * rs);
    }
}
bool gr_norm_split() {   // default; STRATA_GR_NORM_SPLIT=0: one work-group per token
    static const bool v = std::getenv("STRATA_GR_NORM_SPLIT") == nullptr || std::atoi(std::getenv("STRATA_GR_NORM_SPLIT")) != 0;
    return v;
}

// Step 2 of `gr_down_kernel` for T tokens.  One warp per row (so each lane accumulates the same chunks in the
// same order as the single-token kernel); per tile the lane's weight chunks are loaded BEFORE the activation
// tile is staged, so the DRAM and L2 traffic are in flight together.
//
// TILEV = xn floats per token staged at a time.  The tile only changes the staging granularity: the lane's chunk
// order (lane + 32*q within the tile, tiles ascending) is strictly increasing for either value, so the results are
// bitwise identical to `gr_down_kernel` for both.  2560 stages 320 chunks of 8 per tile (10 per lane); cards whose
// opt-in below 8 * 2560 * 4 B slices the tokens - sm_75 (64 KiB) carries 6 tokens of it - run TILEV 1280 instead,
// which fits all eight tokens in one 40 KiB launch and stages 160 chunks of 8 (5 per lane, half the registers held
// for the weight prefetch); smaller tiles raise how many blocks share an SM (the 41-block grid), e.g. three blocks
// of four tokens instead of one on sm_75.
template <int TILEV, int MAX_T = kFusedGrMaxT>
/*
DPCT1110: The total declared local variable size in device function
gr_down_multi_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gr_down_multi_kernel(GrMulti m, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int TQ = TILEV / 8 / 32; // uint4 weight chunks per lane per tile
    auto tile = (float *)dpct_local;   // [T][TILEV]
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const int gid = (int) item_ct1.get_group(2);
    const bool inject_block = gid == DOWN_BLOCKS * DOWN_SPLIT;
    const int split = inject_block ? 0 : gid % DOWN_SPLIT, rb = inject_block ? 0 : gid / DOWN_SPLIT;
    const int row = inject_block ? warp : rb * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const sycl::uint4 *w4 = reinterpret_cast<const sycl::uint4 *>(wrow);
    float acc[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
    for (int base = inject_block ? 0 : split * TILEV; base < D; base += inject_block ? TILEV : TILEV * DOWN_SPLIT) {
        sycl::uint4 wv[TQ];
        if (active) {
#pragma unroll
            /*
            DPCT1098: The '*' expression is used instead of the __ldg call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            for (int q = 0; q < TQ; ++q)
                wv[q] = *(w4 + base / 8 + lane + 32 * q);
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); // the previous tile is consumed
        const sycl::float4 *src4 = reinterpret_cast<const sycl::float4 *>(m.xn);
        sycl::float4 *tile4 = reinterpret_cast<sycl::float4 *>(tile);
        for (int i = t; i < T * (TILEV / 4); i += THREADS) {
            const int k = i / (TILEV / 4), off = i - k * (TILEV / 4);
            tile4[i] = src4[((size_t) k * D + base) / 4 + off];
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (!active) continue;
#pragma unroll
        for (int q = 0; q < TQ; ++q) {
            const int j = lane + 32 * q;
#pragma unroll
            for (int k = 0; k < MAX_T; ++k)
                if (k < T) acc[k] += dot8(wv[q], tile + k * TILEV + j * 8);
        }
    }
    if (!active) return;
    float s[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) s[k] = k < T ? warp_sum(acc[k]) : 0.0f;
    // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if (k >= T || lane != k) continue;
        if (inject_block) {
            m.a[k].inject_out[row] = s[k];
        } else {
            m.part[((size_t) split * kFusedGrMaxT + k) * LR + row] = s[k];   // the up kernel sums the splits
        }
    }
}

// SYCL port: gr_down_multi_kernel without the shared-memory tile. The tiled kernel stages T x 2560 floats per
// work-group (60 KB at T = 6) for 8 rows (40 KB of weights): the copy outweighs the weights and the SLM caps the
// B70 at ~2 groups per Xe core (unitrace: 64 us per call, ~100 GB/s). Here each lane reads its 8 xn floats per
// token straight from global memory (T x 40 KB in all, L1/L2-resident, shared by every group). Same per-lane
// order of the sums, so the outputs are bitwise the tiled kernel's.
__dpct_inline__ void gr_down_multi_direct_kernel(GrMulti m) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const int gid = (int) item_ct1.get_group(2);
    const bool inject_block = gid == DOWN_BLOCKS * DOWN_SPLIT;
    const int split = inject_block ? 0 : gid % DOWN_SPLIT, rb = inject_block ? 0 : gid / DOWN_SPLIT;
    const int row = inject_block ? warp : rb * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    if (!active) return;
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) row * D;
    const sycl::uint4* w4 = reinterpret_cast<const sycl::uint4*>(wrow);
    float acc[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) acc[k] = 0.0f;
    for (int base = inject_block ? 0 : split * TILE; base < D; base += inject_block ? TILE : TILE * DOWN_SPLIT) {
        sycl::uint4 wv[TQ];
#pragma unroll
        for (int q = 0; q < TQ; ++q) wv[q] = *(w4 + base / 8 + lane + 32 * q);
#pragma unroll
        for (int q = 0; q < TQ; ++q) {
            const int j = lane + 32 * q;
#pragma unroll
            for (int k = 0; k < kFusedGrMaxT; ++k)
                if (k < T) {
                    const sycl::float4* x4 = reinterpret_cast<const sycl::float4*>(m.xn + (size_t) k * D + base + j * 8);
                    const sycl::float4 a0 = x4[0], a1 = x4[1];
                    const float xv[8] = {a0.x(), a0.y(), a0.z(), a0.w(), a1.x(), a1.y(), a1.z(), a1.w()};
                    acc[k] += dot8(wv[q], xv);
                }
        }
    }
    float s[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) s[k] = k < T ? warp_sum(acc[k]) : 0.0f;
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) {
        if (k >= T || lane != k) continue;
        if (inject_block) m.a[k].inject_out[row] = s[k];
        else m.part[((size_t) split * kFusedGrMaxT + k) * LR + row] = s[k];
    }
}
// SYCL port: the down projection sliced by columns. The direct kernel above has each warp read T x 32 bytes of
// activations (L2) per 16 bytes of weights - 12x the weight traffic at 6 tokens, ~134 GB/s. Here work-group s owns
// columns [128 s, 128 s + 128) of all 324 rows (320 down + 4 inject): it stages that slice of xn for the T tokens
// once (<= 4 KB of SLM) and streams the rows through it, 4 lanes per row (one 64-byte line per step), reducing over
// those 4 lanes once per row. gr_down_reduce then sums the 80 slice partials in a fixed order.
constexpr int GRS_COLS = 128, GRS_SLICES = D / GRS_COLS, GRS_ROWS = LR + HC;   // 80 slices, 324 rows
__dpct_inline__ void gr_down_sliced_kernel(GrMulti m, float* tile) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = (int) item.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = m.T, sl = (int) item.get_group(2), c0 = sl * GRS_COLS;
    for (int i = t; i < T * GRS_COLS; i += THREADS) {
        const int k = i / GRS_COLS, c = i - k * GRS_COLS;
        tile[i] = m.xn[(size_t) k * D + c0 + c];
    }
    item.barrier(sycl::access::fence_space::local_space);
    const int rsub = lane >> 2, q = lane & 3;
    const bool inj = m.a[0].w_inject != nullptr;
    for (int r0 = 0; r0 < GRS_ROWS; r0 += WARPS * 8) {
        const int r = r0 + warp * 8 + rsub;
        const bool ok = r < LR || (r < GRS_ROWS && inj);
        const uint16_t* wrow = r < LR ? m.a[0].w_down + (size_t) r * D : m.a[0].w_inject + (size_t) (ok ? r - LR : 0) * D;
        sycl::uint4 wv[4];
#pragma unroll
        for (int cc = 0; cc < 4; ++cc)
            wv[cc] = ok ? reinterpret_cast<const sycl::uint4*>(wrow + c0)[q + 4 * cc] : sycl::uint4(0, 0, 0, 0);
        float acc[kFusedGrMaxT];
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            acc[k] = 0.0f;
            if (k < T) {
#pragma unroll
                for (int cc = 0; cc < 4; ++cc) acc[k] += dot8(wv[cc], tile + k * GRS_COLS + (q + 4 * cc) * 8);
            }
        }
        auto sg = item.get_sub_group();
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            float v = acc[k];
            v += sycl::permute_group_by_xor(sg, v, 1);
            v += sycl::permute_group_by_xor(sg, v, 2);
            if (q == 0 && ok) m.part2[((size_t) sl * kFusedGrMaxT + k) * GRS_ROWS + r] = v;
        }
    }
}
__dpct_inline__ void gr_down_reduce_kernel(GrMulti m) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = (int) item.get_global_id(2);
    if (i >= m.T * GRS_ROWS) return;
    const int k = i / GRS_ROWS, r = i - k * GRS_ROWS;
    if (r >= LR && m.a[0].w_inject == nullptr) return;
    float x = 0.0f;
    for (int sl = 0; sl < GRS_SLICES; ++sl) x += m.part2[((size_t) sl * kFusedGrMaxT + k) * GRS_ROWS + r];
    if (r < LR) m.part[(size_t) k * LR + r] = x;   // split 0
    else m.a[k].inject_out[r - LR] = x;
}
bool gr_down_sliced() {   // default (gr_bench: GR read 108.6 -> 76.5 us at 6 tokens); STRATA_GR_DOWN_SLICED=0: direct
    static const bool v = std::getenv("STRATA_GR_DOWN_SLICED") == nullptr || std::atoi(std::getenv("STRATA_GR_DOWN_SLICED")) != 0;
    return v;
}
float* slice_partials(sycl::queue* q) {
    static std::mutex mu;
    static std::unordered_map<sycl::queue*, float*> bufs;
    std::lock_guard<std::mutex> lk(mu);
    auto it = bufs.find(q);
    if (it != bufs.end()) return it->second;
    float* p = sycl::malloc_device<float>((size_t) GRS_SLICES * kFusedGrMaxT * GRS_ROWS, *q);
    bufs[q] = p;
    return p;
}
bool gr_down_direct() {   // default; STRATA_GR_DOWN_DIRECT=0: the tiled kernel (gr_bench: 13-19% slower at 3-6 tokens)
    static const bool v = std::getenv("STRATA_GR_DOWN_DIRECT") == nullptr || std::atoi(std::getenv("STRATA_GR_DOWN_DIRECT")) != 0;
    return v;
}

constexpr int UPM_COLS = 16;                      // columns per block (x 4 streams = 64 rows, 8 per warp)
constexpr int UPM_BLOCKS = N / UPM_COLS;          // 160

// `gr_up_kernel` for T tokens: each row of w_up read once; the T dots reduced by xor so every lane holds every
// sum, and lane k runs token k's epilogue - the T epilogues in parallel instead of one after another.
template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
/*
DPCT1110: The total declared local variable size in device function
gr_up_multi_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gr_up_multi_kernel(GrMulti m) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &lo =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<float[MAX_T][LR]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &g = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[MAX_T][HC][UPM_COLS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const int d0 = item_ct1.get_group(2) * UPM_COLS;
#pragma unroll
    for (int i = t; i < T * LR; i += THREADS) {   // the down kernel's partial sums, in split order, then silu
        const int k = i / LR, r = i % LR;
        float x = 0.0f;
#pragma unroll
        for (int sp = 0; sp < DOWN_SPLIT; ++sp) if (sp < m.nsplit) x += m.part[((size_t) sp * kFusedGrMaxT + k) * LR + r];
        x /= (float) HC;
        const float v = x / (1.0f + sycl::native::exp(-x));
        lo[k][r] = v;
        m.a[k].lo[r] = v;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int r = warp; r < HC * UPM_COLS; r += WARPS) {
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const sycl::uint4 *w4 =
            reinterpret_cast<const sycl::uint4 *>(m.a[0].w_up + (size_t)i * LR);
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const Bf16x8 wa = unpack8(*(w4 + lane));
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const Bf16x8 wb = unpack8(lane < LR / 8 - 32 ? *(w4 + 32 + lane)
                                                     : sycl::uint4(0, 0, 0, 0));
        // the epilogue inputs of this lane's token, fetched while the dots run
        float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            rsc = a.rs[c];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if (!EXACT_T && k >= T) break;
            float acc = dot8u_ptr(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8u_ptr(wb, lo[k] + (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = sycl::fma(bo, 2.0f * sigmoidf_(ip / (float)HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsc;
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE
}


// ================================ hc read v3 (opt-in: STRATA_GR_V3=1) - two kernels, stream-split ================
// The norm kernel runs on only T blocks (~16 us of pure latency per call) and `down` on 41 blocks (~280 GB/s).
// v3: `down` is split over (row group, stream[, column half]) = 164 or 328 blocks; each block stages its slice of
// R' * w_norm for the T tokens, reduces that slice's sum of squares itself, and writes UNSCALED partial dots.  The
// rms scale is per stream, so  w_down . xn = sum_c rs[c] * (w_down[:, c] . (R'[c] * w_norm[c]))  - `up` applies it
// in its prologue (lo, inject, rs).  Same maths, ANOTHER SUMMATION ORDER: not bitwise the default kernels, hence
// opt-in.  Dynamic shared memory is T * (N / S) floats: S (1 or 2 column halves) is the smallest that fits the
// card's opt-in limit at kFusedGrMaxT tokens (Ampere 99 KB: S = 1; Turing / HIP 64 KB: S = 2); a card where
// neither fits keeps the default kernels.
constexpr int PR = LR + HC;                        // partial rows per (token, stream): 320 down + 4 inject
constexpr int TQ3 = N / 8 / 32;                    // uint4 weight chunks per lane in one stream's slice (10)

template <int S, int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
/*
DPCT1110: The total declared local variable size in device function
gr_down_v3_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gr_down_v3_kernel(GrMulti m, float *__restrict__ part,
                                       float *__restrict__ ssg,
                                       uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto xs = (float *)dpct_local;                 // [T][N / S]
    constexpr int R2 = 1;                          // down rows per warp
    auto &red = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[WARPS][MAX_T]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    // S = 2: each stream's 2560 columns in two halves (blockIdx.y = stream * S + half): twice the blocks
    constexpr int SL = N / S, TQS = SL / 8 / 32;
    const int rg = item_ct1.get_group(2), c = item_ct1.get_group(1) / S,
              h = item_ct1.get_group(1) - (item_ct1.get_group(1) / S) * S;
    constexpr int NDB = LR / (WARPS * R2);          // down row blocks per stream; block NDB = the inject rows
    const bool inject_block = rg == NDB;
    // warp w owns rows row0 + w * R2 + r (r < R2); the inject block: warps 0-3, one row each
    const int row0 = inject_block ? warp : (rg * WARPS + warp) * R2;
    const int nrows = inject_block ? ((m.a[0].w_inject != nullptr && warp < HC) ? 1 : 0) : R2;
    const bool active = nrows > 0;
    const uint16_t* wbase = inject_block ? m.a[0].w_inject : m.a[0].w_down;
    float ssp[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        ssp[k] = 0.0f;
        if (!EXACT_T && k >= T) continue;
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
        const sycl::float4 *R4 = reinterpret_cast<const sycl::float4 *>(
            a.R + (size_t)c * N + (size_t)h * SL);
        const sycl::float4 *G4 = reinterpret_cast<const sycl::float4 *>(
            a.w_norm + (size_t)c * N + (size_t)h * SL);
        const sycl::float4 *B4 =
            reinterpret_cast<const sycl::float4 *>(a.bo_prev + (size_t)h * SL);
        sycl::float4 *X4 =
            reinterpret_cast<sycl::float4 *>(xs + (size_t)k * SL);
        for (int i = t; i < SL / 4; i += THREADS) {
            sycl::float4 r = R4[i];
            if (a.apply) {
                const sycl::float4 b = B4[i];
                r.x() = sycl::fma((float)(b.x()), (float)gw, r.x());
                    r.y() = sycl::fma((float)(b.y()), (float)gw, r.y());
                r.z() = sycl::fma((float)(b.z()), (float)gw, r.z());
                    r.w() = sycl::fma((float)(b.w()), (float)gw, r.w());
            }
            const sycl::float4 g = G4[i];
            ssp[k] +=
                r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
            X4[i] = sycl::float4(r.x() * g.x(), r.y() * g.y(), r.z() * g.z(),
                                 r.w() * g.w());
        }
    }
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if (!EXACT_T && k >= T) break;
        const float v = warp_sum(ssp[k]);
        if (lane == 0) red[warp][k] = v;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (rg == 0 && t < T) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) sum += red[w][t];
        ssg[(t * HC + c) * S + h] = sum;
    }
    if (!active) return;
#pragma unroll
    for (int r = 0; r < R2; ++r) {
        if (r >= nrows) break;
        const sycl::uint4 *w4 = reinterpret_cast<const sycl::uint4 *>(
            wbase + (size_t)(row0 + r) * D + (size_t)c * N + (size_t)h * SL);
        float acc[MAX_T];
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int q = 0; q < TQS; ++q) {
            const int j = lane + 32 * q;
            /*
            DPCT1098: The '*' expression is used instead of the __ldg call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            const Bf16x8 wvq = unpack8(*(w4 + j));
#pragma unroll
            for (int k = 0; k < MAX_T; ++k)
                if (EXACT_T || k < T) acc[k] += dot8u_ptr(wvq, xs + (size_t) k * SL + j * 8);
        }
        const int prow = inject_block ? LR + warp : row0 + r;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if (!EXACT_T && k >= T) break;
            const float v = warp_sum(acc[k]);
            if (lane == 0) part[(((size_t) k * HC + c) * S + h) * PR + prow] = v;
        }
    }
}

template <int S, int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
/*
DPCT1110: The total declared local variable size in device function
gr_up_v3_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gr_up_v3_kernel(GrMulti m, const float *__restrict__ part,
                                     const float *__restrict__ ssg) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &lo =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<float[MAX_T][LR]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &rsS =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[MAX_T][HC]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &g = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[MAX_T][HC][UPM_COLS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const int d0 = item_ct1.get_group(2) * UPM_COLS;
    constexpr int RPW = HC * UPM_COLS / WARPS;     // 8 rows per warp
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
#pragma unroll
        for (int h = 0; h < S; ++h) ss += ssg[t * S + h];
        const float r = sycl::rsqrt(ss / (float)N + m.a[k].eps);
        rsS[k][c] = r;
        if (item_ct1.get_group(2) == 0) m.a[k].rs[c] = r;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = t; i < T * LR; i += THREADS) {
        const int k = i / LR, r = i - k * LR;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
#pragma unroll
            for (int h = 0; h < S; ++h) p += part[(((size_t) k * HC + c) * S + h) * PR + r];
            sum = sycl::fma(rsS[k][c], p, sum);
        }
        const float x = sum / (float) HC;
        lo[k][r] = x / (1.0f + sycl::native::exp(-x));
    }
    if (item_ct1.get_group(2) == 0 && t < T * HC) {
        const int k = t / HC, cc = t - k * HC;
        if (m.a[k].w_inject != nullptr) {
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                float p = 0.0f;
#pragma unroll
                for (int h = 0; h < S; ++h) p += part[(((size_t) k * HC + c) * S + h) * PR + LR + cc];
                sum = sycl::fma(rsS[k][c], p, sum);
            }
            m.a[k].inject_out[cc] = sum;
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#pragma unroll
    for (int q = 0; q < RPW; ++q) {
        const int r = warp + q * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const sycl::uint4 *w4 =
            reinterpret_cast<const sycl::uint4 *>(m.a[0].w_up + (size_t)i * LR);
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const Bf16x8 wa = unpack8(*(w4 + lane));
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const Bf16x8 wb = unpack8(lane < LR / 8 - 32 ? *(w4 + 32 + lane)
                                                     : sycl::uint4(0, 0, 0, 0));
        float rv = 0.0f, wn = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if (!EXACT_T && k >= T) break;
            float acc = dot8u_ptr(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8u_ptr(wb, lo[k] + (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = sycl::fma(bo, 2.0f * sigmoidf_(ip / (float)HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsS[lane][c];
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
}
// ================================ #315: split / staged - the same arithmetic, split differently ==================
// The multi read's variants beside the plain read (0.1.31's default; not main's opt-in STRATA_GR_V3 read, which sums
// in another order).  Every variant computes each output with exactly the plain read's operations in its order (so
// bitwise the plain read and the single-token read); only who does what, and when, changes.  `fused_gr_check`
// compares them on the card before a verify window uses them (STRATA_HC_SPLIT below).
//  - the norm (split and staged): one block per token AND stream instead of one per token.  Each thread visits
//    exactly the elements, in the order, it visited for that stream in the plain read, so every sum of squares and rs
//    is the same; then it recomputes its R' * w_norm and writes it times rs (the plain read stored the product and
//    scaled it in place: the same two roundings).  On an RTX 4080 SUPER, the plain read's norm took 0.58 ms per
//    4-token round (96 launches of 1-4 blocks), this one 0.23 ms.
//  - the down projection, split: the plain read's kernel.
//  - the down projection, staged: the plain read's 8 rows per block and lane order, but the activations arrive by
//    cp.async in half-stream tiles, two in flight, so no thread waits on a chain of loads, and they sit in shared
//    memory as two planes (floats 0-3 and 4-7 of every 8), so a warp's reads hit no bank twice.  The next tile's
//    weights are loaded while the current one is used.  A tile is 160 = 5 x 32 chunks of 8, so a lane still
//    accumulates its chunks lane + 32 q in ascending order, as in the plain read with either tile.  Before sm_80
//    (and on HIP) the staging is a plain copy: the same bits, only not asynchronous.
constexpr int kHcPlain = 1, kHcSplit = 2, kHcStaged = 3;
constexpr int H_TILE = 1280;                          // staged tile: half a stream = 160 chunks of 8, 5 per lane
constexpr int HQ = H_TILE / 8 / 32;
constexpr int N_HTILES = D / H_TILE;                  // 8
static_assert(N % H_TILE == 0, "a staged tile never straddles two streams");
static_assert(H_TILE % 256 == 0, "a staged tile holds whole rounds of 32 chunks: the plain read's lane order");

/*
DPCT1110: The total declared local variable size in device function
gr_norm_split_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gr_norm_split_kernel(GrMulti m) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARPS]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_rs = *sycl::ext::oneapi::group_local_memory_for_overwrite<float>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const FusedGrArgs &a = m.a[item_ct1.get_group(2)];
    const int c = item_ct1.get_group(1);
    float *xn = m.xn + (size_t)item_ct1.get_group(2) * D;
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss = 0.0f;
    for (int i = t * 4; i < D; i += THREADS * 4) {
        if (i / N != c) continue;                       // another stream's element: another block's
        const int d = i - c * N;
        sycl::float4 r = *reinterpret_cast<const sycl::float4 *>(a.R + i);
        if (a.apply) {
            const sycl::float4 b =
                *reinterpret_cast<const sycl::float4 *>(a.bo_prev + d);
            r.x() = sycl::fma((float)(b.x()), (float)gw, r.x());
                r.y() = sycl::fma((float)(b.y()), (float)gw, r.y());
            r.z() = sycl::fma((float)(b.z()), (float)gw, r.z());
                r.w() = sycl::fma((float)(b.w()), (float)gw, r.w());
        }
        const float sq =
            r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
        ss += sq;
    }
    const float v = warp_sum(ss);
    if (lane == 0) part[warp] = v;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t == 0) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) s += part[w];
        s_rs = sycl::rsqrt(s / (float)N + a.eps);
        a.rs[c] = s_rs;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const float rs = s_rs;
    for (int i = t * 4; i < D; i += THREADS * 4) {
        if (i / N != c) continue;
        const int d = i - c * N;
        sycl::float4 r = *reinterpret_cast<const sycl::float4 *>(a.R + i);
        if (a.apply) {
            const sycl::float4 b =
                *reinterpret_cast<const sycl::float4 *>(a.bo_prev + d);
            r.x() = sycl::fma((float)(b.x()), (float)gw, r.x());
                r.y() = sycl::fma((float)(b.y()), (float)gw, r.y());
            r.z() = sycl::fma((float)(b.z()), (float)gw, r.z());
                r.w() = sycl::fma((float)(b.w()), (float)gw, r.w());
        }
        const sycl::float4 g =
            *reinterpret_cast<const sycl::float4 *>(a.w_norm + i);
        const float px = r.x() * g.x(), py = r.y() * g.y(), pz = r.z() * g.z(),
                    pw = r.w() * g.w();
        *reinterpret_cast<sycl::float4 *>(xn + i) =
            sycl::float4(px * rs, py * rs, pz * rs, pw * rs);
    }
}

#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 800 &&      \
    !defined(__HIPCC__)
#define STRATA_GR_CP_ASYNC 1
#endif
__dpct_inline__ void cp_async16(void *smem, const void *gmem) {
#if defined(STRATA_GR_CP_ASYNC)
    auto sa = smem;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa),
                 "l"(gmem)
                 : "memory");
#else
    *(((uint32_t *)(uintptr_t)sa)) = *(((uint32_t *)(uintptr_t)gmem));
    if (16 > 4)
        *(((uint32_t *)(uintptr_t)sa) + 1) =
            *(((uint32_t *)(uintptr_t)gmem) + 1);
    if (16 > 8)
        *(((uint32_t *)(uintptr_t)sa) + 2) =
            *(((uint32_t *)(uintptr_t)gmem) + 2);
    if (16 > 12)
        *(((uint32_t *)(uintptr_t)sa) + 3) =
            *(((uint32_t *)(uintptr_t)gmem) + 3);
#endif
#else
    *reinterpret_cast<float4*>(smem) = *reinterpret_cast<const float4*>(gmem);
#endif
}
__dpct_inline__ void cp_async_commit() {
#if defined(STRATA_GR_CP_ASYNC)
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#else

#endif
#endif
}
__dpct_inline__ void cp_async_wait1() {
#if defined(STRATA_GR_CP_ASYNC)
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
#else

#endif
#endif
}
__dpct_inline__ void cp_async_wait0() {
#if defined(STRATA_GR_CP_ASYNC)
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#else

#endif
#endif
}

// `dot8` with its 8 activations as two float4: the same eight fmaf in the same order
__dpct_inline__ float dot8v(const sycl::uint4 w, const sycl::float4 x0,
                            const sycl::float4 x1) {
    float acc = 0.0f;
    acc = sycl::fma(sycl::bit_cast<float>(w.x() << 16), (float)(x0.x()), acc);
    acc = sycl::fma(sycl::bit_cast<float>(w.x() & 0xffff0000u), (float)(x0.y()),
                    acc);
    acc = sycl::fma(sycl::bit_cast<float>(w.y() << 16), (float)(x0.z()), acc);
    acc = sycl::fma(sycl::bit_cast<float>(w.y() & 0xffff0000u), (float)(x0.w()),
                    acc);
    acc = sycl::fma(sycl::bit_cast<float>(w.z() << 16), (float)(x1.x()), acc);
    acc = sycl::fma(sycl::bit_cast<float>(w.z() & 0xffff0000u), (float)(x1.y()),
                    acc);
    acc = sycl::fma(sycl::bit_cast<float>(w.w() << 16), (float)(x1.z()), acc);
    acc = sycl::fma(sycl::bit_cast<float>(w.w() & 0xffff0000u), (float)(x1.w()),
                    acc);
    return acc;
}

// Stage tile `h` of every token into `buf`: [T][2 planes][160 chunks] float4, plane 0 = floats 0-3 of a chunk.
__dpct_inline__ void stage_htile(const GrMulti &m, int T, int h,
                                 sycl::float4 *buf, int t) {
    for (int i = t; i < T * (H_TILE / 4); i += THREADS) {
        const int k = i / (H_TILE / 4), s4 = i - k * (H_TILE / 4);   // s4: float4 of the tile, chunk s4/2, half s4&1
        const float* src = m.xn + (size_t) k * D + (size_t) h * H_TILE + (size_t) s4 * 4;
        cp_async16(buf + (size_t) k * (H_TILE / 4) + (s4 & 1) * (H_TILE / 8) + (s4 >> 1), src);
    }
}

template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
/*
DPCT1110: The total declared local variable size in device function
gr_down_staged_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gr_down_staged_kernel(GrMulti m, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto hbuf = (sycl::float4 *)dpct_local; // 2 buffers x [T][2][160] float4
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const bool inject_block = item_ct1.get_group(2) == DOWN_BLOCKS;
    const int row = inject_block ? warp : item_ct1.get_group(2) * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const sycl::uint4 *w4 = reinterpret_cast<const sycl::uint4 *>(wrow);
    const size_t buf_f4 = (size_t) T * (H_TILE / 4);    // float4 per buffer (the second one follows the first)
    float acc[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
    sycl::uint4 wv[HQ], wnext[HQ];
    if (active) {
#pragma unroll
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        for (int q = 0; q < HQ; ++q) wv[q] = *(w4 + lane + 32 * q);
    }
    stage_htile(m, T, 0, hbuf, t);
    cp_async_commit();
    stage_htile(m, T, 1, hbuf + buf_f4, t);
    cp_async_commit();
#pragma unroll 1
    for (int h = 0; h < N_HTILES; ++h) {
        if (active && h + 1 < N_HTILES) {
#pragma unroll
            /*
            DPCT1098: The '*' expression is used instead of the __ldg call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            for (int q = 0; q < HQ; ++q)
                wnext[q] = *(w4 + (h + 1) * (H_TILE / 8) + lane + 32 * q);
        }
        if (h + 1 < N_HTILES) cp_async_wait1();         // tile h has landed (h + 1 may still be on its way)
        else cp_async_wait0();
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const sycl::float4 *cur = hbuf + (h & 1) * buf_f4;
        if (active) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) {
                const int j = lane + 32 * q;
                const Bf16x8 wvq = unpack8(wv[q]);
#pragma unroll
                for (int k = 0; k < MAX_T; ++k) {
                    if (EXACT_T || k < T) {
                        const sycl::float4 *pk = cur + (size_t)k * (H_TILE / 4);
                        acc[k] += dot8u(wvq, pk[j], pk[H_TILE / 8 + j]);
                    }
                }
            }
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); // every warp is done with buffer h & 1
        if (h + 2 < N_HTILES) stage_htile(m, T, h + 2, hbuf + (h & 1) * buf_f4, t);
        cp_async_commit();                              // an empty group at the end keeps the wait counts simple
        if (h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wv[q] = wnext[q];
        }
    }
    if (!active) return;
    float s[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) s[k] = (EXACT_T || k < T) ? warp_sum(acc[k]) : 0.0f;
    // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if ((!EXACT_T && k >= T) || lane != k) continue;
        if (inject_block) {
            m.a[k].inject_out[row] = s[k];
        } else {
            const float x = s[k] / (float) HC;
            m.a[k].lo[row] = x / (1.0f + sycl::native::exp(-x));
        }
    }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
void gr_down_staged_kernel_wrapper(GrMulti m) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range(localMemSize), cgh);

        cgh.parallel_for<dpct_kernel_name<class gr_down_staged_kernel_wrap, dpct_kernel_scalar<MAX_T>,
                                          dpct_kernel_scalar<EXACT_T>>>(
            nr, exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gr_down_staged_kernel<MAX_T, EXACT_T>(
                    m, dpct_local_acc_ct1
                           .get_multi_ptr<sycl::access::decorated::no>()
                           .get());
            });
    });
}

// The tokens a down kernel may carry in one launch on the current card, and the plain read's tile: the plain read
// stages n_tok * TILEV floats (1280 on sm_75, 2560 elsewhere), staged two tiles of n_tok * H_TILE.  The shared-memory
// opt-in is set here once per device (a per-DEVICE setting: a layer split runs these kernels on two cards).
int down_chunk(bool staged, int *tile_out) try {
    static bool attr[64] = {};
    static int chunk[64] = {};   // tokens the plain down kernel may carry in one launch on this card
    static int chunk_staged[64] = {};  // the same for staged
    static int tile[64] = {};    // the plain down kernel's TILEV on this card (1280 on sm_75, 2560 elsewhere)
    int dev = 0;
    dev = dpct::get_current_device_id();
    if (dev < 0 || dev >= 64) {
        *tile_out = 2560;
        return kFusedGrMaxT;
    }
    if (!attr[dev]) {
        int optin = 0;
        /*
        DPCT1019: local_mem_size in SYCL is not a complete equivalent of
        cudaDevAttrMaxSharedMemoryPerBlockOptin in CUDA. You may need to adjust
        the code.
        */
        optin = dpct::get_device(dev).get_local_mem_size();
        optin = strata::smem_optin_of(optin);   // STRATA_EMULATE_CC (tests only)
        // the down kernel stages n_tok*TILEV floats of dynamic shared memory - 80 KB at the full 8 tokens of the
        // CUDA tile.  sm_75 gets the smaller tile: all eight tokens fit one 40 KiB launch there (no more slicing),
        // the TQ-5 prefetch holds half the registers, and the smaller blocks raise how many of the 41-block grid
        // share an SM.  Cards whose opt-in is still below that (or that report no opt-in at all) slice the tokens;
        // the down kernel's outputs (lo, inject_out) are strictly per-token, so the chunk boundaries are safe, and
        // the up kernel below still sees every token of the batch in one launch.
#if defined(__HIPCC__)
#if defined(STRATA_HIP_GFX906)
        const bool small_tile = false;  // gfx906: 64 KiB LDS, the 2560 tile as before (8 tokens slice into launches)
#else
        const bool small_tile = true;   // all eight tokens fit gfx1100's 64 KiB LDS at this tile
#endif
#else
        int cc_maj = 0, cc_min = 0;
        cc_maj = dpct::get_device(dev).get_major_version();
        cc_min = dpct::get_device(dev).get_minor_version();
        const bool small_tile = strata::cc_major_of(cc_maj) * 10 + strata::cc_minor_of(cc_min) == 75;
#endif
        const int tv = small_tile ? 1280 : 2560;
        tile[dev] = tv;
        int want = (int) (kFusedGrMaxT * tv * sizeof(float));
        if (optin > 0 && want > optin) want = optin;
        if (small_tile) {
            /*
            DPCT1026: The call to cudaFuncSetAttribute was removed because
            SYCL currently does not support corresponding setting.
            */
        } else {
            /*
            DPCT1026: The call to cudaFuncSetAttribute was removed because
            SYCL currently does not support corresponding setting.
            */
        }
        auto set_staged_attr = [&](auto kernel, int max_t) {
            int ws = (int) (2 * max_t * H_TILE * sizeof(float));
            if (optin > 0 && ws > optin) ws = optin;
            if (ws > 48 * 1024)
                /*
                DPCT1026: The call to cudaFuncSetAttribute was removed
                because SYCL currently does not support corresponding setting.
                */
                ;
            /*
            DPCT1026: The call to cudaFuncSetAttribute was removed because
            SYCL currently does not support corresponding setting.
            */
        };
        set_staged_attr(gr_down_staged_kernel_wrapper<1, true>, 1);
        set_staged_attr(gr_down_staged_kernel_wrapper<2, true>, 2);
        set_staged_attr(gr_down_staged_kernel_wrapper<3, true>, 3);
        set_staged_attr(gr_down_staged_kernel_wrapper<4, true>, 4);
        set_staged_attr(gr_down_staged_kernel_wrapper<5, true>, 5);
        set_staged_attr(gr_down_staged_kernel_wrapper<6, true>, 6);
        set_staged_attr(gr_down_staged_kernel_wrapper<4, false>, 4);
        set_staged_attr(gr_down_staged_kernel_wrapper<kFusedGrMaxT, false>,
                        kFusedGrMaxT);
        /*
        DPCT1026: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
; // drop any error the attempt left behind
  // The opt-in is a promise a pre-Volta card does not keep: an sm_60 answers
  // 65536 and accepts the cudaFuncSetAttribute for 61440 B, then fails the
  // LAUNCH with "invalid argument".  What such a card will launch is its
  // per-block limit, so the capacity comes from that below sm_70 - the same
  // tokens the "no opt-in" branch assumes, but taken from the attribute that is
  // actually enforced.
#if defined(__HIPCC__)
        const int usable = optin > 0 ? optin : 48 * 1024;
#else
        int cc = 0, per_block = 0;
        cc = dpct::get_device(dev).get_major_version();
        per_block = (int) dpct::get_device(dev).get_local_mem_size();   // SYCL: the work-group local memory
        cc = strata::cc_major_of(cc);
        const int usable = (cc >= 7 && optin > 0) ? optin : per_block;
#endif
        const int capacity = usable / (int) (tv * sizeof(float));
        chunk[dev] = capacity < 1 ? 1 : (capacity > kFusedGrMaxT ? kFusedGrMaxT : capacity);
        const int capacity_staged = usable / (int) (2 * H_TILE * sizeof(float));
        chunk_staged[dev] = capacity_staged < 1 ? 1 : (capacity_staged > kFusedGrMaxT ? kFusedGrMaxT : capacity_staged);
        attr[dev] = true;
    }
    *tile_out = tile[dev] ? tile[dev] : 2560;
    return staged ? chunk_staged[dev] : chunk[dev];
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// The multi read as `variant` (kHcPlain, kHcSplit or kHcStaged): the norm, the down projection in launches of as
// many tokens as fit the card, the up projection; the profile's stamps after the norm and after the down projection.
// kHcPlain is the default read exactly as before (never main's opt-in STRATA_GR_V3 path, which fused_gr_read_multi
// takes first).
#if defined(STRATA_HIP_GFX906)
__global__ void __launch_bounds__(THREADS) gr_norm_fast_kernel(GrMulti m);   // below, with the AMD fast path
__global__ void __launch_bounds__(THREADS) gr_up_fast_kernel(GrMulti m);
}  // namespace
static bool gr_fast();
namespace {
#endif
// STRATA_GR_DOWN_MAX4=1 (opt-in): launches of up to 4 tokens hold 4 tokens' sums per thread instead of kFusedGrMaxT (the
// same bits; checked bitwise on the 5070 for 0.1.40). Default off: a 10-pair interleaved decode A/B on the RTX 5070 (Q2_0,
// default config) did not show the +2.8-3.5% of bench #832 (medians 0.97-0.99 of the off arm).
bool gr_down_max4() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_GR_DOWN_MAX4");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}

void launch_multi(const GrMulti &m, int variant, dpct::queue_ptr st,
                  unsigned long long *stamp_buf, int stamp_i0) {
    const int n_tok = m.T;
#if defined(STRATA_HIP_GFX906)
    // gfx906: the latency-hidden norm/up (STRATA_GR_FAST=0: off) - the same sums in the same order as the kernels
    // they replace (gr_parity checks the multi-token read against single-token calls bitwise)
    const bool fast = gr_fast();
    if (variant >= kHcSplit) gr_norm_split_kernel<<<dim3((unsigned) n_tok, HC), THREADS, 0, st>>>(m);
    else if (fast) gr_norm_fast_kernel<<<n_tok, THREADS, 0, st>>>(m);
    else gr_norm_multi_kernel<<<n_tok, THREADS, 0, st>>>(m);
#else
    if (variant >= kHcSplit) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class gr_norm_split_kernel_200589>>(
            sycl::nd_range<3>(sycl::range(1, HC, (unsigned)n_tok) *
                                  sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gr_norm_split_kernel(m);
            });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class gr_norm_multi_kernel_2f4d93>>(
            sycl::nd_range<3>(sycl::range(1, 1, n_tok) *
                                  sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gr_norm_multi_kernel(m);
            });
    }
#endif
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, (void*) st);
    const bool staged = variant >= kHcStaged;
    int tv = 2560;
    const int chunk_tok = down_chunk(staged, &tv);
    /*
    DPCT1083: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    const size_t per_tok =
        (staged ? (size_t)2 * H_TILE : (size_t)tv) * sizeof(float);
    static const bool no_multi_gr = [] {
        const char* no = std::getenv("STRATA_NO_MULTI_GR");
        return no != nullptr && no[0] != '\0' && no[0] != '0';
    }();
    const bool exact_t = !no_multi_gr;
    for (int c0 = 0; c0 < n_tok; c0 += chunk_tok) {
        const int ct = n_tok - c0 < chunk_tok ? n_tok - c0 : chunk_tok;
        GrMulti c{};
        if (ct == n_tok) {
            c = m;
        } else {
            c.xn = m.xn + (size_t) c0 * D;
            c.T = ct;
            for (int k = 0; k < ct; ++k) c.a[k] = m.a[c0 + k];
        }
        const size_t smem = (size_t) ct * per_tok;
        // #443, STRATA_GR_DOWN_MAX4: launches of up to 4 tokens hold 4 tokens' sums per thread instead of
        // kFusedGrMaxT - the same tile, block size, accumulation order and plain/split/staged path, so the same bits
        // (decided once per device: this runs per layer when decode is not captured)
        const bool max4 = ct <= 4 && gr_down_max4();
        if (staged) {
            // #783 PR-g (stuchapin909): a launch of exactly ct <= 6 tokens is its own instantiation, the loop bounds
            // are compile-time (the same sums in the same order); STRATA_NO_MULTI_GR=1 keeps the generic kernels
            if (exact_t && ct == 1) {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                st->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range(smem), cgh);

                    cgh.parallel_for<dpct_kernel_name<
                        class gr_down_staged_kernel_41b428,
                        dpct_kernel_scalar<1>, dpct_kernel_scalar<true>>>(
                        sycl::nd_range<3>(sycl::range(1, 1, DOWN_BLOCKS + 1) *
                                              sycl::range(1, 1, THREADS),
                                          sycl::range(1, 1, THREADS)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                gr_down_staged_kernel<1, true>(
                                    c, dpct_local_acc_ct1
                                           .get_multi_ptr<
                                               sycl::access::decorated::no>()
                                           .get());
                            });
                });
            } else if (exact_t && ct == 2) {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                st->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range(smem), cgh);

                    cgh.parallel_for<dpct_kernel_name<
                        class gr_down_staged_kernel_41b428,
                        dpct_kernel_scalar<2>, dpct_kernel_scalar<true>>>(
                        sycl::nd_range<3>(sycl::range(1, 1, DOWN_BLOCKS + 1) *
                                              sycl::range(1, 1, THREADS),
                                          sycl::range(1, 1, THREADS)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                gr_down_staged_kernel<2, true>(
                                    c, dpct_local_acc_ct1
                                           .get_multi_ptr<
                                               sycl::access::decorated::no>()
                                           .get());
                            });
                });
            } else if (exact_t && ct == 3) {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                st->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range(smem), cgh);

                    cgh.parallel_for<dpct_kernel_name<
                        class gr_down_staged_kernel_41b428,
                        dpct_kernel_scalar<3>, dpct_kernel_scalar<true>>>(
                        sycl::nd_range<3>(sycl::range(1, 1, DOWN_BLOCKS + 1) *
                                              sycl::range(1, 1, THREADS),
                                          sycl::range(1, 1, THREADS)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                gr_down_staged_kernel<3, true>(
                                    c, dpct_local_acc_ct1
                                           .get_multi_ptr<
                                               sycl::access::decorated::no>()
                                           .get());
                            });
                });
            } else if (exact_t && ct == 4) {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                st->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range(smem), cgh);

                    cgh.parallel_for<dpct_kernel_name<
                        class gr_down_staged_kernel_41b428,
                        dpct_kernel_scalar<4>, dpct_kernel_scalar<true>>>(
                        sycl::nd_range<3>(sycl::range(1, 1, DOWN_BLOCKS + 1) *
                                              sycl::range(1, 1, THREADS),
                                          sycl::range(1, 1, THREADS)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                gr_down_staged_kernel<4, true>(
                                    c, dpct_local_acc_ct1
                                           .get_multi_ptr<
                                               sycl::access::decorated::no>()
                                           .get());
                            });
                });
            } else if (exact_t && ct == 5) {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                st->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range(smem), cgh);

                    cgh.parallel_for<dpct_kernel_name<
                        class gr_down_staged_kernel_41b428,
                        dpct_kernel_scalar<5>, dpct_kernel_scalar<true>>>(
                        sycl::nd_range<3>(sycl::range(1, 1, DOWN_BLOCKS + 1) *
                                              sycl::range(1, 1, THREADS),
                                          sycl::range(1, 1, THREADS)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                gr_down_staged_kernel<5, true>(
                                    c, dpct_local_acc_ct1
                                           .get_multi_ptr<
                                               sycl::access::decorated::no>()
                                           .get());
                            });
                });
            } else if (exact_t && ct == 6) {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                st->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range(smem), cgh);

                    cgh.parallel_for<dpct_kernel_name<
                        class gr_down_staged_kernel_41b428,
                        dpct_kernel_scalar<6>, dpct_kernel_scalar<true>>>(
                        sycl::nd_range<3>(sycl::range(1, 1, DOWN_BLOCKS + 1) *
                                              sycl::range(1, 1, THREADS),
                                          sycl::range(1, 1, THREADS)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                gr_down_staged_kernel<6, true>(
                                    c, dpct_local_acc_ct1
                                           .get_multi_ptr<
                                               sycl::access::decorated::no>()
                                           .get());
                            });
                });
            } else if (max4) {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                st->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range(smem), cgh);

                    /*
                    DPCT1050: The template argument of the dpct_kernel_name
                    could not be deduced. You need to update this code.
                    */
                    cgh.parallel_for<dpct_kernel_name<
                        class gr_down_staged_kernel_41b428,
                        dpct_kernel_scalar<4>, dpct_kernel_scalar<false>>>(
                        sycl::nd_range<3>(sycl::range(1, 1, DOWN_BLOCKS + 1) *
                                              sycl::range(1, 1, THREADS),
                                          sycl::range(1, 1, THREADS)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                gr_down_staged_kernel<4>(
                                    c, dpct_local_acc_ct1
                                           .get_multi_ptr<
                                               sycl::access::decorated::no>()
                                           .get());
                            });
                });
            } else {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                st->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range(smem), cgh);

                    /*
                    DPCT1050: The template argument of the dpct_kernel_name
                    could not be deduced. You need to update this code.
                    */
                    cgh.parallel_for<dpct_kernel_name<
                        class gr_down_staged_kernel_41b428,
                        dpct_kernel_scalar<kFusedGrMaxT>, dpct_kernel_scalar<false>>>(
                        sycl::nd_range<3>(sycl::range(1, 1, DOWN_BLOCKS + 1) *
                                              sycl::range(1, 1, THREADS),
                                          sycl::range(1, 1, THREADS)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                gr_down_staged_kernel(
                                    c, dpct_local_acc_ct1
                                           .get_multi_ptr<
                                               sycl::access::decorated::no>()
                                           .get());
                            });
                });
            }
        } else if (tv == 1280) {
            if (max4) {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                st->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range(smem), cgh);

                    cgh.parallel_for<dpct_kernel_name<
                        class gr_down_multi_kernel_4a9670,
                        dpct_kernel_scalar<1280>, dpct_kernel_scalar<4>>>(
                        sycl::nd_range<3>(sycl::range(1, 1, DOWN_BLOCKS + 1) *
                                              sycl::range(1, 1, THREADS),
                                          sycl::range(1, 1, THREADS)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                gr_down_multi_kernel<1280, 4>(
                                    c, dpct_local_acc_ct1
                                           .get_multi_ptr<
                                               sycl::access::decorated::no>()
                                           .get());
                            });
                });
            } else {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                st->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range(smem), cgh);

                    /*
                    DPCT1050: The template argument of the dpct_kernel_name
                    could not be deduced. You need to update this code.
                    */
                    cgh.parallel_for<dpct_kernel_name<
                        class gr_down_multi_kernel_6a749c,
                        dpct_kernel_scalar<1280>, dpct_kernel_scalar<1280>>>(
                        sycl::nd_range<3>(sycl::range(1, 1, DOWN_BLOCKS + 1) *
                                              sycl::range(1, 1, THREADS),
                                          sycl::range(1, 1, THREADS)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                gr_down_multi_kernel<1280>(
                                    c, dpct_local_acc_ct1
                                           .get_multi_ptr<
                                               sycl::access::decorated::no>()
                                           .get());
                            });
                });
            }
        } else {
            if (max4) {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                st->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range(smem), cgh);

                    cgh.parallel_for<dpct_kernel_name<
                        class gr_down_multi_kernel_2d8885,
                        dpct_kernel_scalar<2560>, dpct_kernel_scalar<4>>>(
                        sycl::nd_range<3>(sycl::range(1, 1, DOWN_BLOCKS + 1) *
                                              sycl::range(1, 1, THREADS),
                                          sycl::range(1, 1, THREADS)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                gr_down_multi_kernel<2560, 4>(
                                    c, dpct_local_acc_ct1
                                           .get_multi_ptr<
                                               sycl::access::decorated::no>()
                                           .get());
                            });
                });
            } else {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                st->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range(smem), cgh);

                    /*
                    DPCT1050: The template argument of the dpct_kernel_name
                    could not be deduced. You need to update this code.
                    */
                    cgh.parallel_for<dpct_kernel_name<
                        class gr_down_multi_kernel_df2ead,
                        dpct_kernel_scalar<2560>, dpct_kernel_scalar<2560>>>(
                        sycl::nd_range<3>(sycl::range(1, 1, DOWN_BLOCKS + 1) *
                                              sycl::range(1, 1, THREADS),
                                          sycl::range(1, 1, THREADS)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                gr_down_multi_kernel<2560>(
                                    c, dpct_local_acc_ct1
                                           .get_multi_ptr<
                                               sycl::access::decorated::no>()
                                           .get());
                            });
                });
            }
        }
    }
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, (void*) st);
#if defined(STRATA_HIP_GFX906)
    if (fast) gr_up_fast_kernel<<<UPM_BLOCKS, THREADS, 0, st>>>(m);
    else
#endif
    switch (no_multi_gr ? kFusedGrMaxT : n_tok) {
        case 1: {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<dpct_kernel_name<class gr_up_multi_kernel_1a7280,
                                              dpct_kernel_scalar<1>,
                                              dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, UPM_BLOCKS) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gr_up_multi_kernel<1, true>(m);
                    });
        } break;
        case 2: {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<dpct_kernel_name<class gr_up_multi_kernel_1a7280,
                                              dpct_kernel_scalar<2>,
                                              dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, UPM_BLOCKS) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gr_up_multi_kernel<2, true>(m);
                    });
        } break;
        case 3: {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<dpct_kernel_name<class gr_up_multi_kernel_1a7280,
                                              dpct_kernel_scalar<3>,
                                              dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, UPM_BLOCKS) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gr_up_multi_kernel<3, true>(m);
                    });
        } break;
        case 4: {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<dpct_kernel_name<class gr_up_multi_kernel_1a7280,
                                              dpct_kernel_scalar<4>,
                                              dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, UPM_BLOCKS) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gr_up_multi_kernel<4, true>(m);
                    });
        } break;
        case 5: {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<dpct_kernel_name<class gr_up_multi_kernel_1a7280,
                                              dpct_kernel_scalar<5>,
                                              dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, UPM_BLOCKS) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gr_up_multi_kernel<5, true>(m);
                    });
        } break;
        case 6: {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<dpct_kernel_name<class gr_up_multi_kernel_1a7280,
                                              dpct_kernel_scalar<6>,
                                              dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, UPM_BLOCKS) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gr_up_multi_kernel<6, true>(m);
                    });
        } break;
        default: {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<dpct_kernel_name<class gr_up_multi_kernel_1a7280,
                                              dpct_kernel_scalar<kFusedGrMaxT>,
                                              dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(sycl::range(1, 1, UPM_BLOCKS) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gr_up_multi_kernel<kFusedGrMaxT, false>(m);
                    });
        } break;
    }
}

/// STRATA_HC_SPLIT: unset or 2 = the newest the check accepts (staged), 1 = at most split, 0 = the plain read
int env_variant() {
    const char* e = std::getenv("STRATA_HC_SPLIT");
    if (e == nullptr || e[0] == '\0') return kHcStaged;
    if (e[0] == '0') return kHcPlain;
    if (e[0] == '1') return kHcSplit;
    return kHcStaged;
}

// per device: the variant `fused_gr_check` chose (0 = not checked yet)
std::atomic<int> g_variant[64];


#if defined(STRATA_HIP_GFX906)
// ---- AMD fast path (STRATA_GR_FAST, default 1; 0 = the old kernels): the same arithmetic per value in the same order, latency
// hidden.  gr_norm: one block per token (as before) but a thread's 10 float4 of R / bo / w_norm are loaded
// before any is used, and xn stays in registers until rs is known (was: store, then re-read the 40 KB).
// gr_up: a warp's 8 rows of w_up and their epilogue inputs are loaded before the dots (was: one row's load chain
// after another).
constexpr int NQ = D / (THREADS * 4);   // 10 float4 per thread
static_assert(D % (THREADS * 4) == 0, "whole float4 per thread");
__global__ void __launch_bounds__(THREADS) gr_norm_fast_kernel(GrMulti m) {
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const FusedGrArgs& a = m.a[blockIdx.x];
    float* xn = m.xn + (size_t) blockIdx.x * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float4 r[NQ], g[NQ];
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const int i = t * 4 + k * THREADS * 4;
        r[k] = *reinterpret_cast<const float4*>(a.R + i);
        g[k] = *reinterpret_cast<const float4*>(a.w_norm + i);
    }
    if (a.apply) {
#pragma unroll
        for (int k = 0; k < NQ; ++k) {
            const int i = t * 4 + k * THREADS * 4, c = i / N, d = i - c * N;
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r[k].x = fmaf(b.x, gw[c], r[k].x); r[k].y = fmaf(b.y, gw[c], r[k].y);
            r[k].z = fmaf(b.z, gw[c], r[k].z); r[k].w = fmaf(b.w, gw[c], r[k].w);
        }
    }
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const int i = t * 4 + k * THREADS * 4, c = i / N;
        const float sq = r[k].x * r[k].x + r[k].y * r[k].y + r[k].z * r[k].z + r[k].w * r[k].w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        r[k] = make_float4(r[k].x * g[k].x, r[k].y * g[k].y, r[k].z * g[k].z, r[k].w * g[k].w);
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = rsqrtf(s / (float) N + a.eps);
        a.rs[t] = s_rs[t];
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const int i = t * 4 + k * THREADS * 4;
        const float sr = s_rs[i / N];
        *reinterpret_cast<float4*>(xn + i) = make_float4(r[k].x * sr, r[k].y * sr, r[k].z * sr, r[k].w * sr);
    }
}
// gr_up: 8 lanes per row instead of a 32-lane warp.  Lane j holds the old lanes j, j+8, j+16, j+24 (and chunk
// 32+j, the old lane j's second chunk): the xor tree's stages 16 and 8 are its own adds (a + b == b + a), stages
// 4, 2, 1 are shuffles inside the 8 - the same tree, bitwise the same sum, 3 shuffles a token instead of 5 and no
// idle lanes on the tail chunks.  A block's 64 rows are 2 passes of 32 groups, their weights loaded up front.
__device__ __forceinline__ float xor8(float v) {
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
    return v;
}
__global__ void __launch_bounds__(THREADS) gr_up_fast_kernel(GrMulti m) {
    __shared__ __align__(16) float lo[kFusedGrMaxT][LR];
    __shared__ float g[kFusedGrMaxT][HC][UPM_COLS];
    const int t = threadIdx.x, j = t & 7, grp = t >> 3;
    const int T = m.T;
    const int d0 = blockIdx.x * UPM_COLS;
    static_assert(LR == 40 * 8 && HC * UPM_COLS == 64 && THREADS == 256, "geometry");
    uint4 w[2][5];
    float rv[2] = {0.0f, 0.0f}, wn[2] = {0.0f, 0.0f}, rsc[2] = {0.0f, 0.0f}, bo[2] = {0.0f, 0.0f}, ip[2] = {0.0f, 0.0f};
    const bool apply = j < T && m.a[j < T ? j : 0].apply;
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int r = grp + 32 * p, c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
#pragma unroll
        for (int q = 0; q < 4; ++q) w[p][q] = __ldg(w4 + j + 8 * q);
        w[p][4] = __ldg(w4 + 32 + j);
        if (j < T) {
            const FusedGrArgs& a = m.a[j];
            rv[p] = a.R[i];
            wn[p] = a.w_norm[i];
            rsc[p] = a.rs[c];
            if (apply) { bo[p] = a.bo_prev[d0 + dd]; ip[p] = a.inj_prev[c]; }
        }
    }
    for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    __syncthreads();
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int r = grp + 32 * p, c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            const float* l = lo[k];
            const float p0 = dot8(w[p][0], l + j * 8) + dot8(w[p][4], l + (32 + j) * 8);   // old lane j
            const float p1 = dot8(w[p][1], l + (j + 8) * 8);                               // old lane j + 8
            const float p2 = dot8(w[p][2], l + (j + 16) * 8);                              // old lane j + 16
            const float p3 = dot8(w[p][3], l + (j + 24) * 8);                              // old lane j + 24
            const float s = xor8((p0 + p2) + (p1 + p3));   // stage 16: (j, j+16), (j+8, j+24); stage 8; then 4, 2, 1
            if (j == k) mine = s;
        }
        if (j < T) {
            float x0 = rv[p];
            if (apply) {
                x0 = fmaf(bo[p], 2.0f * sigmoidf_(ip[p] / (float) HC), x0);
                m.a[j].R_out[i] = x0;
            }
            const float x = x0 * wn[p] * rsc[p];
            g[j][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
}
#endif

#if defined(STRATA_HIP_GFX906)
// ---- AMD: `gr_down_multi` split along K.  The CUDA kernel is one warp per row of w_down (320 + 4 rows), 41
// blocks: on a 60-CU gfx906 that left most of the card idle and read the 6.5 MB matrix at ~100 GB/s.  Here a block
// is 8 wavefronts = 8 rows of one K-slice of 2048 (the slice of every token's xn staged in LDS once), 41 x 5
// blocks, and a second kernel adds the 5 slice sums in a fixed order and runs the epilogue.  A token's result does
// not depend on how many tokens share the window (each column is reduced on its own).
constexpr int GS_SL = 2048;                   // K per slice
constexpr int GS_S = D / GS_SL;               // 5 slices
constexpr int GS_ROWS = LR + HC;              // 324: the down rows, then the inject rows
constexpr int GS_WAVES = 8;
static_assert(D % GS_SL == 0 && GS_SL == 64 * 8 * 4, "a lane takes 4 chunks of 8 per slice");
// the slice sums: a module-scope device array is per device by construction, and needs no allocation during a
// graph capture
__device__ float g_gr_part[GS_S * kFusedGrMaxT * GS_ROWS];
__global__ void __launch_bounds__(GS_WAVES * 64) gr_down_split_kernel(GrMulti m) {
    float* part = g_gr_part;
    extern __shared__ __align__(16) float tile[];   // [T][GS_SL]
    const int t = threadIdx.x, lane = t & 63, wave = t >> 6;
    const int T = m.T, sl = blockIdx.y;
    const bool inject_block = blockIdx.x == DOWN_BLOCKS;
    const int row = inject_block ? wave : blockIdx.x * GS_WAVES + wave;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || wave >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow) + sl * (GS_SL / 8);
    uint4 wv[4];
    if (active) {
#pragma unroll
        for (int q = 0; q < 4; ++q) wv[q] = __ldg(w4 + lane + 64 * q);
    }
    const float4* src4 = reinterpret_cast<const float4*>(m.xn);
    float4* tile4 = reinterpret_cast<float4*>(tile);
    for (int i = t; i < T * (GS_SL / 4); i += GS_WAVES * 64) {
        const int k = i / (GS_SL / 4), off = i - k * (GS_SL / 4);
        tile4[i] = src4[((size_t) k * D + (size_t) sl * GS_SL) / 4 + off];
    }
    __syncthreads();
    if (!active) return;
    float acc[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) acc[k] = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int j = lane + 64 * q;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k)
            if (k < T) acc[k] += dot8(wv[q], tile + k * GS_SL + j * 8);
    }
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) {
        if (k >= T) break;
        float v = acc[k];
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
        if (lane == 0) part[((size_t) sl * kFusedGrMaxT + k) * GS_ROWS + (inject_block ? LR + row : row)] = v;
    }
}
__global__ void gr_down_finish_kernel(GrMulti m) {
    const float* part = g_gr_part;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = i / GS_ROWS, r = i - k * GS_ROWS;
    if (k >= m.T) return;
    const bool inject = r >= LR;
    if (inject && m.a[0].w_inject == nullptr) return;
    float s = 0.0f;
#pragma unroll
    for (int sl = 0; sl < GS_S; ++sl) s += part[((size_t) sl * kFusedGrMaxT + k) * GS_ROWS + r];
    if (inject) {
        m.a[k].inject_out[r - LR] = s;
    } else {
        const float x = s / (float) HC;
        m.a[k].lo[r] = x / (1.0f + __expf(-x));
    }
}

#endif

// ================================ S23 experiment (STRATA_HC_Q8=1): the read with the GGUF's Q8_0 projections ======
// The pack holds the hyper-connection projections as BF16 (iq_pack.py --compat-bf16 rounds the GGUF's Q8_0 values to
// BF16: 1.20 GiB per verify step); this read takes the Q8_0 bytes as stored (0.65 GiB - and the GGUF's own values,
// not a rounding of them).  The layout is the stream split of the v3 read above, at 4 chunks per stream: `down` runs
// on (10 groups of 32 rows + the inject rows) x 16 chunks of 640 columns = 176 blocks, each staging its chunk of
// R' * w_norm (unscaled) for the T tokens and writing the chunk's partial dots and sums of squares; `up` sums them
// per stream in a fixed order, applies rs, SwiGLU-free silu, and runs the default epilogue.  Another summation order
// and other weights than the default read: an output-changing step (measured, quality-checked).
constexpr int Q8B = 34;                             // Q8_0 block: fp16 d + 32 int8
constexpr int Q8_RPW = 4;                           // down rows per warp
constexpr int Q8_RG = LR / (WARPS * Q8_RPW);        // 10 row groups (+1: the inject rows)
constexpr int Q8_KC = 640, Q8_NKC = D / Q8_KC;      // 16 chunks, 4 per stream
constexpr int Q8_CPS = N / Q8_KC;                   // chunks per stream
constexpr int Q8_SPB = Q8_KC / 128;                 // 5 steps of 4 Q8_0 blocks (8 lanes x 4 values each)
__dpct_inline__ uint32_t q8_ld16(const uint8_t *p) {
    return *(const uint16_t *)p;
}
__dpct_inline__ float q8_half(uint32_t h) {
    return sycl::vec<sycl::half, 1>(
               sycl::bit_cast<sycl::half, unsigned short>((unsigned short)h))
        .convert<float, sycl::rounding_mode::automatic>()[0];
}
__dpct_inline__ sycl::float4 q8_four(uint32_t q, float d) {
    return sycl::float4(
        d * (float)(int8_t)(q & 255u), d * (float)(int8_t)((q >> 8) & 255u),
        d * (float)(int8_t)((q >> 16) & 255u), d * (float)(int8_t)(q >> 24));
}

template <int T>
/*
DPCT1110: The total declared local variable size in device function
gr_down_q8_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gr_down_q8_kernel(GrMulti m, float *__restrict__ part,
                                       float *__restrict__ ssg) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &xs =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<float[T][Q8_KC]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARPS][T]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int rg = item_ct1.get_group(2), kc = item_ct1.get_group(1),
              c = kc / Q8_CPS;
    const bool inj = rg == Q8_RG;
    // S25: the inject rows are F32 in the GGUF with BF16-exact values, so by default they are read as the pack's BF16
    // (exact); a Q8_0 copy (STRATA_HC_Q8_INJECT=1) is the other option
    const bool inj_bf16 = inj && m.a[0].q8_inject == nullptr;
    const int nrows = inj ? (((m.a[0].q8_inject != nullptr || m.a[0].w_inject != nullptr) && warp < HC) ? 1 : 0) : Q8_RPW;
    const int row0 = inj ? warp : (rg * WARPS + warp) * Q8_RPW;
    const uint8_t* wb = inj ? m.a[0].q8_inject : m.a[0].q8_down;
    const int sub = lane & 7, bl = lane >> 3;       // lanes 8b..8b+7: one Q8_0 block, 4 values each
    uint32_t q[Q8_RPW][Q8_SPB];
    float dq[Q8_RPW][Q8_SPB];
#pragma unroll
    for (int r = 0; r < Q8_RPW; ++r) {
        if (r >= nrows || inj_bf16) break;
        const uint8_t* row = wb + (size_t) (row0 + r) * (D / 32) * Q8B + (size_t) kc * (Q8_KC / 32) * Q8B;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const uint8_t* blk = row + (4 * s + bl) * Q8B;
            dq[r][s] = q8_half(q8_ld16(blk));
            q[r][s] = q8_ld16(blk + 2 + 4 * sub) | q8_ld16(blk + 4 + 4 * sub) << 16;
        }
    }
    float ssp[T];
#pragma unroll
    for (int k = 0; k < T; ++k) {
        ssp[k] = 0.0f;
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
        for (int i = t; i < Q8_KC / 4; i += THREADS) {
            const int col = kc * Q8_KC + 4 * i;
            sycl::float4 r = *reinterpret_cast<const sycl::float4 *>(a.R + col);
            if (a.apply) {
                const sycl::float4 b = *reinterpret_cast<const sycl::float4 *>(
                    a.bo_prev + (col - c * N));
                r.x() = sycl::fma((float)(b.x()), (float)gw, r.x());
                    r.y() = sycl::fma((float)(b.y()), (float)gw, r.y());
                r.z() = sycl::fma((float)(b.z()), (float)gw, r.z());
                    r.w() = sycl::fma((float)(b.w()), (float)gw, r.w());
            }
            const sycl::float4 g =
                *reinterpret_cast<const sycl::float4 *>(a.w_norm + col);
            ssp[k] +=
                r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
            *reinterpret_cast<sycl::float4 *>(&xs[k][4 * i]) = sycl::float4(
                r.x() * g.x(), r.y() * g.y(), r.z() * g.z(), r.w() * g.w());
        }
    }
#pragma unroll
    for (int k = 0; k < T; ++k) {
        const float v = warp_sum(ssp[k]);
        if (lane == 0) red[warp][k] = v;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (rg == 0 && t < T) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) s += red[w][t];
        ssg[t * Q8_NKC + kc] = s;
    }
    if (nrows == 0) return;
#pragma unroll
    for (int r = 0; r < Q8_RPW; ++r) {
        if (r >= nrows) break;
        float acc[T];
#pragma unroll
        for (int k = 0; k < T; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const int v = 32 * (4 * s + bl) + 4 * sub;
            sycl::float4 w;
            if (inj_bf16) {
                const sycl::uint2 b = *reinterpret_cast<const sycl::uint2 *>(
                    m.a[0].w_inject + (size_t)row0 * D + (size_t)kc * Q8_KC +
                    v);
                w = sycl::float4(sycl::bit_cast<float>(b.x() << 16),
                                 sycl::bit_cast<float>(b.x() & 0xffff0000u),
                                 sycl::bit_cast<float>(b.y() << 16),
                                 sycl::bit_cast<float>(b.y() & 0xffff0000u));
            } else {
                w = q8_four(q[r][s], dq[r][s]);
            }
#pragma unroll
            for (int k = 0; k < T; ++k) {
                const sycl::float4 x =
                    *reinterpret_cast<const sycl::float4 *>(&xs[k][v]);
                acc[k] = sycl::fma(w.x(), (float)(x.x()), acc[k]);
                    acc[k] = sycl::fma(w.y(), (float)(x.y()), acc[k]);
                acc[k] = sycl::fma(w.z(), (float)(x.z()), acc[k]);
                    acc[k] = sycl::fma(w.w(), (float)(x.w()), acc[k]);
            }
        }
        const int prow = inj ? LR + warp : row0 + r;
#pragma unroll
        for (int k = 0; k < T; ++k) {
            const float v = warp_sum(acc[k]);
            if (lane == 0) part[((size_t) k * Q8_NKC + kc) * PR + prow] = v;
        }
    }
}

template <int T>
/*
DPCT1110: The total declared local variable size in device function
gr_up_q8_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gr_up_q8_kernel(GrMulti m, const float *__restrict__ part,
                                     const float *__restrict__ ssg) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &lo = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[T][LR]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &rsS =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[T][HC]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &g = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[T][HC][UPM_COLS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int d0 = item_ct1.get_group(2) * UPM_COLS;
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < Q8_CPS; ++j) ss += ssg[k * Q8_NKC + c * Q8_CPS + j];
        const float r = sycl::rsqrt(ss / (float)N + m.a[k].eps);
        rsS[k][c] = r;
        if (item_ct1.get_group(2) == 0) m.a[k].rs[c] = r;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = t; i < T * PR; i += THREADS) {
        const int k = i / PR, r = i - k * PR;
        if (r >= LR && (item_ct1.get_group(2) != 0 ||
                        m.a[k].w_inject == nullptr)) continue;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
#pragma unroll
            for (int j = 0; j < Q8_CPS; ++j) p += part[((size_t) k * Q8_NKC + c * Q8_CPS + j) * PR + r];
            sum = sycl::fma(rsS[k][c], p, sum);
        }
        if (r < LR) {
            const float x = sum / (float) HC;
            lo[k][r] = x / (1.0f + sycl::native::exp(-x));
        } else {
            m.a[k].inject_out[r - LR] = sum;
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    constexpr int RPW8 = HC * UPM_COLS / WARPS;     // 8 rows per warp
#pragma unroll 1
    for (int qq = 0; qq < RPW8; ++qq) {
        const int r = warp + qq * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint8_t* row = m.a[0].q8_up + (size_t) i * (LR / 32) * Q8B;
        sycl::float4 w[3];
#pragma unroll
        for (int s = 0; s < 3; ++s) {   // lane: values 4 lane + 128 s (the third step: lanes 0-15)
            const int v = 4 * lane + 128 * s;
            if (v < LR) {
                const uint8_t* blk = row + (v >> 5) * Q8B;
                w[s] = q8_four(q8_ld16(blk + 2 + (v & 31)) | q8_ld16(blk + 4 + (v & 31)) << 16, q8_half(q8_ld16(blk)));
            } else {
                w[s] = sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        float rv = 0.0f, wn = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < T; ++k) {
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < 3; ++s) {
                const int v = 4 * lane + 128 * s;
                if (v < LR) {
                    const sycl::float4 x =
                        *reinterpret_cast<const sycl::float4 *>(&lo[k][v]);
                    acc = sycl::fma(w[s].x(), (float)(x.x()), acc);
                        acc = sycl::fma(w[s].y(), (float)(x.y()), acc);
                    acc = sycl::fma(w[s].z(), (float)(x.z()), acc);
                        acc = sycl::fma(w[s].w(), (float)(x.w()), acc);
                }
            }
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = sycl::fma(bo, 2.0f * sigmoidf_(ip / (float)HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsS[lane][c];
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE
}


// S26 STRATA_TSUM=1: gr_down_q8_kernel / gr_up_q8_kernel with each row's T per-token warp sums (and the sums of
// squares) as one transposed butterfly (s26_tsum.cuh): bitwise the same values (S26 ws4 harness, every output equal,
// T 1-4: 1.00 / 1.03 / 1.03 / 1.15x per read). PF (all rows' weights in registers before the part / lo phase) was
// slower (0.92-1.03x) and is off.
using s26ts::tsum;
using s26ts::tsum_token;
using s26ts::tsum_lane;
using s26ts::pow2_ceil;
// gr_down_q8_kernel / gr_up_q8_kernel with every per-token warp sum done as one transposed butterfly per row (tsum)
template <int T, int RPW = Q8_RPW>
/*
DPCT1110: The total declared local variable size in device function
gr_down_q8_fast_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gr_down_q8_fast_kernel(GrMulti m, float *__restrict__ part,
                                            float *__restrict__ ssg) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int P = pow2_ceil(T);
    constexpr int RG_ = LR / (WARPS * RPW);
    auto &xs =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[T][Q8_KC]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARPS][T]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int rg = item_ct1.get_group(2), kc = item_ct1.get_group(1),
              c = kc / Q8_CPS;
    const bool inj = rg == RG_;
    const bool inj_bf16 = inj && m.a[0].q8_inject == nullptr;
    const int nrows = inj ? (((m.a[0].q8_inject != nullptr || m.a[0].w_inject != nullptr) && warp < HC) ? 1 : 0) : RPW;
    const int row0 = inj ? warp : (rg * WARPS + warp) * RPW;
    const uint8_t* wb = inj ? m.a[0].q8_inject : m.a[0].q8_down;
    const int sub = lane & 7, bl = lane >> 3;
    const int tk = tsum_token<P>(lane);
    const bool owner = lane == tsum_lane<P>(tk) && tk < T;
    uint32_t q[RPW][Q8_SPB];
    float dq[RPW][Q8_SPB];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        if (r >= nrows || inj_bf16) break;
        const uint8_t* row = wb + (size_t) (row0 + r) * (D / 32) * Q8B + (size_t) kc * (Q8_KC / 32) * Q8B;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const uint8_t* blk = row + (4 * s + bl) * Q8B;
            dq[r][s] = q8_half(q8_ld16(blk));
            q[r][s] = q8_ld16(blk + 2 + 4 * sub) | q8_ld16(blk + 4 + 4 * sub) << 16;
        }
    }
    float ssp[P];
#pragma unroll
    for (int k = 0; k < P; ++k) ssp[k] = 0.0f;
#pragma unroll
    for (int k = 0; k < T; ++k) {
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
        for (int i = t; i < Q8_KC / 4; i += THREADS) {
            const int col = kc * Q8_KC + 4 * i;
            sycl::float4 r = *reinterpret_cast<const sycl::float4 *>(a.R + col);
            if (a.apply) {
                const sycl::float4 b = *reinterpret_cast<const sycl::float4 *>(
                    a.bo_prev + (col - c * N));
                r.x() = sycl::fma((float)(b.x()), (float)gw, r.x());
                    r.y() = sycl::fma((float)(b.y()), (float)gw, r.y());
                r.z() = sycl::fma((float)(b.z()), (float)gw, r.z());
                    r.w() = sycl::fma((float)(b.w()), (float)gw, r.w());
            }
            const sycl::float4 g =
                *reinterpret_cast<const sycl::float4 *>(a.w_norm + col);
            ssp[k] +=
                r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
            *reinterpret_cast<sycl::float4 *>(&xs[k][4 * i]) = sycl::float4(
                r.x() * g.x(), r.y() * g.y(), r.z() * g.z(), r.w() * g.w());
        }
    }
    {
        const float v = tsum<P>(ssp, lane);
        if (owner) red[warp][tk] = v;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (rg == 0 && t < T) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) s += red[w][t];
        ssg[t * Q8_NKC + kc] = s;
    }
    if (nrows == 0) return;
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        if (r >= nrows) break;
        float acc[P];
#pragma unroll
        for (int k = 0; k < P; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const int v = 32 * (4 * s + bl) + 4 * sub;
            sycl::float4 w;
            if (inj_bf16) {
                const sycl::uint2 b = *reinterpret_cast<const sycl::uint2 *>(
                    m.a[0].w_inject + (size_t)row0 * D + (size_t)kc * Q8_KC +
                    v);
                w = sycl::float4(sycl::bit_cast<float>(b.x() << 16),
                                 sycl::bit_cast<float>(b.x() & 0xffff0000u),
                                 sycl::bit_cast<float>(b.y() << 16),
                                 sycl::bit_cast<float>(b.y() & 0xffff0000u));
            } else {
                w = q8_four(q[r][s], dq[r][s]);
            }
#pragma unroll
            for (int k = 0; k < T; ++k) {
                const sycl::float4 x =
                    *reinterpret_cast<const sycl::float4 *>(&xs[k][v]);
                acc[k] = sycl::fma(w.x(), (float)(x.x()), acc[k]);
                    acc[k] = sycl::fma(w.y(), (float)(x.y()), acc[k]);
                acc[k] = sycl::fma(w.z(), (float)(x.z()), acc[k]);
                    acc[k] = sycl::fma(w.w(), (float)(x.w()), acc[k]);
            }
        }
        const int prow = inj ? LR + warp : row0 + r;
        const float v = tsum<P>(acc, lane);
        if (owner) part[((size_t) tk * Q8_NKC + kc) * PR + prow] = v;
    }
}

template <int T, bool PF = false>
/*
DPCT1110: The total declared local variable size in device function
gr_up_q8_fast_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gr_up_q8_fast_kernel(GrMulti m,
                                          const float *__restrict__ part,
                                          const float *__restrict__ ssg) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int P = pow2_ceil(T);
    constexpr int RPW8 = HC * UPM_COLS / WARPS;
    auto &lo =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[T][LR]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &rsS =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[T][HC]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &g = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[T][HC][UPM_COLS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int d0 = item_ct1.get_group(2) * UPM_COLS;
    const int tk = tsum_token<P>(lane);
    const bool owner = lane == tsum_lane<P>(tk) && tk < T;
    // PF: every row's weights in registers before the part / lo phase (its latency hides the weight read)
    uint32_t pq[PF ? RPW8 : 1][3];
    float pd[PF ? RPW8 : 1][3];
    if constexpr (PF) {
#pragma unroll
        for (int qq = 0; qq < RPW8; ++qq) {
            const int r = warp + qq * WARPS;
            const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
            const uint8_t* row = m.a[0].q8_up + (size_t) i * (LR / 32) * Q8B;
#pragma unroll
            for (int s = 0; s < 3; ++s) {
                const int v = 4 * lane + 128 * s;
                if (v < LR) {
                    const uint8_t* blk = row + (v >> 5) * Q8B;
                    pq[qq][s] = q8_ld16(blk + 2 + (v & 31)) | q8_ld16(blk + 4 + (v & 31)) << 16;
                    pd[qq][s] = q8_half(q8_ld16(blk));
                } else {
                    pq[qq][s] = 0u; pd[qq][s] = 0.0f;
                }
            }
        }
    }
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < Q8_CPS; ++j) ss += ssg[k * Q8_NKC + c * Q8_CPS + j];
        const float r = sycl::rsqrt(ss / (float)N + m.a[k].eps);
        rsS[k][c] = r;
        if (item_ct1.get_group(2) == 0) m.a[k].rs[c] = r;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = t; i < T * PR; i += THREADS) {
        const int k = i / PR, r = i - k * PR;
        if (r >= LR && (item_ct1.get_group(2) != 0 ||
                        m.a[k].w_inject == nullptr)) continue;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
#pragma unroll
            for (int j = 0; j < Q8_CPS; ++j) p += part[((size_t) k * Q8_NKC + c * Q8_CPS + j) * PR + r];
            sum = sycl::fma(rsS[k][c], p, sum);
        }
        if (r < LR) {
            const float x = sum / (float) HC;
            lo[k][r] = x / (1.0f + sycl::native::exp(-x));
        } else {
            m.a[k].inject_out[r - LR] = sum;
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#pragma unroll(PF ? RPW8 : 2)
    for (int qq = 0; qq < RPW8; ++qq) {
        const int r = warp + qq * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint8_t* row = m.a[0].q8_up + (size_t) i * (LR / 32) * Q8B;
        sycl::float4 w[3];
#pragma unroll
        for (int s = 0; s < 3; ++s) {
            const int v = 4 * lane + 128 * s;
            if (PF) {
                w[s] = v < LR ? q8_four(pq[PF ? qq : 0][s], pd[PF ? qq : 0][s])
                              : sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
            } else if (v < LR) {
                const uint8_t* blk = row + (v >> 5) * Q8B;
                w[s] = q8_four(q8_ld16(blk + 2 + (v & 31)) | q8_ld16(blk + 4 + (v & 31)) << 16, q8_half(q8_ld16(blk)));
            } else {
                w[s] = sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        float rv = 0.0f, wn = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (owner) {
            const FusedGrArgs& a = m.a[tk];
            rv = a.R[i];
            wn = a.w_norm[i];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float acc[P];
#pragma unroll
        for (int k = 0; k < P; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int k = 0; k < T; ++k) {
#pragma unroll
            for (int s = 0; s < 3; ++s) {
                const int v = 4 * lane + 128 * s;
                if (v < LR) {
                    const sycl::float4 x =
                        *reinterpret_cast<const sycl::float4 *>(&lo[k][v]);
                    acc[k] = sycl::fma(w[s].x(), (float)(x.x()), acc[k]);
                        acc[k] = sycl::fma(w[s].y(), (float)(x.y()), acc[k]);
                    acc[k] = sycl::fma(w[s].z(), (float)(x.z()), acc[k]);
                        acc[k] = sycl::fma(w[s].w(), (float)(x.w()), acc[k]);
                }
            }
        }
        const float mine = tsum<P>(acc, lane);
        if (owner) {
            if (apply) {
                rv = sycl::fma(bo, 2.0f * sigmoidf_(ip / (float)HC), rv);
                m.a[tk].R_out[i] = rv;
            }
            const float x = rv * wn * rsS[tk][c];
            g[tk][c][dd] = x * sigmoidf_(mine);
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE
}

template <int T>
void launch_q8_t(const GrMulti &m, float *part, float *ssg, dpct::queue_ptr st,
                 unsigned long long *stamp_buf, int stamp_i0) {
    static const bool ts = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, (void*) st);
    if (ts) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<
            dpct_kernel_name<class gr_down_q8_fast_kernel_c914e2,
                             dpct_kernel_scalar<T>, dpct_kernel_scalar<4>>>(
            sycl::nd_range<3>(sycl::range(1, Q8_NKC, LR / (WARPS * 4) + 1) *
                                  sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gr_down_q8_fast_kernel<T, 4>(m, part, ssg);
            });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class gr_down_q8_kernel_f1ac48,
                                          dpct_kernel_scalar<T>>>(
            sycl::nd_range<3>(sycl::range(1, Q8_NKC, Q8_RG + 1) *
                                  sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gr_down_q8_kernel<T>(m, part, ssg);
            });
    }
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, (void*) st);
    if (ts) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<
            dpct_kernel_name<class gr_up_q8_fast_kernel_ae7d51,
                             dpct_kernel_scalar<T>, dpct_kernel_scalar<false>>>(
            sycl::nd_range<3>(sycl::range(1, 1, UPM_BLOCKS) *
                                  sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gr_up_q8_fast_kernel<T, false>(m, part, ssg);
            });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class gr_up_q8_kernel_74ed29,
                                          dpct_kernel_scalar<T>>>(
            sycl::nd_range<3>(sycl::range(1, 1, UPM_BLOCKS) *
                                  sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gr_up_q8_kernel<T>(m, part, ssg);
            });
    }
}
void launch_q8(const GrMulti &m, float *scratch, dpct::queue_ptr st,
               unsigned long long *stamp_buf, int stamp_i0) {
    float* part = scratch;
    float* ssg = scratch + (size_t) m.T * Q8_NKC * PR;
    switch (m.T) {
        case 1: launch_q8_t<1>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 2: launch_q8_t<2>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 3: launch_q8_t<3>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 4: launch_q8_t<4>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 5: launch_q8_t<5>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 6: launch_q8_t<6>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        case 7: launch_q8_t<7>(m, part, ssg, st, stamp_buf, stamp_i0); break;
        default: launch_q8_t<8>(m, part, ssg, st, stamp_buf, stamp_i0); break;
    }
}

}  // namespace

// one partials buffer per queue (the verifier's and the drafter's launches never share a queue; within a queue the
// launches are ordered, and a graph capture records the pointer)
static float* down_partials(sycl::queue* q) {
    static std::mutex mu;
    static std::unordered_map<sycl::queue*, float*> bufs;
    std::lock_guard<std::mutex> lk(mu);
    auto it = bufs.find(q);
    if (it != bufs.end()) return it->second;
    float* p = sycl::malloc_device<float>((size_t) DOWN_SPLIT * kFusedGrMaxT * LR, *q);
    bufs[q] = p;
    return p;
}

#if defined(STRATA_HIP_GFX906)
int g_gr_fast = -1;   // fused_gr_set_fast (the bench); -1 = STRATA_GR_FAST
static bool gr_fast() {
    static const bool env = [] { const char* v = std::getenv("STRATA_GR_FAST"); return v ? std::atoi(v) != 0 : true; }();
    return g_gr_fast >= 0 ? g_gr_fast != 0 : env;
}
#else
int g_gr_fast = -1;
#endif
void fused_gr_set_fast(int on) { g_gr_fast = on; }

bool fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream, unsigned long long* stamp_buf,
                         int stamp_i0) {
    if (n_tok < 1 || n_tok > kFusedGrMaxT || xn_scratch == nullptr) {
        std::fprintf(stderr, "fused_gr_read_multi: invalid arguments\n");
        std::exit(1);
    }
    GrMulti m;
    for (int t = 0; t < n_tok; ++t) {
        m.a[t] = a[t];
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || !x.w_down || !x.w_up || !x.lo || !x.rs || !x.mixed || (x.w_inject && !x.inject_out) ||
            (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) || x.w_down != a[0].w_down || x.w_up != a[0].w_up ||
            x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm || x.q8_down != a[0].q8_down ||
            x.q8_up != a[0].q8_up || x.q8_inject != a[0].q8_inject) {
            std::fprintf(stderr, "fused_gr_read_multi: invalid arguments for token %d\n", t);
            std::exit(1);
        }
    }
    // S26 STRATA_QFUSE: every token needs its q8_1 destination and token 0 the counters; otherwise none is written
    bool q8 = a[0].q8_cnt != nullptr;
    for (int t = 0; t < n_tok; ++t) q8 = q8 && a[t].q8_mixed != nullptr;
    if (!q8)
        for (int t = 0; t < n_tok; ++t) m.a[t].q8_mixed = nullptr;
    m.xn = xn_scratch;
    m.T = n_tok;
    dpct::queue_ptr st = strata::q_of(stream);
    if (a[0].q8_down != nullptr && a[0].q8_up != nullptr) {   // inject: Q8_0 copy, or the BF16 rows
        launch_q8(m, xn_scratch, st, stamp_buf, stamp_i0);   // S23 experiment: STRATA_HC_Q8=1
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use
        the error codes. The cudaGetLastError function call was replaced with 0.
        You need to rewrite this code.
        */
        const dpct::err0 eq = 0;

        return q8;
    }
    // STRATA_GR_V3=1: the two-kernel read above (another summation order - opt-in)
    static const bool v3 = [] { const char* v = std::getenv("STRATA_GR_V3"); return v != nullptr && std::atoi(v) != 0; }();
    static int split3[64] = {};   // per device: 0 = not decided yet, 1 / 2 = column halves S, -1 = does not fit
    int dev3 = 0;
    if (v3) {
        dev3 = dpct::get_current_device_id();
        if (dev3 >= 0 && dev3 < 64 && split3[dev3] == 0) {
            int optin = 0;
            // SYCL port: the device's local memory is the limit (no per-kernel opt-in to raise)
            optin = (int) dpct::get_device(dev3).get_local_mem_size();
            const int limit = optin > 0 ? optin : 48 * 1024;
            const int need1 = (int) (kFusedGrMaxT * N * sizeof(float)), need6 = (int) (6 * N * sizeof(float)),
                      need5 = (int) (5 * N * sizeof(float)), need2 = need1 / 2;
            int split = -1;
            if (need1 <= limit &&
                /*
                DPCT1027: The call to cudaFuncSetAttribute was replaced
                with 0 because SYCL currently does not support corresponding
                setting.
                */
                0 == 0 &&
                /*
                DPCT1027: The call to cudaFuncSetAttribute was replaced
                with 0 because SYCL currently does not support corresponding
                setting.
                */
                0 == 0 &&
                /*
                DPCT1027: The call to cudaFuncSetAttribute was replaced
                with 0 because SYCL currently does not support corresponding
                setting.
                */
                0 == 0) {
                /*
                DPCT1026: The call to cudaFuncSetAttribute was removed
                because SYCL currently does not support corresponding setting.
                */
                /*
                DPCT1026: The call to cudaFuncSetAttribute was removed
                because SYCL currently does not support corresponding setting.
                */
                /*
                DPCT1026: The call to cudaFuncSetAttribute was removed
                because SYCL currently does not support corresponding setting.
                */
                /*
                DPCT1026: The call to cudaFuncSetAttribute was removed
                because SYCL currently does not support corresponding setting.
                */
                /*
                DPCT1026: The call to cudaFuncSetAttribute was removed
                because SYCL currently does not support corresponding setting.
                */
                /*
                DPCT1026: The call to cudaFuncSetAttribute was removed
                because SYCL currently does not support corresponding setting.
                */
                /*
                DPCT1026: The call to cudaFuncSetAttribute was removed
                because SYCL currently does not support corresponding setting.
                */
                /*
                DPCT1026: The call to cudaFuncSetAttribute was removed
                because SYCL currently does not support corresponding setting.
                */
                split = 1;
            } else if (need2 <= limit)
                // #375 (kenh0u): the S = 2 split (a 64 KB opt-in card: Turing) disagrees with itself in gr_parity
                // (graph replay vs direct call) - such a card keeps the default read until that split is fixed
                std::fprintf(stderr, "strata: STRATA_GR_V3=1 needs the two-half split on this card, which fails its "
                                     "checks (#375): the default read is used\n");
            /*
            DPCT1026: The call to cudaGetLastError was removed because this
            functionality is redundant in SYCL.
            */
; // drop any error the attempts left behind
            split3[dev3] = split;
        }
    }
    const int split = v3 && dev3 >= 0 && dev3 < 64 ? split3[dev3] : -1;
    if (split > 0) {   // 2 kernels; scratch = partials + sums of squares
        float* part = xn_scratch;
        float* ssg = xn_scratch + (size_t) n_tok * HC * 2 * PR;   // room for S = 2
        /*
        DPCT1083: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        const size_t sm = (size_t)n_tok * (N / split) * sizeof(float);
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);
        if (split == 2) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range(sm), cgh);

                /*
                DPCT1050: The template argument of the dpct_kernel_name
                could not be deduced. You need to update this code.
                */
                cgh.parallel_for<dpct_kernel_name<
                    class gr_down_v3_kernel_c14ff7, dpct_kernel_scalar<2>>>(
                    sycl::nd_range<3>(sycl::range(1, HC * 2, LR / WARPS + 1) *
                                          sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        gr_down_v3_kernel<2>(
                            m, part, ssg,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        } else if (n_tok == 1) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range(sm), cgh);

                cgh.parallel_for<dpct_kernel_name<
                    class gr_down_v3_kernel_39bd1c, dpct_kernel_scalar<1>,
                    dpct_kernel_scalar<1>, dpct_kernel_scalar<true>>>(
                    sycl::nd_range<3>(sycl::range(1, HC, LR / WARPS + 1) *
                                          sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        gr_down_v3_kernel<1, 1, true>(
                            m, part, ssg,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        } else if (n_tok == 2) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range(sm), cgh);

                cgh.parallel_for<dpct_kernel_name<
                    class gr_down_v3_kernel_bb24ba, dpct_kernel_scalar<1>,
                    dpct_kernel_scalar<2>, dpct_kernel_scalar<true>>>(
                    sycl::nd_range<3>(sycl::range(1, HC, LR / WARPS + 1) *
                                          sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        gr_down_v3_kernel<1, 2, true>(
                            m, part, ssg,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        } else if (n_tok == 3) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range(sm), cgh);

                cgh.parallel_for<dpct_kernel_name<
                    class gr_down_v3_kernel_6652bc, dpct_kernel_scalar<1>,
                    dpct_kernel_scalar<3>, dpct_kernel_scalar<true>>>(
                    sycl::nd_range<3>(sycl::range(1, HC, LR / WARPS + 1) *
                                          sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        gr_down_v3_kernel<1, 3, true>(
                            m, part, ssg,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        } else if (n_tok == 4) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range(sm), cgh);

                cgh.parallel_for<dpct_kernel_name<
                    class gr_down_v3_kernel_de08f5, dpct_kernel_scalar<1>,
                    dpct_kernel_scalar<4>, dpct_kernel_scalar<true>>>(
                    sycl::nd_range<3>(sycl::range(1, HC, LR / WARPS + 1) *
                                          sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        gr_down_v3_kernel<1, 4, true>(
                            m, part, ssg,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        } else if (n_tok == 5) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range(sm), cgh);

                cgh.parallel_for<dpct_kernel_name<
                    class gr_down_v3_kernel_c1973f, dpct_kernel_scalar<1>,
                    dpct_kernel_scalar<5>, dpct_kernel_scalar<true>>>(
                    sycl::nd_range<3>(sycl::range(1, HC, LR / WARPS + 1) *
                                          sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        gr_down_v3_kernel<1, 5, true>(
                            m, part, ssg,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        } else if (n_tok == 6) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range(sm), cgh);

                cgh.parallel_for<dpct_kernel_name<
                    class gr_down_v3_kernel_69049a, dpct_kernel_scalar<1>,
                    dpct_kernel_scalar<6>, dpct_kernel_scalar<true>>>(
                    sycl::nd_range<3>(sycl::range(1, HC, LR / WARPS + 1) *
                                          sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        gr_down_v3_kernel<1, 6, true>(
                            m, part, ssg,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        } else {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range(sm), cgh);

                cgh.parallel_for<dpct_kernel_name<
                    class gr_down_v3_kernel_88bc89, dpct_kernel_scalar<1>,
                    dpct_kernel_scalar<kFusedGrMaxT>,
                    dpct_kernel_scalar<false>>>(
                    sycl::nd_range<3>(sycl::range(1, HC, LR / WARPS + 1) *
                                          sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        gr_down_v3_kernel<1, kFusedGrMaxT, false>(
                            m, part, ssg,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        }
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
        if (split == 2) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<dpct_kernel_name<class gr_up_v3_kernel_6e945c,
                                              dpct_kernel_scalar<2>>>(
                    sycl::nd_range<3>(sycl::range(1, 1, UPM_BLOCKS) *
                                          sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            gr_up_v3_kernel<2>(m, part, ssg);
                        });
        } else if (n_tok == 1) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<dpct_kernel_name<
                class gr_up_v3_kernel_17ac52, dpct_kernel_scalar<1>,
                dpct_kernel_scalar<1>, dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, UPM_BLOCKS) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gr_up_v3_kernel<1, 1, true>(m, part, ssg);
                    });
        } else if (n_tok == 2) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<dpct_kernel_name<
                class gr_up_v3_kernel_834a79, dpct_kernel_scalar<1>,
                dpct_kernel_scalar<2>, dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, UPM_BLOCKS) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gr_up_v3_kernel<1, 2, true>(m, part, ssg);
                    });
        } else if (n_tok == 3) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<dpct_kernel_name<
                class gr_up_v3_kernel_9cf67e, dpct_kernel_scalar<1>,
                dpct_kernel_scalar<3>, dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, UPM_BLOCKS) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gr_up_v3_kernel<1, 3, true>(m, part, ssg);
                    });
        } else if (n_tok == 4) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<dpct_kernel_name<
                class gr_up_v3_kernel_51c807, dpct_kernel_scalar<1>,
                dpct_kernel_scalar<4>, dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, UPM_BLOCKS) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gr_up_v3_kernel<1, 4, true>(m, part, ssg);
                    });
        } else if (n_tok == 5) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<dpct_kernel_name<
                class gr_up_v3_kernel_f8fea5, dpct_kernel_scalar<1>,
                dpct_kernel_scalar<5>, dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, UPM_BLOCKS) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gr_up_v3_kernel<1, 5, true>(m, part, ssg);
                    });
        } else if (n_tok == 6) {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<dpct_kernel_name<
                class gr_up_v3_kernel_2dc77c, dpct_kernel_scalar<1>,
                dpct_kernel_scalar<6>, dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, UPM_BLOCKS) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gr_up_v3_kernel<1, 6, true>(m, part, ssg);
                    });
        } else {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<dpct_kernel_name<
                class gr_up_v3_kernel_716779, dpct_kernel_scalar<1>,
                dpct_kernel_scalar<kFusedGrMaxT>, dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(sycl::range(1, 1, UPM_BLOCKS) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gr_up_v3_kernel<1, kFusedGrMaxT, false>(m, part, ssg);
                    });
        }
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use
        the error codes. The cudaGetLastError function call was replaced with 0.
        You need to rewrite this code.
        */
        const dpct::err0 e3 = 0;

        return false;   // the v3 read: no q8_1 (the caller quantizes)
    }
#if defined(STRATA_HIP_GFX906)
    // gfx906, opt-in STRATA_GR_SPLIT=1: the latency-hidden norm/up (STRATA_GR_FAST) with the K-split down (~3% faster
    // per verify window).  Its fixed-order finish sums the K slices in another order than the single-token
    // fused_gr_read, so the multi-token read is then no longer bitwise equal to T single-token calls (gr_parity's
    // contract); off by default for that reason.
    static const bool split_on = std::getenv("STRATA_GR_SPLIT") && std::string(std::getenv("STRATA_GR_SPLIT")) == "1";
    if (split_on) {
        const bool fast = gr_fast();
        if (fast) gr_norm_fast_kernel<<<n_tok, THREADS, 0, st>>>(m);
        else gr_norm_multi_kernel<<<n_tok, THREADS, 0, st>>>(m);
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);
        gr_down_split_kernel<<<dim3(DOWN_BLOCKS + 1, GS_S), GS_WAVES * 64, (size_t) n_tok * GS_SL * sizeof(float), st>>>(
            m);
        gr_down_finish_kernel<<<(unsigned) ((n_tok * GS_ROWS + 255) / 256), 256, 0, st>>>(m);
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
        if (fast) gr_up_fast_kernel<<<UPM_BLOCKS, THREADS, 0, st>>>(m);
        else gr_up_multi_kernel<<<UPM_BLOCKS, THREADS, 0, st>>>(m);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "fused_gr_read_multi: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
        return;
    }
#endif
    m.part = down_partials(st);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        if (gr_norm_split())
            st->parallel_for<dpct_kernel_name<class gr_norm_split_k>>(
                sycl::nd_range<3>(sycl::range(1, 1, n_tok * HC) * sycl::range(1, 1, THREADS), sycl::range(1, 1, THREADS)),
                [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { gr_norm_split_port_kernel(m); });
        else
        st->parallel_for<dpct_kernel_name<class gr_norm_multi_kernel_2f4d92>>(
            sycl::nd_range<3>(sycl::range(1, 1, n_tok) *
                                  sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gr_norm_multi_kernel(m);
            });
    }
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);
    // the local-memory capacity is a per-DEVICE property: once per device (a layer split runs this on two cards).
    // SYCL port: upstream picks TILEV 1280 on sm_75 and takes the per-block limit below sm_70 (CUDA compute
    // capabilities, cudaFuncSetAttribute opt-ins); here the tile is the port's TILE and the device's local memory is
    // the only limit. The tiled kernel is the fallback (STRATA_GR_DOWN_SLICED=0 STRATA_GR_DOWN_DIRECT=0); the
    // default is the sliced kernel below.
    static bool attr[64] = {};
    static int chunk[64] = {};   // tokens the tiled down kernel may carry in one launch on this card
    int dev = 0;
    dev = dpct::get_current_device_id();
    if (dev >= 0 && dev < 64 && !attr[dev]) {
        const int optin = (int) dpct::get_device(dev).get_local_mem_size();
        const int capacity = (optin > 0 ? optin : 48 * 1024) / (int) (TILE * sizeof(float));
        chunk[dev] = capacity < 1 ? 1 : (capacity > kFusedGrMaxT ? kFusedGrMaxT : capacity);
        attr[dev] = true;
    }
    const int chunk_tok = (dev >= 0 && dev < 64 && chunk[dev]) ? chunk[dev] : kFusedGrMaxT;
    if (gr_down_sliced()) {
        m.part2 = slice_partials(st);
        m.nsplit = 1;
        st->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> tl(sycl::range<1>((size_t) kFusedGrMaxT * GRS_COLS), cgh);
            cgh.parallel_for<dpct_kernel_name<class gr_down_sliced_k>>(
                sycl::nd_range<3>(sycl::range(1, 1, GRS_SLICES) * sycl::range(1, 1, THREADS), sycl::range(1, 1, THREADS)),
                [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                    gr_down_sliced_kernel(m, tl.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
        const unsigned nred = unsigned((n_tok * GRS_ROWS + 127) / 128);
        st->parallel_for<dpct_kernel_name<class gr_down_reduce_k>>(
            sycl::nd_range<3>(sycl::range(1, 1, nred) * sycl::range(1, 1, 128), sycl::range(1, 1, 128)),
            [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { gr_down_reduce_kernel(m); });
    } else if (gr_down_direct()) {
        st->parallel_for<dpct_kernel_name<class gr_down_multi_direct>>(
            sycl::nd_range<3>(sycl::range(1, 1, DOWN_BLOCKS * DOWN_SPLIT + 1) * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { gr_down_multi_direct_kernel(m); });
    } else if (chunk_tok >= n_tok) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        st->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range((size_t)n_tok * TILE * sizeof(float)), cgh);
            cgh.parallel_for<
                dpct_kernel_name<class gr_down_multi_kernel_7f5820, dpct_kernel_scalar<TILE>>>(
                sycl::nd_range<3>(sycl::range(1, 1, DOWN_BLOCKS * DOWN_SPLIT + 1) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gr_down_multi_kernel<TILE>(
                            m, dpct_local_acc_ct1
                                   .get_multi_ptr<sycl::access::decorated::no>()
                                   .get());
                    });
        });
    } else {
        for (int c0 = 0; c0 < n_tok; c0 += chunk_tok) {
            const int ct = n_tok - c0 < chunk_tok ? n_tok - c0 : chunk_tok;
            GrMulti c{};
            c.xn = xn_scratch + (size_t) c0 * D;
            c.T = ct;
            for (int k = 0; k < ct; ++k) c.a[k] = a[c0 + k];
            c.part = m.part;
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            st->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range((size_t)ct * TILE * sizeof(float)), cgh);
                cgh.parallel_for<
                    dpct_kernel_name<class gr_down_multi_kernel_88cb86, dpct_kernel_scalar<TILE>>>(
                    sycl::nd_range<3>(sycl::range(1, 1, DOWN_BLOCKS * DOWN_SPLIT + 1) *
                                          sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            gr_down_multi_kernel<TILE>(
                                c, dpct_local_acc_ct1
                                       .get_multi_ptr<sycl::access::decorated::no>()
                                       .get());
                        });
            });
        }
    }
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class gr_up_multi_kernel_1a727f>>(
            sycl::nd_range<3>(sycl::range(1, 1, UPM_BLOCKS) *
                                  sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gr_up_multi_kernel(m);
            });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;

    return q8;
}

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    return n_embd == N && hc == HC && hc_lr == LR;
}

void fused_gr_read(const FusedGrArgs& a, void* stream) {
    if (!a.R || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs || !a.mixed ||
        (a.w_inject && !a.inject_out) || (a.apply && (!a.bo_prev || !a.inj_prev || !a.R_out)) ||
        (a.apply && a.inj_prev == a.inject_out)) {
        std::fprintf(stderr, "fused_gr_read: invalid arguments\n");
        std::exit(1);
    }
    dpct::queue_ptr st = strata::q_of(stream);
    // SYCL port: the multi-token read's sliced down kernel and per-(token, stream) norm sum in their own orders, so a
    // window's token would differ in its last bits from this kernel's. The header promises "every token's outputs
    // are bitwise fused_gr_read(a[t])" (gr_parity checks it): with those paths on, the single read IS the multi read
    // of one token (each token's sums run in the same order whatever the window size).
    if (gr_down_sliced() || gr_norm_split() || gr_down_direct()) {
        static std::mutex mu;
        static std::unordered_map<sycl::queue*, float*> xn;
        float* scratch;
        {
            std::lock_guard<std::mutex> lk(mu);
            float*& p = xn[st];
            if (!p) p = sycl::malloc_device<float>((size_t) D, *st);
            scratch = p;
        }
        fused_gr_read_multi(&a, 1, scratch, stream);
        return;
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class gr_down_kernel_a5fa71>>(
            sycl::nd_range<3>(sycl::range(1, 1, DOWN_BLOCKS + 1) *
                                  sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gr_down_kernel(a);
            });
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class gr_up_kernel_fe3258>>(
            sycl::nd_range<3>(sycl::range(1, 1, UP_BLOCKS) *
                                  sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                gr_up_kernel(a);
            });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
}


namespace {

/// The plain read, split, staged and (for one token) the single-token read on random bf16 weights and random inputs,
/// 1..8 tokens, with and without the pending write; every output of split and staged compared with the plain read's
/// bit for bit (and the plain read's with the single-token read's).  `why[v]` gets the first difference of variant
/// v; false if the check itself could not run.
bool fused_gr_selftest(bool ok_variant[4], std::string why[4]) try {
    constexpr int TM = kFusedGrMaxT;
    constexpr int NV = 4;                             // sets: 0 = plain, 1 = split, 2 = staged, 3 = single-token
    std::mt19937 rng(20260930u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> h_down((size_t) LR * D), h_up((size_t) D * LR), h_inj((size_t) HC * D);
    std::vector<float> h_norm(D), h_R((size_t) TM * D), h_bo((size_t) TM * N), h_ip((size_t) TM * HC);
    for (auto& w : h_down) w = bf16_from_f32(0.02f * nd(rng));
    for (auto& w : h_up) w = bf16_from_f32(0.05f * nd(rng));
    for (auto& w : h_inj) w = bf16_from_f32(0.02f * nd(rng));
    for (auto& w : h_norm) w = 1.0f + 0.1f * nd(rng);
    for (auto& x : h_R) x = nd(rng);
    for (auto& x : h_bo) x = 0.5f * nd(rng);
    for (auto& x : h_ip) x = 2.0f * nd(rng);
    for (int v = 0; v < 4; ++v) { ok_variant[v] = v == 1; why[v].clear(); }

    const size_t n_set = (size_t) TM * (D + D + LR + HC + HC + N);   // R_out, xn, lo, rs, inject, mixed (floats)
    const size_t bytes = h_down.size() * 2 + h_up.size() * 2 + h_inj.size() * 2 +
                         (h_norm.size() + h_R.size() + h_bo.size() + h_ip.size() + NV * n_set) * 4 + 32 * 256;
    uint8_t* base = nullptr;
    if (DPCT_CHECK_ERROR(base = (uint8_t *)sycl::malloc_device(
                             bytes, dpct::get_in_order_queue())) != 0) {
        /*
        DPCT1026: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        why[0] =
            "no room for the check (" + std::to_string(bytes >> 20) + " MiB)";
        return false;
    }
    size_t off = 0;
    auto take = [&](size_t b) { void* p = base + off; off += (b + 255) / 256 * 256; return p; };
    uint16_t* d_down = (uint16_t*) take(h_down.size() * 2);
    uint16_t* d_up = (uint16_t*) take(h_up.size() * 2);
    uint16_t* d_inj = (uint16_t*) take(h_inj.size() * 2);
    float* d_norm = (float*) take(h_norm.size() * 4);
    float* d_R = (float*) take(h_R.size() * 4);
    float* d_bo = (float*) take(h_bo.size() * 4);
    float* d_ip = (float*) take(h_ip.size() * 4);
    struct Set { float *R_out, *xn, *lo, *rs, *inj, *mixed; } set[NV];
    for (Set& x : set) {
        x.R_out = (float*) take((size_t) TM * D * 4); x.xn = (float*) take((size_t) TM * D * 4);
        x.lo = (float*) take((size_t) TM * LR * 4); x.rs = (float*) take((size_t) TM * HC * 4);
        x.inj = (float*) take((size_t) TM * HC * 4); x.mixed = (float*) take((size_t) TM * N * 4);
    }
    dpct::queue_ptr st = &dpct::get_in_order_queue();
    /*
    DPCT1025: The SYCL queue is created ignoring the flag and priority
    options.
    */
    bool ok =
        off <= bytes &&
        DPCT_CHECK_ERROR(st = dpct::get_current_device().create_queue(true)) ==
            0 &&
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        DPCT_CHECK_ERROR(
            st->memcpy(d_down, h_down.data(), h_down.size() * 2)) == 0 &&
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        DPCT_CHECK_ERROR(st->memcpy(d_up, h_up.data(), h_up.size() * 2)) == 0 &&
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        DPCT_CHECK_ERROR(st->memcpy(d_inj, h_inj.data(), h_inj.size() * 2)) ==
            0 &&
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        DPCT_CHECK_ERROR(
            st->memcpy(d_norm, h_norm.data(), h_norm.size() * 4)) == 0 &&
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        DPCT_CHECK_ERROR(st->memcpy(d_R, h_R.data(), h_R.size() * 4)) == 0 &&
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        DPCT_CHECK_ERROR(st->memcpy(d_bo, h_bo.data(), h_bo.size() * 4)) == 0 &&
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        DPCT_CHECK_ERROR(st->memcpy(d_ip, h_ip.data(), h_ip.size() * 4)) == 0;
    if (!ok) why[0] = "setting up the check failed";
    std::vector<float> h1, h2;
    // true when equal; false with the first difference in `w` (or a read-back failure in `ok`)
    auto same = [&](const float* d1, const float* d2, size_t n, const char* what, int T, int apply, std::string& w) {
        h1.resize(n);
        h2.resize(n);
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        if (DPCT_CHECK_ERROR(st->memcpy(h1.data(), d1, n * 4)) != 0 ||
            /*
            DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            DPCT_CHECK_ERROR(st->memcpy(h2.data(), d2, n * 4)) != 0 ||
            DPCT_CHECK_ERROR(st->wait()) != 0) {
            why[0] = "reading back the check failed";
            ok = false;
            return false;
        }
        for (size_t i = 0; i < n; ++i) {
            uint32_t u1, u2;
            std::memcpy(&u1, &h1[i], 4);
            std::memcpy(&u2, &h2[i], 4);
            if (u1 != u2) {
                char buf[192];
                std::snprintf(buf, sizeof buf, "%s differs for %d token(s)%s at %zu (%08x, not %08x)", what, T,
                              apply ? " with the pending write" : "", i, u2, u1);
                w = buf;
                return false;
            }
        }
        return true;
    };
    auto all_same = [&](const Set& s1, const Set& s2, int T, int apply, std::string& w) {
        return same(s1.lo, s2.lo, (size_t) T * LR, "lo", T, apply, w) &&
               same(s1.rs, s2.rs, (size_t) T * HC, "rs", T, apply, w) &&
               same(s1.inj, s2.inj, (size_t) T * HC, "inject", T, apply, w) &&
               same(s1.mixed, s2.mixed, (size_t) T * N, "mixed", T, apply, w) &&
               same(s1.R_out, s2.R_out, (size_t) T * D, "R", T, apply, w);
    };
    ok_variant[2] = ok_variant[3] = ok;
    bool single_ok = ok;
    for (int apply = 0; apply < 2 && ok; ++apply) {
        for (int T = 1; T <= TM && ok; ++T) {
            FusedGrArgs a[NV][TM];
            for (int v = 0; v < NV; ++v) {
                const size_t span = (size_t) ((uint8_t*) (set[v].mixed + (size_t) TM * N) - (uint8_t*) set[v].R_out);
                ok = ok &&
                     DPCT_CHECK_ERROR(st->memset(set[v].R_out, 0xFF, span)) ==
                         0; // the whole set
                for (int k = 0; k < T; ++k) {
                    FusedGrArgs& x = a[v][k];
                    x.R = d_R + (size_t) k * D; x.R_out = set[v].R_out + (size_t) k * D; x.apply = apply != 0;
                    x.bo_prev = d_bo + (size_t) k * N; x.inj_prev = d_ip + (size_t) k * HC;
                    x.w_norm = d_norm; x.w_down = d_down; x.w_up = d_up; x.w_inject = d_inj; x.eps = 1e-6f;
                    x.lo = set[v].lo + (size_t) k * LR; x.rs = set[v].rs + (size_t) k * HC;
                    x.inject_out = set[v].inj + (size_t) k * HC; x.mixed = set[v].mixed + (size_t) k * N;
                }
            }
            for (int v = 0; v < 3; ++v) {
                GrMulti m;
                for (int k = 0; k < T; ++k) m.a[k] = a[v][k];
                m.xn = set[v].xn;
                m.T = T;
                launch_multi(m, v + 1, st, nullptr, 0);
            }
            if (T == 1) fused_gr_read(a[3][0], st);
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not
            use the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            if (0 != 0 || DPCT_CHECK_ERROR(st->wait()) != 0) {
                why[0] = "a kernel of the check failed";
                ok = false;
                break;
            }
            if (ok_variant[2] && !all_same(set[0], set[1], T, apply, why[2])) ok_variant[2] = false;
            if (ok && ok_variant[3] && !all_same(set[0], set[2], T, apply, why[3])) ok_variant[3] = false;
            if (ok && T == 1 && single_ok && !all_same(set[3], set[0], T, apply, why[1])) single_ok = false;
        }
    }
    if (!single_ok && ok) {                            // the plain read itself disagrees with the single-token read
        why[2] = why[3] = "the plain read differs from the single-token read: " + why[1];
        ok_variant[2] = ok_variant[3] = false;
    }
    if (st != &dpct::get_in_order_queue()) {
        st->wait();
        dpct::get_current_device().destroy_queue(st);
    }
    sycl::free(base, dpct::get_in_order_queue());
    /*
    DPCT1026: The call to cudaGetLastError was removed because this
    functionality is redundant in SYCL.
    */
    if (!ok) ok_variant[2] = ok_variant[3] = false;
    return ok;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace

int fused_gr_variant() {
    int dev = 0;
    dev = dpct::get_current_device_id();
    const int v = dev >= 0 && dev < 64 ? g_variant[dev].load() : 0;
    if (v > 0) return v;
    // not checked on this card: the plain read, unless STRATA_HC_SPLIT names a variant (a test such as gr_parity)
    const char* e = std::getenv("STRATA_HC_SPLIT");
    return e != nullptr && (e[0] == '1' || e[0] == '2') ? env_variant() : kHcPlain;
}

void fused_gr_check() {
    int dev = 0;
    dev = dpct::get_current_device_id();
    if (dev < 0 || dev >= 64 || g_variant[dev].load() > 0) return;
    // SYCL port: the port's own read (sliced down / split norm, STRATA_GR_DOWN_SLICED etc.) does not use upstream's
    // variants, and their self-test crashes on the B70 (0.1.32 merge, under investigation): opt-in only.
    if (std::getenv("STRATA_HC_CHECK") == nullptr) {
        g_variant[dev].store(kHcPlain);
        return;
    }
    const int want = env_variant();
    if (want == kHcPlain) {
        g_variant[dev].store(kHcPlain);
        std::fprintf(stderr, "strata hc: CUDA%d: the hyper-connection read runs as the plain one (STRATA_HC_SPLIT=0)\n",
                     dev);
        return;
    }
    bool okv[4];
    std::string why[4];
    const bool ran = fused_gr_selftest(okv, why);
    int use = kHcPlain;
    if (want >= kHcStaged && okv[kHcStaged]) use = kHcStaged;
    else if (okv[kHcSplit]) use = kHcSplit;
    g_variant[dev].store(use);
    if (!ran)
        std::fprintf(stderr, "strata hc: CUDA%d: the check of split/staged could not run (%s)\n", dev, why[0].c_str());
    static const char* const name[4] = {"", "plain", "split", "staged"};
    for (int v = kHcStaged; v >= kHcSplit; --v)
        if (v <= want && !okv[v] && ran)
            std::fprintf(stderr, "strata hc: CUDA%d: the %s read differs from the plain read on this card - not used: "
                                 "%s\n", dev, name[v], why[v].c_str());
    static const char* const what[4] = {
        "", "the plain read (the norm per token, the down projection on 41 blocks)",
        "split (the norm per token and stream, then the plain read's down projection)",
        "staged (the norm per token and stream, the down projection's activations staged ahead by cp.async)"};
    std::fprintf(stderr, "strata hc: CUDA%d: the hyper-connection read runs as %s%s\n", dev, what[use],
                 use >= kHcSplit ? "; checked bit for bit against the plain read on this card (STRATA_HC_SPLIT=1 or 0 "
                                   "for the earlier ones)" : "");
}


}  // namespace strata::kernels
