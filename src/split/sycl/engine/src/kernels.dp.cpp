// fnx kernels. Quantized dot products come from ggml (vecdotq.cuh), the rest follows llama.cpp's qwen4exp graph
// (src/models/qwen4exp.cpp) operation by operation.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "kernels.hpp"
#include "native_qsa.hpp"
#include "strata/kprof.hpp"
#include "strata/kernels/qsa_prompt_attn_esimd.hpp"
#include "env.hpp"

   // HIP: src/hip/mma.h (gfx12 WMMA)

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if FNX_HIP
// the tree's hip_compat/intrinsics.hpp names these as macros; ggml's HIP vendor header defines its own: it goes first, then ours return
#undef __vsubss4
#undef __vsub4
#undef __vcmpne4
#undef __shfl_sync
#undef __shfl_up_sync
#undef __shfl_xor_sync
#endif
#include "common.dp.hpp"
#include "vecdotq.dp.hpp"
#if FNX_HIP
#undef __shfl_sync
#undef __shfl_up_sync
#undef __shfl_xor_sync
#define __shfl_xor_sync(...) (::strata::hip_compat::shfl_xor_sync(__VA_ARGS__))
#define __shfl_up_sync(...) (::strata::hip_compat::shfl_up_sync(__VA_ARGS__))
#define __shfl_sync(...) (::strata::hip_compat::shfl_sync(__VA_ARGS__))
#undef cudaFuncSetAttribute   // ggml's maps it to hipFuncSetAttribute(const void*, ...); the tree's template takes the kernel
#endif
#include "iqdot_hip.dp.hpp" // HIP: the i-quant dots with a cheaper sign expansion (same integers as ggml's)

// compiled in parts (Makefile: build/kernels_pN.o with -DFNX_KPART=N), each with its share of the launchers below and
// so only the template kernels they instantiate: one file's ~700 kernels took ~7 minutes on one core
#ifndef FNX_KPART
#error "kernels.cu: compile with -DFNX_KPART=0..8 (one object per part)"
#endif

namespace fnx {

namespace {

void ck_launch(const char* what) {
    
    const dpct::err0 e = 0;
    
    
    
    if (e != 0) throw std::runtime_error(std::string(what) + ": " +
                                         dpct::get_error_string_dummy(e));
}

 __dpct_inline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), v, o);
    return v;
}
 __dpct_inline__ float wmax(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v = sycl::fmax(
        v, dpct::permute_sub_group_by_xor(
               sycl::ext::oneapi::this_work_item::get_sub_group(), v, o));
    return v;
}
 __dpct_inline__ float sigm(float x) {
    return 1.0f / (1.0f + sycl::native::exp(-x));
}
 __dpct_inline__ float silu_(float x) {
    return x / (1.0f + sycl::native::exp(-x));
}
 __dpct_inline__ float bf(uint16_t h) {
    return sycl::bit_cast<float>((unsigned)h << 16);
}
 __dpct_inline__ float f16(uint16_t h) {
    return sycl::vec<sycl::half, 1>(
               sycl::bit_cast<sycl::half, unsigned short>(h))
        .convert<float, sycl::rounding_mode::automatic>()[0];
}

// block-wide sum for blockDim.x <= 1024; every thread gets the result
 float block_sum(float v, float *sh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    v = wsum(v);
    const int w = item_ct1.get_local_id(2) >> 5,
              l = item_ct1.get_local_id(2) & 31,
              nw = (item_ct1.get_local_range(2) + 31) >> 5;
    
    item_ct1.barrier();
    if (l == 0) sh[w] = v;
    
    item_ct1.barrier();
    float s = 0.f;
    for (int i = 0; i < nw; ++i) s += sh[i];
    return s;
}
float block_max(float v, float* sh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    v = wmax(v);
    const int w = item_ct1.get_local_id(2) >> 5,
              l = item_ct1.get_local_id(2) & 31,
              nw = (item_ct1.get_local_range(2) + 31) >> 5;
    
    item_ct1.barrier();
    if (l == 0) sh[w] = v;
    
    item_ct1.barrier();
    float s = -INFINITY;
    for (int i = 0; i < nw; ++i) s = sycl::fmax(s, sh[i]);
    return s;
}

// ---------------------------------------------------------------- q8_1 quantization (ggml's rounding)
}  // namespace
// the split twins of q8_1 activation buffers: one registry for the engine (part 0 holds it)
struct Q8Twin {
    const char* q8;
    size_t bytes;
    int8_t* tq;
    float* td;
};
std::vector<Q8Twin>& q8_twins();
namespace {
// the twin of the q8_1 rows starting at y (rows of n columns), or nulls
void twin_of(const void* y, int n, int8_t*& tq, float*& td) {
    tq = nullptr;
    td = nullptr;
    for (const Q8Twin& w : q8_twins())
        if ((const char*) y >= w.q8 && (const char*) y < w.q8 + w.bytes) {
            const size_t off = (size_t) ((const char*) y - w.q8), rb = (size_t) (n / 32) * 36;
            if (off % rb) throw std::runtime_error("q8 twin: a q8_1 pointer between rows");
            const size_t r = off / rb;
            tq = w.tq + r * n;
            td = w.td + r * (n / 32);
            return;
        }
}

void k_quantize(const float* x, block_q8_1* q, int n, int8_t* tq, float* td) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(1),
              b = item_ct1.get_group(2) * (item_ct1.get_local_range(2) / 32) +
                  (item_ct1.get_local_id(2) >> 5),
              lane = item_ct1.get_local_id(2) & 31;
    if (b >= n / 32) return;
    const float xi = x[(size_t) row * n + b * 32 + lane];
    const float amax = wmax(sycl::fabs(xi)), sum = wsum(xi);
    const float d = amax / 127.0f;
    const int8_t qv = amax == 0.0f ? 0 : (int8_t)sycl::round(xi / d);
    block_q8_1& y = q[(size_t) row * (n / 32) + b];
    y.qs[lane] = qv;
    if (lane == 0) y.ds = sycl::half2(
        sycl::vec<float, 1>(d)
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0],
        sycl::vec<float, 1>(sum)
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0]);
    if (tq) {
        tq[(size_t) row * n + b * 32 + lane] = qv;
        if (lane == 0) td[(size_t)row * (n / 32) + b] =
            sycl::vec<sycl::half, 1>(
                sycl::vec<float, 1>(d)
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0])
                .convert<float, sycl::rounding_mode::automatic>()[0];
    }
}

// ---------------------------------------------------------------- jobs of quantized matrix x vectors
template <ggml_type type>
 __dpct_inline__ float
vdot(const void *v, const block_q8_1 *y, int kbx, int iqs,
     const uint64_t *iq2s_grid, const uint32_t *iq3xxs_grid,
     const uint32_t *iq3s_grid, const int8_t *kvalues_iq4nl) {
    if constexpr (type == GGML_TYPE_Q8_0) return vec_dot_q8_0_q8_1(v, y, kbx, iqs);
    else if constexpr (type == GGML_TYPE_Q4_K) return vec_dot_q4_K_q8_1(v, y, kbx, iqs);
    else if constexpr (type == GGML_TYPE_Q5_K) return vec_dot_q5_K_q8_1(v, y, kbx, iqs);
    else if constexpr (type == GGML_TYPE_Q5_1) return vec_dot_q5_1_q8_1(v, y, kbx, iqs);
    else if constexpr (type == GGML_TYPE_Q4_0) return vec_dot_q4_0_q8_1(v, y, kbx, iqs);
    else if constexpr (type == GGML_TYPE_Q6_K) return vec_dot_q6_K_q8_1(v, y, kbx, iqs);
    else if constexpr (type == GGML_TYPE_IQ4_NL) return vec_dot_iq4_nl_q8_1(
        v, y, kbx, iqs, kvalues_iq4nl);
    else if constexpr (type == GGML_TYPE_IQ4_XS) return vec_dot_iq4_xs_q8_1(
        v, y, kbx, iqs, kvalues_iq4nl);
#if FNX_HIP
    else if constexpr (type == GGML_TYPE_IQ3_S) return fnx_vec_dot_iq3_s_q8_1(v, y, kbx, iqs);
#else
    else if constexpr (type == GGML_TYPE_IQ3_S) return vec_dot_iq3_s_q8_1(
        v, y, kbx, iqs, iq3s_grid);
#endif
#if FNX_HIP
    else if constexpr (type == GGML_TYPE_IQ3_XXS) return fnx_vec_dot_iq3_xxs_q8_1(v, y, kbx, iqs);
#else
    else if constexpr (type == GGML_TYPE_IQ3_XXS) return vec_dot_iq3_xxs_q8_1(
        v, y, kbx, iqs, iq3xxs_grid);
#endif
#if FNX_HIP
    else if constexpr (type == GGML_TYPE_IQ2_S) return fnx_vec_dot_iq2_s_q8_1(v, y, kbx, iqs);
#else
    else if constexpr (type == GGML_TYPE_IQ2_S) return vec_dot_iq2_s_q8_1(
        v, y, kbx, iqs, iq2s_grid);
#endif
    else return 0.0f;
}
template <ggml_type type>  constexpr int vdr_of() {
    if constexpr (type == GGML_TYPE_Q8_0) return VDR_Q8_0_Q8_1_MMVQ;
    else if constexpr (type == GGML_TYPE_Q4_K) return VDR_Q4_K_Q8_1_MMVQ;
    else if constexpr (type == GGML_TYPE_Q5_K) return VDR_Q5_K_Q8_1_MMVQ;
    else if constexpr (type == GGML_TYPE_Q5_1) return VDR_Q5_1_Q8_1_MMVQ;
    else if constexpr (type == GGML_TYPE_Q4_0) return VDR_Q4_0_Q8_1_MMVQ;
    else if constexpr (type == GGML_TYPE_Q6_K) return VDR_Q6_K_Q8_1_MMVQ;
    else if constexpr (type == GGML_TYPE_IQ4_NL) return VDR_IQ4_NL_Q8_1_MMVQ;
    else if constexpr (type == GGML_TYPE_IQ4_XS) return VDR_IQ4_XS_Q8_1_MMVQ;
    else if constexpr (type == GGML_TYPE_IQ3_S) return VDR_IQ3_S_Q8_1_MMVQ;
    else if constexpr (type == GGML_TYPE_IQ3_XXS) return VDR_IQ3_XXS_Q8_1_MMVQ;
    else return VDR_IQ2_S_Q8_1_MMVQ;
}

constexpr int MMVQ_THREADS = 128;

// G lanes per weight row; a block covers MMVQ_THREADS / G rows of one job
template <ggml_type type, int G>

void k_mmvq(const char *__restrict__ wbase, size_t w_stride, size_t row_bytes,
            int nrows, int ncols, const Job *__restrict__ jobs,
            const int *__restrict__ njobs, int njobs_max,
            const block_q8_1 *__restrict__ y, int y_stride,
            float *__restrict__ out, int out_stride, uint64_t *iq2s_grid,
            uint32_t *iq3xxs_grid, uint32_t *iq3s_grid, int8_t *kvalues_iq4nl,
            Job &jb) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int ji = item_ct1.get_group(1);
    if (ji >= (njobs ? *njobs : njobs_max)) return;

    if (item_ct1.get_local_id(2) == 0) jb = jobs[ji];
    
    item_ct1.barrier();
    constexpr int qk = ggml_cuda_type_traits<type>::qk, qi = ggml_cuda_type_traits<type>::qi, vdr = vdr_of<type>();
    constexpr int per_block = qi / vdr;                  // lanes that share one weight block
    static_assert(G >= per_block && G % per_block == 0, "G must cover a block");
    constexpr int bpi = G / per_block;                   // weight blocks a group advances per step
    const int gl = item_ct1.get_local_id(2) % G,
              row = item_ct1.get_group(2) * (MMVQ_THREADS / G) +
                    item_ct1.get_local_id(2) / G;
    const int bpr = ncols / qk, n = jb.n;
    const bool live = row < nrows;
    const char* wr = wbase + (size_t) jb.w * w_stride + (size_t) (live ? row : 0) * row_bytes;
    float tmp[MAXT] = {};
    if (live)
        for (int kbx = gl / per_block; kbx < bpr; kbx += bpi) {
            const int kby = kbx * (qk / QK8_1), kqs = vdr * (gl % per_block);
#pragma unroll
            for (int j = 0; j < MAXT; ++j) {
                if (j >= n) break;
                tmp[j] += vdot<type>(wr, y + (size_t)jb.y[j] * y_stride + kby,
                                     kbx, kqs, iq2s_grid, iq3xxs_grid,
                                     iq3s_grid, kvalues_iq4nl);
            }
        }
#pragma unroll
    for (int j = 0; j < MAXT; ++j) {
        if (j >= n) break;
        float v = tmp[j];
#pragma unroll
        for (int o = G / 2; o; o >>= 1) v += dpct::permute_sub_group_by_xor(
            sycl::ext::oneapi::this_work_item::get_sub_group(), v, o);
        if (live && gl == 0) out[(size_t) jb.out[j] * out_stride + row] = v;
    }
}

#include "dense.dp.hpp"
#include "moe.dp.hpp"
#include "moe_q4k.dp.hpp"

// ---------------------------------------------------------------- f32 matrix x vectors (router, alpha / beta, gates)
// y[t][row] = W[row] . x[t]: a warp per row (WPR = 1) or a block per row (WPR = 4 warps, the columns split), n = N
template <int T, int WPR>

void k_gemv_f32(const float *__restrict__ W, const float *__restrict__ x,
                float *__restrict__ y, int rows, float red[T][WPR]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int NC = N / 4 / (32 * WPR); // float4 chunks per lane
    static_assert(N % (4 * 32 * WPR) == 0, "chunks per lane");
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int row =
        WPR == 1 ? item_ct1.get_group(2) * 8 + warp : item_ct1.get_group(2);
    const int part = WPR == 1 ? 0 : warp;     // this warp's share of the columns
    if (row >= rows) return;
    const sycl::float4 *w4 =
        reinterpret_cast<const sycl::float4 *>(W + (size_t)row * N) +
        part * NC * 32;
    sycl::float4 wv[NC];
#pragma unroll
    
    for (int i = 0; i < NC; ++i) wv[i] = *(w4 + i * 32 + lane);
    float s[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const sycl::float4 *x4 =
            reinterpret_cast<const sycl::float4 *>(x + (size_t)t * N) +
            part * NC * 32;
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < NC; ++i) {
            const sycl::float4 b = x4[i * 32 + lane];
            a += wv[i].x() * b.x() + wv[i].y() * b.y() + wv[i].z() * b.z() +
                 wv[i].w() * b.w();
        }
        s[t] = wsum(a);
    }
    if (WPR == 1) {
        if (lane < T)
#pragma unroll
            for (int t = 0; t < T; ++t)
                if (t == lane) y[(size_t) t * rows + row] = s[t];
        return;
    }

    if (lane == 0)
#pragma unroll
        for (int t = 0; t < T; ++t) red[t][warp] = s[t];
    
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) < T) {
        float v = 0.f;
#pragma unroll
        for (int k = 0; k < WPR; ++k) v += red[item_ct1.get_local_id(2)][k];
        y[(size_t)item_ct1.get_local_id(2) * rows + row] = v;
    }
}

// k_gemv_f32<T, 4>'s block for `row`, in a block of Q8S_THREADS (warps past the four take no columns), the same sums
template <int T>

__dpct_inline__ void
gemv4_block(const float *__restrict__ W, const float *__restrict__ x,
            float *__restrict__ y, int rows, int row, float red[T][4 /*WPR*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int WPR = 4, NC = N / 4 / (32 * WPR);
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;

    if (warp < WPR) {
        const sycl::float4 *w4 =
            reinterpret_cast<const sycl::float4 *>(W + (size_t)row * N) +
            warp * NC * 32;
        sycl::float4 wv[NC];
#pragma unroll
        
        for (int i = 0; i < NC; ++i) wv[i] = *(w4 + i * 32 + lane);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const sycl::float4 *x4 =
                reinterpret_cast<const sycl::float4 *>(x + (size_t)t * N) +
                warp * NC * 32;
            float a = 0.f;
#pragma unroll
            for (int i = 0; i < NC; ++i) {
                const sycl::float4 b = x4[i * 32 + lane];
                a += wv[i].x() * b.x() + wv[i].y() * b.y() + wv[i].z() * b.z() +
                     wv[i].w() * b.w();
            }
            a = wsum(a);
            if (lane == 0) red[t][warp] = a;
        }
    }
    
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) < T) {
        float v = 0.f;
#pragma unroll
        for (int k = 0; k < WPR; ++k) v += red[item_ct1.get_local_id(2)][k];
        y[(size_t)item_ct1.get_local_id(2) * rows + row] = v;
    }
}

// a Q8_0 matrix (q8s) and a small f32 one (gemv_f32's block-per-row case) in one launch instead of two streams: the
// f32 rows' blocks first (short; they start at once), then the q8s blocks
template <int T, int R, bool SPLIT>
void k_q8s_ab(const Q8Segs g, int rows, int cols, const block_q8_1* __restrict__ y,
                                                       const int8_t* __restrict__ tq, const float* __restrict__ td, const float* __restrict__ W,
                                                       const float* __restrict__ x, float* __restrict__ yf, int rf,
                                                       float red[T][4/*WPR*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if ((int)item_ct1.get_group(2) < rf)
        gemv4_block<T>(W, x, yf, rf, item_ct1.get_group(2), red);
    else q8s_block<T, R, SPLIT, false>(g, rows, cols, y, tq, td,
                                       item_ct1.get_group(2) - rf);
}

// ---------------------------------------------------------------- hyper-connection read
#include "hc.dp.hpp"

// ---------------------------------------------------------------- MoE routing, swiglu, combine
void k_topk(const float* logits, int* topi, float* topw, float *p, float *sh,
            int *si) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int t = item_ct1.get_group(2), e = item_ct1.get_local_id(2);
    const float l = logits[(size_t) t * NEXP + e];
    const float m = block_max(l, sh);
    const float ex = sycl::native::exp(l - m);
    const float z = block_sum(ex, sh);
    p[e] = ex / z;
    
    item_ct1.barrier();
    float sel_sum = 0.f;
    for (int k = 0; k < TOPK; ++k) {
        // argmax of p over the block (ties: the lower index)
        float v = p[e];
        int idx = e;
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float ov = dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), v, o);
            const int oi = dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), idx, o);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        
        
        item_ct1.barrier();
        if ((e & 31) == 0) { sh[e >> 5] = v; si[e >> 5] = idx; }
        
        
        item_ct1.barrier();
        if (e == 0) {
            float bv = sh[0];
            int bi = si[0];
            for (int w = 1; w < NEXP / 32; ++w)
                if (sh[w] > bv || (sh[w] == bv && si[w] < bi)) { bv = sh[w]; bi = si[w]; }
            topi[t * TOPK + k] = bi;
            topw[t * TOPK + k] = bv;
            p[bi] = -1.0f;
        }
        
        
        item_ct1.barrier();
    }
    if (e == 0) {
        for (int k = 0; k < TOPK; ++k) sel_sum += topw[t * TOPK + k];
        for (int k = 0; k < TOPK; ++k) topw[t * TOPK + k] /= sel_sum;
    }
}

// one thread: distinct experts in first-seen order, each with the tokens (and pairs) that picked it
void k_jobs(const int* topi, int T, Job* gu, Job* dn, int* njobs) {
    int nj = 0;
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < TOPK; ++k) {
            const int e = topi[t * TOPK + k], pair = t * TOPK + k;
            int j = 0;
            while (j < nj && gu[j].w != e) ++j;
            if (j == nj) { gu[j].w = e; gu[j].n = 0; dn[j].w = e; dn[j].n = 0; ++nj; }
            gu[j].y[gu[j].n] = t;
            gu[j].out[gu[j].n] = pair;
            ++gu[j].n;
            dn[j].y[dn[j].n] = pair;
            dn[j].out[dn[j].n] = pair;
            ++dn[j].n;
        }
    njobs[0] = nj;
    njobs[1] = nj;
}

void k_swiglu_q8(const float* g, const float* u, block_q8_1* q, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(1),
              b = item_ct1.get_group(2) * (item_ct1.get_local_range(2) / 32) +
                  (item_ct1.get_local_id(2) >> 5),
              lane = item_ct1.get_local_id(2) & 31;
    if (b >= n / 32) return;
    const size_t i = (size_t) row * n + b * 32 + lane;
    const float xi = silu_(g[i]) * u[i];
    const float amax = wmax(sycl::fabs(xi)), sum = wsum(xi);
    const float d = amax / 127.0f;
    block_q8_1& y = q[(size_t) row * (n / 32) + b];
    y.qs[lane] = amax == 0.0f ? 0 : (int8_t)sycl::round(xi / d);
    if (lane == 0) y.ds = sycl::half2(
        sycl::vec<float, 1>(d)
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0],
        sycl::vec<float, 1>(sum)
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0]);
}

void k_combine(const float* part, const float* topw, const float* sh, const float* sh_logit, float* out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1),
              i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= N) return;
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k)
        s = sycl::fma((float)(topw[t * TOPK + k]),
                      (float)(part[((size_t)t * TOPK + k) * N + i]), s);
    out[(size_t) t * N + i] = s + sigm(sh_logit[t]) * sh[(size_t) t * N + i];
}

// ---------------------------------------------------------------- recurrent layer
void k_gdn_conv(const float* x, float* hist, const float* w, float* out, int ch, int T) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= ch) return;
    float win[3 + MAXT];
    for (int i = 0; i < 3; ++i) win[i] = hist[i * ch + c];
    for (int t = 0; t < T; ++t) win[3 + t] = x[(size_t) t * ch + c];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = 0; t < T; ++t)
        out[(size_t) t * ch + c] = silu_(w0 * win[t] + w1 * win[t + 1] + w2 * win[t + 2] + w3 * win[t + 3]);
    for (int i = 0; i < 3; ++i) hist[i * ch + c] = win[T + i];
}

// L2 norm of every q and k head: x / sqrt(sum x^2 + eps)
void k_gdn_qk_norm(float* conv, int nk, int ch, float eps, float *sh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1), h = item_ct1.get_group(2),
              i = item_ct1.get_local_id(2); // 2 * nk heads of HS
    float* x = conv + (size_t) t * ch + h * HS;

    const float v = x[i];
    const float ss = block_sum(v * v, sh);
    x[i] = v * sycl::rsqrt(ss + eps);
}

void k_gdn_gates(const float* alpha, const float* beta_raw, const float* dt, const float* a, float* g, float* beta,
                            int nv, int T) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= nv * T) return;
    const int h = i % nv;
    const float x = alpha[i] + dt[h];
    const float sp = x > 20.0f ? x : sycl::log1p(sycl::native::exp(x));
    g[i] = sp * a[h];
    beta[i] = sigm(beta_raw[i]);
}

// one block per value head; thread (col, part) owns rows [part*32, part*32+32) of state column col.
// S is stored transposed: S_T[col][i] = S[i][col].

void k_gdn_step(float *S, const float *conv, const float *g, const float *beta,
                float *o, int nk, int nv, int ch, int T, float *qs, float *ks) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lv = item_ct1.get_group(2), kh = lv % nk,
              col = item_ct1.get_local_id(2) >> 2,
              part = item_ct1.get_local_id(2) & 3;
    float* st = S + ((size_t) lv * HS + col) * HS + part * 32;
    float s[32];
#pragma unroll
    for (int r = 0; r < 32; ++r) s[r] = st[r];

    const float scale = sycl::rsqrt((float)HS);
    for (int t = 0; t < T; ++t) {
        const float* row = conv + (size_t) t * ch;
        
        
        item_ct1.barrier();
        if (item_ct1.get_local_id(2) < HS) qs[item_ct1.get_local_id(2)] =
            row[kh * HS + item_ct1.get_local_id(2)];
        else if (item_ct1.get_local_id(2) < 2 * HS)
            ks[item_ct1.get_local_id(2) - HS] =
                row[nk * HS + kh * HS + item_ct1.get_local_id(2) - HS];
        
        
        item_ct1.barrier();
        const float gv = sycl::native::exp(g[t * nv + lv]),
                    bv = beta[t * nv + lv];
        const float vv = row[2 * nk * HS + lv * HS + col];
        float kv = 0.f;
#pragma unroll
        for (int r = 0; r < 32; ++r) kv += s[r] * ks[part * 32 + r];
        kv += dpct::permute_sub_group_by_xor(
            sycl::ext::oneapi::this_work_item::get_sub_group(), kv, 1);
        kv += dpct::permute_sub_group_by_xor(
            sycl::ext::oneapi::this_work_item::get_sub_group(), kv, 2);
        const float delta = (vv - gv * kv) * bv;
        float at = 0.f;
#pragma unroll
        for (int r = 0; r < 32; ++r) {
            s[r] = gv * s[r] + ks[part * 32 + r] * delta;
            at += s[r] * qs[part * 32 + r];
        }
        at += dpct::permute_sub_group_by_xor(
            sycl::ext::oneapi::this_work_item::get_sub_group(), at, 1);
        at += dpct::permute_sub_group_by_xor(
            sycl::ext::oneapi::this_work_item::get_sub_group(), at, 2);
        if (part == 0) o[(size_t) t * nv * HS + lv * HS + col] = at * scale;
    }
#pragma unroll
    for (int r = 0; r < 32; ++r) st[r] = s[r];
}

// rms_norm(o) * w * sigmoid(z), per value head
void k_gdn_out_norm(const float* o, const float* z, const float* w, float* y, int nv, float eps,
                    float *sh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1), h = item_ct1.get_group(2),
              i = item_ct1.get_local_id(2);
    const size_t at = (size_t) t * nv * HS + h * HS + i;

    const float v = o[at];
    const float ms = block_sum(v * v, sh) / (float) HS;
#if FNX_27B   // the 27B gates with silu(z), Flash-Next with sigmoid(z)
    y[at] = v * rsqrtf(ms + eps) * w[i] * z[at] * sigm(z[at]);
#else
    y[at] = v * sycl::rsqrt(ms + eps) * w[i] * sigm(z[at]);
#endif
}

#include "gdn_core.dp.hpp"
#include "gdn_prompt.dp.hpp"
#include "gdn_tc_prompt.dp.hpp"

// ---------------------------------------------------------------- attention
// per (t, head): RMS norm with the weight, then NEOX rotary on dims [0, NROT): pairs (i, i + NROT/2)
void norm_rope(float* x, const float* w, int pos, float eps, const Rope& base, float* sh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_local_id(2); // HD threads
    const float v = x[i];
    const float ms = block_sum(v * v, sh) / (float) HD;
    const float y = v * sycl::rsqrt(ms + eps) * w[i];
    
    item_ct1.barrier();
    x[i] = y;
    
    item_ct1.barrier();
    if (i < NROT / 2) {
        const float theta = (float) rope_pos(base, pos, i) * rope_unit_freq(base, i, NROT);
        float sn, cs;
        sn = sycl::sincos(
            theta,
            sycl::address_space_cast<sycl::access::address_space::generic_space,
                                     sycl::access::decorated::yes>(&cs));
        sn *= base.mscale; cs *= base.mscale;
        const float x0 = x[i], x1 = x[i + NROT / 2];
        x[i] = x0 * cs - x1 * sn;
        x[i + NROT / 2] = x0 * sn + x1 * cs;
    }
}

void k_attn_prep(float* qg, float* k, const float* qn, const float* kn, int nq, int nkv, const int* pos0p, float eps, Rope base,
                 float *sh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1), h = item_ct1.get_group(2),
              pos0 = *pos0p;

    if (h < nq) norm_rope(qg + ((size_t) t * nq + h) * 2 * HD, qn, pos0 + t, eps, base, sh);
    else norm_rope(k + ((size_t) t * nkv + (h - nq)) * HD, kn, pos0 + t, eps, base, sh);
}

#include "qsa.dp.hpp"

// split-K attention: block (split s, row t, local KV head kvl) takes positions [s * len, (s + 1) * len) of
// [0, pos0 + t] (len: the context over ATT_SPLITS, rounded up to ATT_CH) in chunks of ATT_CH, for the local q heads
// of that KV head, with an online softmax; it writes each head's running max, sum and HD accumulators. The combine
// (block per local head and row) weighs the splits by exp(max_s - max), divides, and applies sigmoid(gate).
constexpr int ATT_SPLITS = 16, ATT_CH = 32, ATT_QPK = NQH / NKVH, ATT_MAXH = ATT_QPK, ATT_OWN = NQH / NCARD;
constexpr int ATT_ROWS = 64;   // rows the attention's scratch and merge ring hold (a prompt group; windows use MAXT)
struct AttPart {
    float m, z;
};

// positions <= p that card owns (block j on card kv_card(j)): its blocks before p's block whole, then p's up to p
 __dpct_inline__ int kv_owned(int p, int card) {
    const int b = p / 4;
    const int blocks = (b + NCARD - 1 - card) / NCARD;   // blocks j < b with j % NCARD == card
    return blocks * 4 + (kv_card(b) == card ? p % 4 + 1 : 0);
}


void k_attn_part(const float *qg, const sycl::half *kc, const sycl::half *vc,
                 KvPages g, float *acc_out, AttPart *mz_out, int nq, int q0,
                 int nkv, int kv0, const int *pos0p,
                 const int *__restrict__ cells, const int *__restrict__ n_cells,
                 int card, LaneRows lr, float q[12 /*ATT_MAXH*/][256 /*HD*/],
                 float kk[32 /*ATT_CH*/][257 /*HD + 1*/],
                 float e[12 /*ATT_MAXH*/][32 /*ATT_CH*/], float *scale) {
    // t: the window's row; its position and pages: its lane's (lr), or the sequence's *pos0p + t
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int s = item_ct1.get_group(2), t = item_ct1.get_group(1),
              kvl = item_ct1.get_group(0), i = item_ct1.get_local_id(2),
              lane = i & 31, warp = i >> 5;
    g = row_pages(g, lr, t);
    // the attended positions (local indices on this card): the selection's list, or every owned one up to the row's
    const int* list = cells && n_cells[t] >= 0 ? cells + (size_t) t * QSA_MAXSEL : nullptr;
    const int ctx = list ? n_cells[t] : kv_owned(row_pos(lr, pos0p, t), card);
    // the local q heads of KV head kv0 + kvl: global heads [(kv0 + kvl) * QPK, + QPK) within [q0, q0 + nq)
    const int g0 = sycl::max(q0, (kv0 + kvl) * ATT_QPK),
              g1 = sycl::min(q0 + nq, (kv0 + kvl + 1) * ATT_QPK);
    const int h0 = g0 - q0, nh = g1 - g0;
    const int len = ((ctx + ATT_SPLITS - 1) / ATT_SPLITS + ATT_CH - 1) / ATT_CH * ATT_CH;
    const int p0 = s * len, p1 = sycl::min(ctx, p0 + len);

    for (int k = i; k < nh * HD; k += HD) q[k / HD][k % HD] = qg[((size_t) t * nq + h0 + k / HD) * 2 * HD + k % HD];
    constexpr int HPW = (ATT_MAXH + HD / 32 - 1) / (HD / 32);   // heads per warp: warp w keeps heads w, w + 8
    float m[HPW], z[HPW];
#pragma unroll
    for (int k = 0; k < HPW; ++k) {
        m[k] = -INFINITY;
        z[k] = 0.f;
    }
    float acc[ATT_MAXH] = {};
    
    item_ct1.barrier();
#pragma unroll 1
    for (int c0 = p0; c0 < p1; c0 += ATT_CH) {
        const int np = sycl::min(ATT_CH, p1 - c0);
        // the chunk's K rows into shared memory and this thread's dimension of its V rows into registers
        float vr[ATT_CH];
#pragma unroll
        for (int pp = 0; pp < ATT_CH; ++pp) {
            const int pos = pp < np ? (list ? list[c0 + pp] : c0 + pp) : 0;
            if (g.q8) {
                const size_t r = kvq_row(g, pos) + (size_t) kvl * HD + i, sc = kvq_scale(g, pos) + ((size_t) kvl * HD + i) / KV8_GROUP;
                kk[pp][i] =
                    pp < np
                        ? reinterpret_cast<const int8_t *>(kc)[r] *
                              sycl::vec<sycl::half, 1>(kc[sc])
                                  .convert<float,
                                           sycl::rounding_mode::automatic>()[0]
                        : 0.f;
                vr[pp] =
                    pp < np
                        ? reinterpret_cast<const int8_t *>(vc)[r] *
                              sycl::vec<sycl::half, 1>(vc[sc])
                                  .convert<float,
                                           sycl::rounding_mode::automatic>()[0]
                        : 0.f;
                continue;
            }
            const size_t r = kvp_row(g, pos) + (size_t) kvl * HD + i;   // nkv == NKVH: a row holds both heads
            kk[pp][i] =
                pp < np
                    ? sycl::vec<sycl::half, 1>(kc[r])
                          .convert<float, sycl::rounding_mode::automatic>()[0]
                    : 0.f;
            vr[pp] =
                pp < np
                    ? sycl::vec<sycl::half, 1>(vc[r])
                          .convert<float, sycl::rounding_mode::automatic>()[0]
                    : 0.f;
        }
        
        
        item_ct1.barrier();
        // head warp + 8 k: lane = position; its score, then the head's online-softmax step
#pragma unroll
        for (int k = 0; k < HPW; ++k) {
            const int hh = warp + k * (HD / 32);
            if (hh >= nh) break;
            float sacc = 0.f;
#pragma unroll 8
            for (int d = 0; d < HD; ++d)
                sacc = sycl::fma(q[hh][d], kk[lane][d], sacc);
            const float sv =
                lane < np ? sacc * sycl::rsqrt((float)HD) : -INFINITY;
            const float mn = sycl::fmax(m[k], wmax(sv));
            const float w = sv == -INFINITY ? 0.f : sycl::native::exp(sv - mn);
            const float sc =
                m[k] == -INFINITY ? 0.f : sycl::native::exp(m[k] - mn);
            z[k] = z[k] * sc + wsum(w);
            m[k] = mn;
            e[hh][lane] = w;
            if (lane == 0) scale[hh] = sc;
        }
        
        
        item_ct1.barrier();
#pragma unroll
        for (int hh = 0; hh < ATT_MAXH; ++hh)
            if (hh < nh) acc[hh] *= scale[hh];
#pragma unroll
        for (int pp = 0; pp < ATT_CH; ++pp)
#pragma unroll
            for (int hh = 0; hh < ATT_MAXH; ++hh)
                if (hh < nh) acc[hh] = sycl::fma(e[hh][pp], vr[pp], acc[hh]);
    }
#pragma unroll
    for (int hh = 0; hh < ATT_MAXH; ++hh) {
        if (hh >= nh) break;
        const size_t slot = ((size_t) t * nq + h0 + hh) * ATT_SPLITS + s;
        acc_out[slot * HD + i] = acc[hh];
    }
#pragma unroll
    for (int k = 0; k < HPW; ++k) {
        const int hh = warp + k * (HD / 32);
        if (hh < nh && lane == 0) mz_out[((size_t) t * nq + h0 + hh) * ATT_SPLITS + s] = AttPart{m[k], z[k]};
    }
}

// this card's splits of head h, row t, as one partial: part[(t * NQH + h) * (HD + 2)] = acc (HD), m, z
void k_attn_local(const float* acc_in, const AttPart* mz_in, float* part) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int h = item_ct1.get_group(2), t = item_ct1.get_group(1),
              i = item_ct1.get_local_id(2);
    const size_t base = ((size_t) t * NQH + h) * ATT_SPLITS;
    float mx = -INFINITY;
#pragma unroll
    for (int s = 0; s < ATT_SPLITS; ++s) mx = sycl::fmax(mx, mz_in[base + s].m);
    float z = 0.f, acc = 0.f;
#pragma unroll
    for (int s = 0; s < ATT_SPLITS; ++s) {
        const AttPart p = mz_in[base + s];
        if (p.m == -INFINITY) continue;
        const float w = sycl::native::exp(p.m - mx);
        z = sycl::fma((float)w, (float)(p.z), z);
        acc = sycl::fma((float)w, (float)(acc_in[(base + s) * HD + i]), acc);
    }
    float* o = part + ((size_t) t * NQH + h) * (HD + 2);
    o[i] = acc;
    if (i == 0) {
        o[HD] = mx;
        o[HD + 1] = z;
    }
}

// (m, z, a) merged with (m2, z2, a2): the softmax of the union of their positions
 __dpct_inline__ void att_merge(float &m, float &z, float &a,
                                             float m2, float z2, float a2) {
    const float mn = sycl::fmax(m, m2);
    const float w1 = m == -INFINITY ? 0.f : sycl::native::exp(m - mn),
                w2 = m2 == -INFINITY ? 0.f : sycl::native::exp(m2 - mn);
    z = w1 * z + w2 * z2;
    a = w1 * a + w2 * a2;
    m = mn;
}

// the cards' partials reduced onto each head's card (chunk c: heads [ATT_OWN c, ATT_OWN (c + 1))) in NCARD - 1 hops on
// the ring: card g sends its partial of chunk g - 1 to g + 1; each card after merges its own into it and sends it on,
// until the chunk's card, which merges its own last and applies the gate: out[t][h - ATT_OWN g] for its heads. fp32 LL
// words, one receive slot per hop.

void k_att_merge(AttRing r, const float *qg, const float *part, float *out,
                 int T, const unsigned *tagp, int call) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const unsigned tag = *tagp * 256u + (unsigned)call + 1u;
    const int g = r.card, i = item_ct1.get_local_id(2),
              hl = item_ct1.get_group(2),
              t = item_ct1.get_group(
                  1); // head hl of a chunk (0..ATT_OWN-1), row t
    constexpr int W = HD + 2;
    const size_t slot_sz = (size_t) ATT_ROWS * ATT_OWN * W, set_sz = (NCARD - 1) * slot_sz;
    unsigned long long* in = (unsigned long long*) r.in + (tag & 1) * set_sz;
    unsigned long long* nin = (unsigned long long*) r.next_in + (tag & 1) * set_sz;
    auto word = [&](unsigned long long* b, int step, int k) { return b + step * slot_sz + ((size_t) t * ATT_OWN + hl) * W + k; };
    auto mine = [&](int chunk, int k) { return part[((size_t) t * NQH + chunk * ATT_OWN + hl) * W + k]; };
    auto put = [&](unsigned long long* p, float v) {
        sycl_port::store_word(p,
            (unsigned long long)sycl::bit_cast<unsigned int>(v) |
            ((unsigned long long)tag << 32));
    };
    auto take = [&](const unsigned long long* p) {
        
        unsigned spins = 0;
        unsigned long long w;
        for (;;) {
            w = sycl_port::load_word(p);
            if ((unsigned) (w >> 32) == tag) break;
            
            if (++spins >= sycl_port::spin_limit) {
                dpct::atomic_fetch_or<
                    sycl::access::address_space::generic_space>(r.err, 1u);
                break;
            }
        }
        return sycl::bit_cast<float>((unsigned)w);
    };
    // hop 1: my partial of chunk g - 1
    {
        const int c = (g + NCARD - 1) % NCARD;
        put(word(nin, 0, i), mine(c, i));
        if (i == 0) {
            put(word(nin, 0, HD), mine(c, HD));
            put(word(nin, 0, HD + 1), mine(c, HD + 1));
        }
    }
    // hop s + 1: chunk g - 1 - s from g - 1, merged with mine, on towards its card
#pragma unroll
    for (int s = 1; s < NCARD - 1; ++s) {
        const int c = (g + 2 * NCARD - 1 - s) % NCARD;
        float m = take(word(in, s - 1, HD)), z = take(word(in, s - 1, HD + 1)), a = take(word(in, s - 1, i));
        att_merge(m, z, a, mine(c, HD), mine(c, HD + 1), mine(c, i));
        put(word(nin, s, i), a);
        if (i == 0) {
            put(word(nin, s, HD), m);
            put(word(nin, s, HD + 1), z);
        }
    }
    // my chunk: the others' merged partial, then mine; the gate
    {
        float m = take(word(in, NCARD - 2, HD)), z = take(word(in, NCARD - 2, HD + 1)), a = take(word(in, NCARD - 2, i));
        att_merge(m, z, a, mine(g, HD), mine(g, HD + 1), mine(g, i));
        const int hg = g * ATT_OWN + hl;
        out[((size_t) t * ATT_OWN + hl) * HD + i] = a / z * sigm(qg[((size_t) t * NQH + hg) * 2 * HD + HD + i]);
    }
}

// The same merge in ONE hop (NCARD > 2): card g pushes its partial of chunk c into owner c's receive slot (slot (g - c - 1) mod NCARD, one per
// sender, the ring's NCARD - 1 slots), then merges the chunk it owns: its own partial and the senders' in card order 0 .. NCARD - 1.
// A hop on HIP costs a cross-card flag round trip (13-20 us); the ring's NCARD - 1 of them were most of the merge's 360 us on four cards.

void k_att_merge_a2a(AttRing r, const float *qg, const float *part, float *out,
                     int T, const unsigned *tagp, int call) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const unsigned tag = *tagp * 256u + (unsigned)call + 1u;
    const int g = r.card, i = item_ct1.get_local_id(2),
              hl = item_ct1.get_group(2), t = item_ct1.get_group(1);
    constexpr int W = HD + 2;
    const size_t slot_sz = (size_t) ATT_ROWS * ATT_OWN * W, set_sz = (NCARD - 1) * slot_sz;
    auto slot_of = [&](int sender, int owner) { return ((sender - owner - 1) % NCARD + NCARD) % NCARD; };   // 0 .. NCARD - 2 for sender != owner
    auto word = [&](unsigned long long* b, int step, int k) { return b + step * slot_sz + ((size_t) t * ATT_OWN + hl) * W + k; };
    auto mine = [&](int chunk, int k) { return part[((size_t) t * NQH + chunk * ATT_OWN + hl) * W + k]; };
    auto put = [&](unsigned long long* p, float v) {
        sycl_port::store_word(p,
            (unsigned long long)sycl::bit_cast<unsigned int>(v) |
            ((unsigned long long)tag << 32));
    };
    auto take = [&](const unsigned long long* p) {
        
        unsigned spins = 0;
        unsigned long long w;
        for (;;) {
            w = sycl_port::load_word(p);
            if ((unsigned) (w >> 32) == tag) break;
            
            if (++spins >= sycl_port::spin_limit) {
                dpct::atomic_fetch_or<
                    sycl::access::address_space::generic_space>(r.err, 1u);
                break;
            }
        }
        return sycl::bit_cast<float>((unsigned)w);
    };
    for (int c = 0; c < NCARD; ++c) {   // push my partial of every other card's chunk to its owner
        if (c == g) continue;
        unsigned long long* in_c = (unsigned long long*) r.peer_in[c] + (tag & 1) * set_sz;
        const int sl = slot_of(g, c);
        put(word(in_c, sl, i), mine(c, i));
        if (i == 0) {
            put(word(in_c, sl, HD), mine(c, HD));
            put(word(in_c, sl, HD + 1), mine(c, HD + 1));
        }
    }
    unsigned long long* in = (unsigned long long*) r.in + (tag & 1) * set_sz;
    float m = 0.f, z = 0.f, a = 0.f;
    for (int src = 0; src < NCARD; ++src) {   // merge in card order: the same on whichever card owns the chunk
        float m2, z2, a2;
        if (src == g) m2 = mine(g, HD), z2 = mine(g, HD + 1), a2 = mine(g, i);
        else {
            const int sl = slot_of(src, g);
            m2 = take(word(in, sl, HD)), z2 = take(word(in, sl, HD + 1)), a2 = take(word(in, sl, i));
        }
        if (src == 0) m = m2, z = z2, a = a2;
        else att_merge(m, z, a, m2, z2, a2);
    }
    const int hg = g * ATT_OWN + hl;
    out[((size_t) t * ATT_OWN + hl) * HD + i] = a / z * sigm(qg[((size_t) t * NQH + hg) * 2 * HD + HD + i]);
}

void k_attn_combine(const float* qg, const float* acc_in, const AttPart* mz_in, float* out,
                                                     int nq) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int h = item_ct1.get_group(2), t = item_ct1.get_group(1),
              i = item_ct1.get_local_id(2);
    const size_t base = ((size_t) t * nq + h) * ATT_SPLITS;
    float mx = -INFINITY;
#pragma unroll
    for (int s = 0; s < ATT_SPLITS; ++s) mx = sycl::fmax(mx, mz_in[base + s].m);
    float z = 0.f, acc = 0.f;
#pragma unroll
    for (int s = 0; s < ATT_SPLITS; ++s) {
        const AttPart p = mz_in[base + s];
        if (p.m == -INFINITY) continue;
        const float w = sycl::native::exp(p.m - mx);
        z = sycl::fma((float)w, (float)(p.z), z);
        acc = sycl::fma((float)w, (float)(acc_in[(base + s) * HD + i]), acc);
    }
    out[((size_t) t * nq + h) * HD + i] = acc / z * sigm(qg[((size_t) t * nq + h) * 2 * HD + HD + i]);
}

#include "attn_prompt.dp.hpp"
#include "attn_prompt_i8.dp.hpp"
#include "topk.dp.hpp"
#include <time.h>

// ---------------------------------------------------------------- PLE block
// part 1 per (row, stream): grouped norms of key and hidden, the gate, the gated value and its grouped norm
void k_ple_gate(const float* R, const float* key, const float* value, const float* nk,
                                                   const float* nq, const float* nc, float* gated, float* normd, float eps,
                                                   float *sh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1), c = item_ct1.get_group(2);

    float kk[3], qq[3], vv[3];
    float sk = 0.f, sq = 0.f;
    for (int r = 0; r < 3; ++r) {
        const int i = item_ct1.get_local_id(2) + r * 1024;
        kk[r] = i < N ? key[(size_t) t * D + c * N + i] : 0.f;
        qq[r] = i < N ? R[(size_t) t * D + c * N + i] : 0.f;
        vv[r] = i < N ? value[(size_t) t * N + i] : 0.f;
        sk += kk[r] * kk[r];
        sq += qq[r] * qq[r];
    }
    const float rk = sycl::rsqrt(block_sum(sk, sh) / (float)N + eps);
    const float rq = sycl::rsqrt(block_sum(sq, sh) / (float)N + eps);
    float dot = 0.f;
    for (int r = 0; r < 3; ++r) {
        const int i = item_ct1.get_local_id(2) + r * 1024;
        if (i < N) dot += (kk[r] * rk * nk[c * N + i]) * (qq[r] * rq * nq[c * N + i]);
    }
    const float s = block_sum(dot, sh) / sycl::sqrt((float)N);
    const float gate = sigm((s > 0.f ? 1.f : (s < 0.f ? -1.f : 0.f)) *
                            sycl::sqrt(sycl::fmax(sycl::fabs(s), 1e-6f)));
    float sg = 0.f;
    for (int r = 0; r < 3; ++r) {
        vv[r] *= gate;
        sg += vv[r] * vv[r];
    }
    const float rg = sycl::rsqrt(block_sum(sg, sh) / (float)N + eps);
    for (int r = 0; r < 3; ++r) {
        const int i = item_ct1.get_local_id(2) + r * 1024;
        if (i < N) {
            gated[(size_t) t * D + c * N + i] = vv[r];
            normd[(size_t) t * D + c * N + i] = vv[r] * rg * nc[c * N + i];
        }
    }
}

// part 2 per channel: the dilated causal conv over the history (9 rows) and this window, silu, residual update
// hsnap [MAXT][9][D]: the history kept when only the first n rows are (variant n - 1, n < T)

void k_ple_conv(float *R, const float *gated, const float *normd,
                const float *w, float *hist, float *hsnap, int T) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int ch = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                   item_ct1.get_local_id(2);
    if (ch >= D) return;
    constexpr int H = 9, DIL = 3;
    float pad[H + MAXT];
    for (int i = 0; i < H; ++i) pad[i] = hist[i * D + ch];
    for (int t = 0; t < T; ++t) pad[H + t] = normd[(size_t) t * D + ch];
    for (int t = 0; t < T; ++t) {
        float s = 0.f;
        for (int k = 0; k < 4; ++k) s += w[ch * 4 + k] * pad[H + t - (3 - k) * DIL];
        R[(size_t) t * D + ch] += gated[(size_t) t * D + ch] + silu_(s);
    }
    for (int i = 0; i < H; ++i) hist[i * D + ch] = pad[T + i];
    for (int n = 1; n < T; ++n)
        for (int i = 0; i < H; ++i) hsnap[((size_t) (n - 1) * H + i) * D + ch] = pad[n + i];
}

// ---------------------------------------------------------------- ring sum (LL words: two fp16 values | tag << 32)
constexpr int RCH = N / NCARD + 1;     // words per slot row (more than a chunk's pairs)
constexpr int RSTEPS = NCARD - 1;      // receive slots: the reduce hops
constexpr int NP = N / 2;          // pairs per row
// fp32 words: one value | tag per word, pair i at words 2i, 2i + 1
__dpct_inline__ void rput32(unsigned long long *p, sycl::float2 v, unsigned tag,
                            unsigned *err) {
    if (!(sycl::isfinite(v.x()) && sycl::isfinite(v.y())))
        dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(err,
                                                                          2u);
    const unsigned long long t = (unsigned long long) tag << 32;
    sycl_port::store_word(p,
        (unsigned long long)sycl::bit_cast<unsigned int>(v.x()) | t);
    sycl_port::store_word(p + 1,
        (unsigned long long)sycl::bit_cast<unsigned int>(v.y()) | t);
}
__dpct_inline__ float rtake1(const unsigned long long *p, unsigned tag,
                             unsigned *err) {
    
    unsigned spins = 0;
    unsigned long long w;
    for (;;) {
        w = sycl_port::load_word(p);
        if ((unsigned) (w >> 32) == tag) break;
        
        if (++spins >= sycl_port::spin_limit) {
            dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(
                err, 1u);
            break;
        }
    }
    return sycl::bit_cast<float>((unsigned)w);
}
__dpct_inline__ void rput(unsigned long long *p, sycl::float2 v, unsigned tag,
                          unsigned *err) {
    if (!(sycl::fabs(v.x()) <= RING_MAX && sycl::fabs(v.y()) <= RING_MAX))
        dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(
            err, 2u); // out of range (or NaN)
    const sycl::half2 h = sycl::float2(v.x(), v.y())
                              .convert<sycl::half, sycl::rounding_mode::rte>();
    sycl_port::store_word(p, (unsigned long long) *reinterpret_cast<const unsigned*>(&h) | ((unsigned long long) tag << 32));
}
__dpct_inline__ sycl::float2 rtake(const unsigned long long *p, unsigned tag,
                                   unsigned *err) {
    
    unsigned spins = 0;
    unsigned long long w;
    for (;;) {
        w = sycl_port::load_word(p);
        if ((unsigned) (w >> 32) == tag) break;
        
        if (++spins >= sycl_port::spin_limit) {
            dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(
                err, 1u);
            break;
        }
    }
    const unsigned lo = (unsigned) w;
    return (*reinterpret_cast<const sycl::half2 *>(&lo))
        .template convert<float, sycl::rounding_mode::automatic>();
}
__dpct_inline__ int clo(int k) { return k * NP / NCARD; } // in pairs
__dpct_inline__ int chi(int k) { return (k + 1) * NP / NCARD; }
__dpct_inline__ sycl::float2 add2(sycl::float2 a, sycl::float2 b) {
    return sycl::float2(a.x() + b.x(), a.y() + b.y());
}

// this card's addend for element (t, i): x, or the routed experts' weighted rows plus the gated shared row
__dpct_inline__ float ring_x(const float *x, const RingMoe &m, int T, int t,
                             int i) {
    if (!m.part) return x[(size_t) t * N + i];
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k)
        s = sycl::fma((float)(m.topw[t * TOPK + k]),
                      (float)(m.part[((size_t)t * TOPK + k) * N + i]), s);
    return sycl::fma(sigm(m.logits[(size_t)t * (NEXP + 1) + NEXP]),
                     (float)(m.part[((size_t)T * TOPK + t) * N + i]), s);
}
__dpct_inline__ sycl::float2 ring_x2(const float *x, const RingMoe &m, int T,
                                     int t, int p) {
    return sycl::float2(ring_x(x, m, T, t, 2 * p),
                        ring_x(x, m, T, t, 2 * p + 1));
}

template <bool F32>

void k_ring(Ring r, float *x, int T, const unsigned *tagp, int call,
            RingMoe m) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int W = F32 ? 2 : 1; // words per pair
    auto put = [&](unsigned long long *p, sycl::float2 v, unsigned tg) {
        if constexpr (F32) rput32(p, v, tg, r.err);
        else rput(p, v, tg, r.err);
    };
    auto take = [&](const unsigned long long* p, unsigned tg) {
        if constexpr (F32) return sycl::float2(rtake1(p, tg, r.err),
                                               rtake1(p + 1, tg, r.err));
        else return rtake(p, tg, r.err);
    };
    const unsigned tag = *tagp * 256u + (unsigned) call + 1u;
    const int n = item_ct1.get_group_range(2) * item_ct1.get_local_range(2),
              tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2),
              g = r.card;
    // two buffer sets, alternating by tag: a card can start sum k+1 before a peer has drained sum k, never k+2
    const size_t in_set = (size_t) RSTEPS * r.rows * RCH * W, ag_set = (size_t) r.rows * N * W;
    unsigned long long* in = (unsigned long long*) r.in + (tag & 1) * in_set;
    unsigned long long* nin = (unsigned long long*) r.next_in + (tag & 1) * in_set;
    unsigned long long* ag = (unsigned long long*) r.ag + (tag & 1) * ag_set;
    unsigned long long* nag = (unsigned long long*) r.next_ag + (tag & 1) * ag_set;
    auto slot = [&](unsigned long long* b, int s, int t, int i) { return b + (((size_t) s * r.rows + t) * RCH + i) * W; };
    {   // step 0: my part of chunk g to g+1
        const int lo = clo(g), len = chi(g) - lo;
        for (int e = tid; e < T * len; e += n) {
            const int t = e / len, i = e % len;
            put(slot(nin, 0, t, i), ring_x2(x, m, T, t, lo + i), tag);
        }
    }
#pragma unroll
    for (int s = 1; s < RSTEPS; ++s) {   // step s: chunk g-s arrived with s parts; add mine, on to g+1
        const int c = (g + NCARD - s) % NCARD, lo = clo(c), len = chi(c) - lo;
        for (int e = tid; e < T * len; e += n) {
            const int t = e / len, i = e % len;
            const sycl::float2 a = take(slot(in, s - 1, t, i), tag);
            put(slot(nin, s, t, i), add2(a, ring_x2(x, m, T, t, lo + i)), tag);
        }
    }
    {   // step NCARD - 1: chunk g+1 with the other cards' parts; add mine: finished. Keep it, send it on
        const int c = (g + 1) % NCARD, lo = clo(c), len = chi(c) - lo;
        for (int e = tid; e < T * len; e += n) {
            const int t = e / len, i = e % len;
            const sycl::float2 a = take(slot(in, RSTEPS - 1, t, i), tag);
            const sycl::float2 v = add2(a, ring_x2(x, m, T, t, lo + i));
            put(ag + ((size_t) t * NP + lo + i) * W, v, tag);
            put(nag + ((size_t) t * NP + lo + i) * W, v, tag);
        }
    }
#pragma unroll
    for (int f = 1; f < RSTEPS; ++f) {   // chunk g-f+1 arrived finished from g-1: forward it (none on two cards)
        const int c = (g + NCARD - f + 1) % NCARD, lo = clo(c), len = chi(c) - lo;
        for (int e = tid; e < T * len; e += n) {
            const int t = e / len, i = e % len;
            put(nag + ((size_t) t * NP + lo + i) * W, take(ag + ((size_t) t * NP + lo + i) * W, tag), tag);
        }
    }
    // everything: written back into x
    for (int e = tid; e < T * NP; e += n) {
        const sycl::float2 v = take(ag + (size_t)e * W, tag);
        x[2 * e] = v.x();
        x[2 * e + 1] = v.y();
    }
}

// The window sum in ONE hop (NCARD > 2, fp16 words): every card writes its addend of every (row, pair) once, as a tagged word, into its own host
// mailbox ([2 sets][MAXT][NP] words, inside the ring's `in` buffer), then reads the other cards' words and adds all NCARD values in card order 0 ..
// NCARD - 1 in fp32 (its own addend rounded to fp16 first, as the others see it): every card gets the same sum, with one fp16 rounding per addend
// instead of the ring's NCARD - 1. A tagged word is its own valid flag, so there is no fence and no flag write; 2 (NCARD - 1) hops of a cross-card
// round trip (13-20 us each on HIP) become one.
void k_ring_a2a(Ring r, float* x, int T, const unsigned* tagp, int call, RingMoe m) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const unsigned tag = *tagp * 256u + (unsigned)call + 1u;
    const int g = r.card,
              n = item_ct1.get_group_range(2) * item_ct1.get_local_range(2),
              tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);
    static_assert(NCARD < 3 || NP <= 2 * (NCARD - 1) * RCH, "the one-hop sum's mailbox (2 sets x MAXT x NP words) must fit the ring's receive buffer");
    const size_t set_sz = (size_t) MAXT * NP;
    unsigned long long* mine = (unsigned long long*) r.in + (tag & 1) * set_sz;
    for (int e = tid; e < T * NP; e += n) {
        if (*(volatile const unsigned*) r.err & 1u) return;   // a peer never came: do not wait out every element's timeout
        const int t = e / NP, p = e % NP;
        const sycl::float2 v = ring_x2(x, m, T, t, p);
        if (!(sycl::fabs(v.x()) <= RING_MAX && sycl::fabs(v.y()) <= RING_MAX))
            dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(
                r.err, 2u); // out of range (or NaN)
        const sycl::half2 h =
            sycl::float2(v.x(), v.y())
                .convert<sycl::half, sycl::rounding_mode::rte>();
        *(volatile unsigned long long*) (mine + e) = (unsigned long long) *reinterpret_cast<const unsigned*>(&h) | ((unsigned long long) tag << 32);   // volatile: system-coherent, as the ring's words
        sycl::float2 sum = sycl::float2(0.f, 0.f);
#pragma unroll
        for (int src = 0; src < NCARD; ++src) {
            const sycl::float2 w =
                src == g ? h.template convert<float,
                                              sycl::rounding_mode::automatic>()
                         : rtake((const unsigned long long *)r.peer_in[src] +
                                     (tag & 1) * set_sz + e,
                                 tag, r.err);
            sum = src == 0 ? w : add2(sum, w);
        }
        x[2 * e] = sum.x();
        x[2 * e + 1] = sum.y();
    }
}

// The window sum in ONE hop with the words PUSHED into the other cards' device memory (FNX_PEER, AMD): every card stores its tagged word of each (row, pair) into
// slot [source = this card] of every other card's mailbox (a posted remote store, 27 GB/s, no cache to go stale), then reads its own mailbox (local memory) for the
// others' words and adds all NCARD values in card order. A flag round trip over peer memory is 2.1-2.3 us against 13-20 us for stream flags; reads FROM a peer are
// slow across the two pairs of this box, which is why nothing here reads remote memory.
void k_ring_a2a_push(Ring r, float* x, int T, const unsigned* tagp, int call, RingMoe m) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const unsigned tag = *tagp * 256u + (unsigned)call + 1u;
    const int g = r.card,
              n = item_ct1.get_group_range(2) * item_ct1.get_local_range(2),
              tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);
    const size_t slot = (size_t) MAXT * NP, set_sz = (size_t) NCARD * slot;
    const size_t off = (size_t) (tag & 1) * set_sz;
    unsigned long long* mine = (unsigned long long*) r.pk + off;
    for (int e = tid; e < T * NP; e += n) {
        if (*(volatile const unsigned*) r.err & 1u) return;   // a peer never came: do not wait out every element's timeout
        const int t = e / NP, p = e % NP;
        const sycl::float2 v = ring_x2(x, m, T, t, p);
        if (!(sycl::fabs(v.x()) <= RING_MAX && sycl::fabs(v.y()) <= RING_MAX))
            dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(
                r.err, 2u); // out of range (or NaN)
        const sycl::half2 h =
            sycl::float2(v.x(), v.y())
                .convert<sycl::half, sycl::rounding_mode::rte>();
        const unsigned long long word = (unsigned long long) *reinterpret_cast<const unsigned*>(&h) | ((unsigned long long) tag << 32);
#pragma unroll
        for (int d = 0; d < NCARD; ++d)
            if (d != g) *(volatile unsigned long long*) ((unsigned long long*) r.peer_pk[d] + off + (size_t) g * slot + e) = word;
        sycl::float2 sum = sycl::float2(0.f, 0.f);
#pragma unroll
        for (int src = 0; src < NCARD; ++src) {
            const sycl::float2 w =
                src == g ? h.template convert<float,
                                              sycl::rounding_mode::automatic>()
                         : rtake(mine + (size_t)src * slot + e, tag, r.err);
            sum = src == 0 ? w : add2(sum, w);
        }
        x[2 * e] = sum.x();
        x[2 * e + 1] = sum.y();
    }
}

// ---------------------------------------------------------------- argmax
// a float's bits made monotone as unsigned (larger float, larger key)
__dpct_inline__ unsigned fkey(float v) {
    const unsigned u = sycl::bit_cast<unsigned int>(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__dpct_inline__ float fkey_inv(unsigned k) {
    return sycl::bit_cast<float>((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

constexpr int AM_BLOCKS = 16, AM_THREADS = 256;
// keys[t] = max over the row of (fkey(value) << 32 | ~index): the largest value, the lower index on a tie. A NaN or
// inf in the row sets err bit 16 (the key order would pick a NaN's token, or skip it, as if it were a logit)
void k_argmax(const float* __restrict__ logits, int rows, unsigned long long* keys, unsigned* err) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1),
              chunk = (rows + AM_BLOCKS - 1) / AM_BLOCKS,
              i0 = item_ct1.get_group(2) * chunk,
              i1 = sycl::min(rows, i0 + chunk);
    const float* r = logits + (size_t) t * rows;
    unsigned long long best = 0;
    bool bad = false;
    for (int i = i0 + item_ct1.get_local_id(2); i < i1; i += AM_THREADS) {
        const float v = r[i];
        bad = bad || !sycl::isfinite((float)v);
        const unsigned long long k = (unsigned long long) fkey(v) << 32 | (unsigned) ~i;
        best = k > best ? k : best;
    }
    if (bad) dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(
        err, 16u);
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        const unsigned long long ok = dpct::permute_sub_group_by_xor(
            sycl::ext::oneapi::this_work_item::get_sub_group(), best, o);
        best = ok > best ? ok : best;
    }
    if ((item_ct1.get_local_id(2) & 31) == 0)
        dpct::atomic_fetch_max<sycl::access::address_space::generic_space>(
            &keys[t], best);
}
// decodes the keys into (value, index + v0) and leaves them at zero for the next argmax
void k_argmax_fin(unsigned long long* keys, int T, int v0, float* bv, int* bi) {
    const int t =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (t >= T) return;
    const unsigned long long k = keys[t];
    bv[t] = fkey_inv((unsigned) (k >> 32));
    bi[t] = (int) ~(unsigned) k + v0;
    keys[t] = 0ull;
}

}  // namespace

// ================================================================ launch wrappers
#if FNX_KPART == 0
std::vector<Q8Twin>& q8_twins() {
    static std::vector<Q8Twin> v;
    return v;
}
void q8_twin(const void* q8, size_t bytes, int8_t* tq, float* td) { q8_twins().push_back({(const char*) q8, bytes, tq, td}); }
void q8_untwin(const void* q8) {
    auto& v = q8_twins();
    const auto it = std::find_if(v.begin(), v.end(), [&](const Q8Twin& w) { return w.q8 == (const char*) q8; });
    if (it == v.end()) throw std::runtime_error("q8_untwin: a buffer with no twin");
    v.erase(it);
}

void quantize_q8_1(const float* x, void* q, int n, int rows, void* st) {
    int8_t* tq;
    float* td;
    twin_of(q, n, tq, td);
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, rows, (n / 32 + 7) / 8) *
                                  sycl::range<3>(1, 1, 256),
                              sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                k_quantize(x, (block_q8_1 *)q, n, tq, td);
            });
    ck_launch("quantize_q8_1");
}

#endif
#if FNX_KPART == 1
namespace {
template <ggml_type type>
void mmvq_t(const DW &W, size_t w_stride, const Job *jobs, const int *njobs,
            int njobs_max, const void *y, float *out, int out_stride,
            dpct::queue_ptr s) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk, per_block = ggml_cuda_type_traits<type>::qi / vdr_of<type>();
    const int lanes = (W.n_in / qk) * per_block;   // lanes that keep every block of a row busy
    int G = per_block;
    while (G < lanes && G < 32) G *= 2;
    const int ys = W.n_in / 32;
#define FNX_G(g)                                                               \
    {                                                                          \
        iq2s_grid.init(*s);                                                    \
        iq3xxs_grid.init(*s);                                                  \
        iq3s_grid.init(*s);                                                    \
        kvalues_iq4nl.init(*s);                                                \
                                                                               \
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});   \
                                                                               \
        strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {                                    \
            auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();                      \
            auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();                  \
            auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();                      \
            auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();              \
                                                                               \
            sycl::local_accessor<Job, 0> jb_acc_ct1(cgh);                      \
                                                                               \
            auto W_p_ct0 = (const char *)W.p;                                  \
            auto w_stride_ct1 = w_stride;                                      \
            auto W_row_bytes_ct2 = W.row_bytes;                                \
            auto W_n_out_ct3 = W.n_out;                                        \
            auto W_n_in_ct4 = W.n_in;                                          \
            auto jobs_ct5 = jobs;                                              \
            auto njobs_ct6 = njobs;                                            \
            auto njobs_max_ct7 = njobs_max;                                    \
            auto y_ct8 = (const block_q8_1 *)y;                                \
            auto ys_ct9 = ys;                                                  \
            auto out_ct10 = out;                                               \
            auto out_stride_ct11 = out_stride;                                 \
                                                                               \
            cgh.parallel_for(                                                  \
                sycl::nd_range<3>(                                             \
                    sycl::range<3>(1, njobs_max,                               \
                                   (W.n_out + MMVQ_THREADS / g - 1) /          \
                                       (MMVQ_THREADS / g)) *                   \
                        sycl::range<3>(1, 1, MMVQ_THREADS),                    \
                    sycl::range<3>(1, 1, MMVQ_THREADS)),                       \
                [=](sycl::nd_item<3> item_ct1)                                 \
                    [[sycl::reqd_sub_group_size(32)]] {                        \
                        k_mmvq<type, g>(                                       \
                            W_p_ct0, w_stride_ct1, W_row_bytes_ct2,            \
                            W_n_out_ct3, W_n_in_ct4, jobs_ct5, njobs_ct6,      \
                            njobs_max_ct7, y_ct8, ys_ct9, out_ct10,            \
                            out_stride_ct11, iq2s_grid_ptr_ct1,                \
                            iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,            \
                            kvalues_iq4nl_ptr_ct1, jb_acc_ct1);                \
                    });                                                        \
        });                                                                    \
    }
    if constexpr (per_block <= 2) { if (G == 2) { FNX_G(2); return; } }
    if constexpr (per_block <= 4) { if (G == 4) { FNX_G(4); return; } }
    if constexpr (per_block <= 8) { if (G == 8) { FNX_G(8); return; } }
    if constexpr (per_block <= 16) { if (G == 16) { FNX_G(16); return; } }
    FNX_G(32);
#undef FNX_G
}
}  // namespace

void mmvq(const DW& W, size_t w_stride, const Job* jobs, const int* njobs, int njobs_max, const void* y, float* out,
          int out_stride, void* st) {
    dpct::queue_ptr s = (dpct::queue_ptr)st;
    switch (W.type) {
        case T_Q8_0: mmvq_t<GGML_TYPE_Q8_0>(W, w_stride, jobs, njobs, njobs_max, y, out, out_stride, s); break;
        case T_Q4_K: mmvq_t<GGML_TYPE_Q4_K>(W, w_stride, jobs, njobs, njobs_max, y, out, out_stride, s); break;
        case T_Q5_K: mmvq_t<GGML_TYPE_Q5_K>(W, w_stride, jobs, njobs, njobs_max, y, out, out_stride, s); break;
        case T_Q5_1: mmvq_t<GGML_TYPE_Q5_1>(W, w_stride, jobs, njobs, njobs_max, y, out, out_stride, s); break;
        case T_Q6_K: mmvq_t<GGML_TYPE_Q6_K>(W, w_stride, jobs, njobs, njobs_max, y, out, out_stride, s); break;
        default: throw std::runtime_error("mmvq: weight type " + std::to_string(W.type) + " not handled");
    }
    ck_launch("mmvq");
}

namespace {
template <int T>
void gemv_f32_t(const float *W, const float *x, float *y, int rows,
                dpct::queue_ptr s) {
    // a block per row while that still gives every SM a block or two; a warp per row beyond
        if (rows <= 160) strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float[T][4], 0> red_acc_ct1(cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, rows) *
                                                   sycl::range<3>(1, 1, 128),
                                               sycl::range<3>(1, 1, 128)),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(32)]] {
                                     k_gemv_f32<T, 4>(W, x, y, rows,
                                                      red_acc_ct1);
                                 });
        });
        else strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float[T][1], 0> red_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, (rows + 7) / 8) *
                                      sycl::range<3>(1, 1, 256),
                                  sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_gemv_f32<T, 1>(W, x, y, rows, red_acc_ct1);
                    });
        });
}
}  // namespace

void gemv_f32(const float* W, const float* x, float* y, int rows, int n, int T, void* st) {
    if (n != N) throw std::runtime_error("gemv_f32: rows of N columns only");
    dpct::queue_ptr s = (dpct::queue_ptr)st;
    switch (T) {
        case 1: gemv_f32_t<1>(W, x, y, rows, s); break;
        case 2: gemv_f32_t<2>(W, x, y, rows, s); break;
        case 3: gemv_f32_t<3>(W, x, y, rows, s); break;
        case 4: gemv_f32_t<4>(W, x, y, rows, s); break;
        case 5: gemv_f32_t<5>(W, x, y, rows, s); break;
        case 6: gemv_f32_t<6>(W, x, y, rows, s); break;
        case 7: gemv_f32_t<7>(W, x, y, rows, s); break;
        case 8: gemv_f32_t<8>(W, x, y, rows, s); break;
        case 9: gemv_f32_t<9>(W, x, y, rows, s); break;
        case 10: gemv_f32_t<10>(W, x, y, rows, s); break;
        case 11: gemv_f32_t<11>(W, x, y, rows, s); break;
        case 12: gemv_f32_t<12>(W, x, y, rows, s); break;
        case 13: gemv_f32_t<13>(W, x, y, rows, s); break;
        case 14: gemv_f32_t<14>(W, x, y, rows, s); break;
        case 15: gemv_f32_t<15>(W, x, y, rows, s); break;
        case 16: gemv_f32_t<16>(W, x, y, rows, s); break;
        default: throw std::runtime_error("gemv_f32: T out of range");
    }
    ck_launch("gemv_f32");
}

#endif
#if FNX_KPART == 3
size_t hc_scratch_floats(int T) { return (size_t) H_NS * T * H_ROWS + (size_t) T * H_NS + (size_t) T * LR / 4 + (size_t) T * LR / 32 + 1; }

// FNX_HC1=R: windows of up to R rows (at most 8) take k_hc's one-barrier form (default 3: kbench 1-2 us faster a read at
// 1-3 rows, the same at 4-5, slower at 8). Read once; the engine reads it when made, so a bad value fails at start
int hc1_rows() {
    static const int rows = [] {
        const long long r = env_int("FNX_HC1", 3);
        if (r < 0 || r > 8) throw std::runtime_error("FNX_HC1: 0 (off) .. 8 rows, got " + std::to_string(r));
        return (int) r;
    }();
    return rows;
}

int hc_blocks() { return H_BLOCKS; }

// the read's grid on each card (0: hc_prepare not run there)
constexpr int HC_MAXDEV = 64;
int hc_grid_[HC_MAXDEV] = {};

int hc_dev() try {
    int dev = -1;
    const dpct::err0 e = DPCT_CHECK_ERROR(dev = dpct::get_current_device_id());
    
    
    
    if (e != 0) throw std::runtime_error(std::string("hc: cudaGetDevice: ") +
                                         dpct::get_error_string_dummy(e));
    if (dev < 0 || dev >= HC_MAXDEV) throw std::runtime_error("hc: card " + std::to_string(dev) + " past the " + std::to_string(HC_MAXDEV) + " it tracks");
    return dev;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int hc_grid() {
    const int dev = hc_dev();
    if (!hc_grid_[dev]) throw std::runtime_error("hc_read: hc_prepare was not run on card " + std::to_string(dev));
    return hc_grid_[dev];
}

int hc_prepare(float *scratch, int cap) try {
    const int dev = hc_dev();
    int sms = 0;
    const dpct::err0 e =
        DPCT_CHECK_ERROR(sms = dpct::get_device(dev).get_max_compute_units());
    
    
    
    if (e != 0) throw std::runtime_error(std::string("hc_prepare: SM count: ") +
                                         dpct::get_error_string_dummy(e));
    if (sms <= 0) throw std::runtime_error("hc_prepare: card " + std::to_string(dev) + " reports " + std::to_string(sms) + " SMs");
    if (cap < 0 || cap > H_BLOCKS) throw std::runtime_error("hc_prepare: cap 1.." + std::to_string(H_BLOCKS) + " (0: none), got " + std::to_string(cap));
    // One resident group avoids assuming CUDA SM occupancy on an Intel GPU.
    hc_grid_[dev] = 1;
#if FNX_HIP
    // FNX_HC_GRID=n (screen): n blocks of the read resident at once instead of one per multiprocessor. An R9700 reports 32 (one per WGP of two CUs), so the 80 blocks of work
    // ran in 2-3 turns each; several blocks fit one WGP, and a barrier that waits too long sets err bit 4 rather than hang
    if (const long long g = env_int("FNX_HC_GRID", 0); g > 0) hc_grid_[dev] = (int) std::min<long long>(g, cap ? cap : H_BLOCKS);
#endif
    // a counter left by another grid sits mid-generation for this one: its next barrier would let blocks through early
    const unsigned zero = 0;
    const dpct::err0 z = DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(scratch + hc_scratch_floats(MAXT) - 1, &zero, sizeof zero)
            .wait());
    
    
    
    if (z != 0) throw std::runtime_error(
        std::string("hc_prepare: the barrier counter: ") +
        dpct::get_error_string_dummy(z));
    return hc_grid_[dev];
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void hc_read(float* R, const float* bo, const float* inj_prev, const float* w_norm, const Q8S& w_down, const Q8S& w_up,
             const uint16_t* w_inj, float eps, float* mixed, float* inject, float* scratch, unsigned* err, int T, void* st,
             void* q8) {
    HcArgs a{R, bo, inj_prev, w_norm, w_down, w_up, w_inj, eps, mixed, inject, nullptr, nullptr, nullptr, nullptr, nullptr, err,
             (block_q8_1*) q8};
    a.part = scratch;
    a.ssp = a.part + (size_t) H_NS * T * H_ROWS;
    a.loq = (int8_t*) (a.ssp + (size_t) T * H_NS);
    a.lod = a.ssp + (size_t) T * H_NS + (size_t) T * LR / 4;
    // the barrier's counter: the last float of a scratch sized for MAXT rows, so it stays put whatever T is
    a.bar = (unsigned*) (scratch + hc_scratch_floats(MAXT) - 1);
    if (q8) twin_of(q8, N, a.tq, a.td);
    dpct::queue_ptr s = (dpct::queue_ptr)st;
    const bool one = T <= hc1_rows();
    const int grid = hc_grid();
    switch (T) {
        case 1: hc_launch<1>(a, s, one, grid); break;
        case 2: hc_launch<2>(a, s, one, grid); break;
        case 3: hc_launch<3>(a, s, one, grid); break;
        case 4: hc_launch<4>(a, s, one, grid); break;
        case 5: hc_launch<5>(a, s, one, grid); break;
        case 6: hc_launch<6>(a, s, one, grid); break;
        case 7: hc_launch<7>(a, s, one, grid); break;
        case 8: hc_launch<8>(a, s, one, grid); break;
        case 9: hc_launch<9>(a, s, one, grid); break;
        case 10: hc_launch<10>(a, s, one, grid); break;
        case 11: hc_launch<11>(a, s, one, grid); break;
        case 12: hc_launch<12>(a, s, one, grid); break;
        case 13: hc_launch<13>(a, s, one, grid); break;
        case 14: hc_launch<14>(a, s, one, grid); break;
        case 15: hc_launch<15>(a, s, one, grid); break;
        case 16: hc_launch<16>(a, s, one, grid); break;
        default: throw std::runtime_error("hc_read: T out of range");
    }
    ck_launch("hc_read");
}

void hc_preset_barrier(float *scratch, unsigned barriers_left) try {
    const unsigned g = (unsigned) hc_grid(), mod = hc_bar_mod(g);
    if (barriers_left == 0 || barriers_left > mod / g)
        throw std::runtime_error("hc_preset_barrier: barriers_left must be 1.." + std::to_string(mod / g));
    const unsigned v = mod - barriers_left * g;
    const dpct::err0 e = DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(scratch + hc_scratch_floats(MAXT) - 1, &v, sizeof v)
            .wait());
    
    
    
    if (e != 0) throw std::runtime_error(std::string("hc_preset_barrier: ") +
                                         dpct::get_error_string_dummy(e));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#endif
#if FNX_KPART == 0
void route(const float* logits, int T, int* topi, float* topw, Job* jobs_gu, Job* jobs_down, int* njobs, void* st) {
    
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
        
        sycl::local_accessor<float, 1> p_acc_ct1(sycl::range<1>(512 /*NEXP*/),
                                                 cgh);
        sycl::local_accessor<float, 1> sh_acc_ct1(sycl::range<1>(32), cgh);
        sycl::local_accessor<int, 1> si_acc_ct1(sycl::range<1>(32), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                  sycl::range<3>(1, 1, NEXP),
                              sycl::range<3>(1, 1, NEXP)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                k_topk(logits, topi, topw,
                       p_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                           .get(),
                       sh_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                           .get(),
                       si_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                           .get());
            });
    });
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
            [=](sycl::nd_item<3> item_ct1) {
                k_jobs(topi, T, jobs_gu, jobs_down, njobs);
            });
    ck_launch("route");
}

void swiglu_q8_1(const float* g, const float* u, void* q, int n, int rows, void* st) {
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, rows, (n / 32 + 7) / 8) *
                                  sycl::range<3>(1, 1, 256),
                              sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                k_swiglu_q8(g, u, (block_q8_1 *)q, n);
            });
    ck_launch("swiglu_q8_1");
}

void moe_combine(const float* part, const float* topw, const float* sh, const float* sh_logit, float* out, int T, void* st) {
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, T, (N + 255) / 256) *
                                             sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                           k_combine(part, topw, sh, sh_logit, out);
                       });
    ck_launch("moe_combine");
}

void gdn_conv(const float* qkv, float* hist, const float* w, float* out, int ch, int T, void* st) {
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, (ch + 255) / 256) *
                                  sycl::range<3>(1, 1, 256),
                              sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) {
                k_gdn_conv(qkv, hist, w, out, ch, T);
            });
    ck_launch("gdn_conv");
}
void gdn_qk_norm(float* conv, int nk, int ch, int T, void* st) {
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sh_acc_ct1(sycl::range<1>(32), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, T, 2 * nk) *
                                  sycl::range<3>(1, 1, HS),
                              sycl::range<3>(1, 1, HS)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                k_gdn_qk_norm(
                    conv, nk, ch, 1e-6f,
                    sh_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    ck_launch("gdn_qk_norm");
}
void gdn_gates(const float* alpha, const float* beta_raw, const float* dt_bias, const float* ssm_a, float* g, float* beta, int nv,
               int T, void* st) {
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, (nv * T + 127) / 128) *
                                  sycl::range<3>(1, 1, 128),
                              sycl::range<3>(1, 1, 128)),
            [=](sycl::nd_item<3> item_ct1) {
                k_gdn_gates(alpha, beta_raw, dt_bias, ssm_a, g, beta, nv, T);
            });
    ck_launch("gdn_gates");
}
void gdn_step(float* S, const float* conv, const float* g, const float* beta, float* o, int nk, int nv, int ch, int T, void* st) {
    
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
        
        sycl::local_accessor<float, 1> qs_acc_ct1(sycl::range<1>(128 /*HS*/),
                                                  cgh);
        
        sycl::local_accessor<float, 1> ks_acc_ct1(sycl::range<1>(128 /*HS*/),
                                                  cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nv) *
                                  sycl::range<3>(1, 1, HS * 4),
                              sycl::range<3>(1, 1, HS * 4)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                k_gdn_step(
                    S, conv, g, beta, o, nk, nv, ch, T,
                    qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    ks_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    ck_launch("gdn_step");
}
void gdn_out_norm(const float* o, const float* z, const float* w, float* y, int nv, int T, float eps, void* st) {
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sh_acc_ct1(sycl::range<1>(32), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, T, nv) *
                                  sycl::range<3>(1, 1, HS),
                              sycl::range<3>(1, 1, HS)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                k_gdn_out_norm(
                    o, z, w, y, nv, eps,
                    sh_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    ck_launch("gdn_out_norm");
}

#endif
#if FNX_KPART == 5
void attn_prep(float* qg, float* k, const float* q_norm, const float* k_norm, int nq, int nkv, const int* pos0, int T, float eps,
               Rope base, void* st) {
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sh_acc_ct1(sycl::range<1>(32), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, T, nq + nkv) *
                                  sycl::range<3>(1, 1, HD),
                              sycl::range<3>(1, 1, HD)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                k_attn_prep(
                    qg, k, q_norm, k_norm, nq, nkv, pos0, eps, base,
                    sh_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    ck_launch("attn_prep");
}
int qsa_maxsel() { return QSA_MAXSEL; }
// the raw-key ring holds a prompt chunk and the block before it (positions mod its rows): twice the chunk, 2048 at least
namespace {
int qsa_ring = 2048;
}
int qsa_ring_rows() { return qsa_ring; }
void set_prompt_rows(int rows) { qsa_ring = std::max(2048, 2 * rows); }

void qsa_index(const float* y, const float* qn, const float* kn, uint16_t* ring, uint16_t* pooled, const KvPages& g, float* q, const int* pos0, int T,
               float eps, Rope base, void* st) {
    dpct::queue_ptr s = (dpct::queue_ptr)st;
    if (T + QSA_POOL - 1 > qsa_ring) throw std::runtime_error("qsa_index: more rows than the raw-key ring holds");
    sycl::half *r = reinterpret_cast<sycl::half *>(ring);
    sycl::half *pl = reinterpret_cast<sycl::half *>(pooled);
    {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> sh_acc_ct1(sycl::range<1>(32), cgh);
            
            sycl::local_accessor<float, 1> buf_acc_ct1(
                sycl::range<1>(128 /*QSA_DIM*/), cgh);

            auto qsa_ring_ct3 = qsa_ring;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, QSA_HEADS + 1, T) *
                                      sycl::range<3>(1, 1, QSA_DIM),
                                  sycl::range<3>(1, 1, QSA_DIM)),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                    32)]] {
                    k_idx_prep(
                        y, qn, r, qsa_ring_ct3, q, pos0, eps, base,
                        sh_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        buf_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    }
    {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> sh_acc_ct1(sycl::range<1>(32), cgh);
            
            sycl::local_accessor<float, 1> buf_acc_ct1(
                sycl::range<1>(128 /*QSA_DIM*/), cgh);

            auto qsa_ring_ct1 = qsa_ring;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                      sycl::range<3>(1, 1, QSA_DIM),
                                  sycl::range<3>(1, 1, QSA_DIM)),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                    32)]] {
                    k_idx_pool(
                        r, qsa_ring_ct1, kn, pl, g, pos0, eps, base,
                        sh_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        buf_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    }
    ck_launch("qsa_index");
}

void qsa_select(const float* q, const uint16_t* pooled, const KvPages& g, float* scores, int* cells, int* n_cells, int max_pools, const int* pos0, int T,
                int card, void* st, NativeQsa* native, int active_pools) {
    dpct::queue_ptr s = (dpct::queue_ptr)st;
    const sycl::half *pl = reinterpret_cast<const sycl::half *>(pooled);
    if (native) {
        native->scores(q, pooled, g, pos0, T, active_pools, scores);
    } else {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
            
            
            sycl::local_accessor<
                sycl::half[64 /*QSC_ROWS * QSA_HEADS*/][136 /*QSC_LD*/], 0>
                qh_acc_ct1(cgh);
            
            
            sycl::local_accessor<
                sycl::half[64 /*QSC_ROWS * QSA_HEADS*/][136 /*QSC_LD*/], 0>
                ql_acc_ct1(cgh);
            
            sycl::local_accessor<float[8][16][20 /*QSC_SLD*/], 0> st_acc_ct1(
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, (T + QSC_ROWS - 1) / QSC_ROWS,
                                   (max_pools + QSC_KEYS - 1) / QSC_KEYS) *
                        sycl::range<3>(1, 1, 256),
                    sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_idx_score_tc(q, pl, g, scores, max_pools, pos0, T,
                                   qh_acc_ct1, ql_acc_ct1, st_acc_ct1);
                });
        });
    }
    
    auto select = [&](auto thread_count) {
        constexpr int Threads = decltype(thread_count)::value;
        strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<unsigned, 1> hist_acc_ct1(sycl::range<1>(256), cgh);
            sycl::local_accessor<unsigned, 0> prefix_s_acc_ct1(cgh);
            sycl::local_accessor<unsigned, 0> want_s_acc_ct1(cgh);
            sycl::local_accessor<int, 1> s_gt_acc_ct1(sycl::range<1>(Threads), cgh);
            sycl::local_accessor<int, 1> s_eq_acc_ct1(sycl::range<1>(Threads), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, T) * sycl::range<3>(1, 1, Threads),
                                  sycl::range<3>(1, 1, Threads)),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                    k_idx_select<Threads>(
                        scores, cells, n_cells, max_pools, pos0, card, LaneRows{},
                        hist_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                        prefix_s_acc_ct1, want_s_acc_ct1,
                        s_gt_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                        s_eq_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    };
    if (native) select(std::integral_constant<int, 256>{});
    else select(std::integral_constant<int, QSA_SEL_THREADS>{});
    ck_launch("qsa_select");
}

void qsa_select_lanes(const float* q, const uint16_t* pooled, const KvPages& g, float* scores, int* cells, int* n_cells, int max_pools,
                      const LaneRows& lr, int T, int card, void* st) {
    dpct::queue_ptr s = (dpct::queue_ptr)st;
    if (!lr.pos || !lr.pt) throw std::runtime_error("qsa_select_lanes: no lane rows");
    static const bool dense_only = env_flag("FNX_QSA_DENSE");
    if (dense_only) {
        s->memset(n_cells, 0xff,
                  (size_t)T * sizeof(int)); // checked by ck_launch below
        ck_launch("qsa_select_lanes");
        return;
    }
    const sycl::half *pl = reinterpret_cast<const sycl::half *>(pooled);
    {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
            
            sycl::local_accessor<float, 1> qs_acc_ct1(
                sycl::range<1>(512 /*QSA_HEADS * QSA_DIM*/), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, T, (max_pools + 255) / 256) *
                        sycl::range<3>(1, 1, 256),
                    sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_idx_score(
                        q, pl, g, scores, max_pools, nullptr, lr,
                        qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    }
    
    strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<unsigned, 1> hist_acc_ct1(sycl::range<1>(256),
                                                       cgh);
        sycl::local_accessor<unsigned, 0> prefix_s_acc_ct1(cgh);
        sycl::local_accessor<unsigned, 0> want_s_acc_ct1(cgh);
        
        sycl::local_accessor<int, 1> s_gt_acc_ct1(
            sycl::range<1>(1024 /*QSA_SEL_THREADS*/), cgh);
        
        sycl::local_accessor<int, 1> s_eq_acc_ct1(
            sycl::range<1>(1024 /*QSA_SEL_THREADS*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                  sycl::range<3>(1, 1, QSA_SEL_THREADS),
                              sycl::range<3>(1, 1, QSA_SEL_THREADS)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                k_idx_select(
                    scores, cells, n_cells, max_pools, nullptr, card, lr,
                    hist_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    prefix_s_acc_ct1, want_s_acc_ct1,
                    s_gt_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    s_eq_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    ck_launch("qsa_select_lanes");
}
int qsa_rows() { return QSA_ROWS; }

void qsa_proj(const uint16_t* wq, const uint16_t* wk, const float* x, float* y, int T, void* st) {
    dpct::queue_ptr s = (dpct::queue_ptr)st;
    const int grid = (QSA_ROWS + 7) / 8;
#define FNX_QP(t)                                                              \
    strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {                                        \
        auto wq_ct0 = wq;                                                      \
        auto wk_ct1 = wk;                                                      \
        auto x_ct2 = x;                                                        \
        auto y_ct3 = y;                                                        \
                                                                               \
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *        \
                                               sycl::range<3>(1, 1, 256),      \
                                           sycl::range<3>(1, 1, 256)),         \
                         [=](sycl::nd_item<3> item_ct1)                        \
                             [[sycl::reqd_sub_group_size(32)]] {               \
                                 k_idx_proj<t>(wq_ct0, wk_ct1, x_ct2, y_ct3);  \
                             });                                               \
    });
    switch (T) {
        case 1: FNX_QP(1); break;
        case 2: FNX_QP(2); break;
        case 3: FNX_QP(3); break;
        case 4: FNX_QP(4); break;
        case 5: FNX_QP(5); break;
        case 6: FNX_QP(6); break;
        case 7: FNX_QP(7); break;
        case 8: FNX_QP(8); break;
        case 9: FNX_QP(9); break;
        case 10: FNX_QP(10); break;
        case 11: FNX_QP(11); break;
        case 12: FNX_QP(12); break;
        case 13: FNX_QP(13); break;
        case 14: FNX_QP(14); break;
        case 15: FNX_QP(15); break;
        case 16: FNX_QP(16); break;
        default: throw std::runtime_error("qsa: T out of range");
    }
#undef FNX_QP
    ck_launch("qsa proj");
}

void qsa(const uint16_t* wq, const uint16_t* wk, const float* x, float* y, const float* qn, const float* kn, uint16_t* ring,
         uint16_t* pooled, const KvPages& g, float* q, float* scores, int* cells, int* n_cells, int max_pools, const int* pos0, int T, float eps,
         Rope base, int card, void* st) {
    dpct::queue_ptr s = (dpct::queue_ptr)st;
    if (wq) qsa_proj(wq, wk, x, y, T, st);
    sycl::half *r = reinterpret_cast<sycl::half *>(ring);
    sycl::half *pl = reinterpret_cast<sycl::half *>(pooled);
    if (T + QSA_POOL > qsa_ring) throw std::runtime_error("qsa: the ring is shorter than a window and a block");
    {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> sh_acc_ct1(sycl::range<1>(32), cgh);
            
            sycl::local_accessor<float, 1> buf_acc_ct1(
                sycl::range<1>(128 /*QSA_DIM*/), cgh);

            auto qsa_ring_ct4 = qsa_ring;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, QSA_HEADS + 2, T) *
                                      sycl::range<3>(1, 1, QSA_DIM),
                                  sycl::range<3>(1, 1, QSA_DIM)),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                    32)]] {
                    k_idx_prep_pool(
                        y, qn, kn, r, qsa_ring_ct4, q, pl, g, pos0, eps, base,
                        sh_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        buf_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    }
    // FNX_QSA_DENSE=1 (for comparisons): no selection, every row attends every position
    static const bool dense_only = env_flag("FNX_QSA_DENSE");
    if (dense_only) {
        s->memset(n_cells, 0xff,
                  (size_t)T * sizeof(int)); // checked by ck_launch below
        ck_launch("qsa");
        return;
    }
    {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
            
            sycl::local_accessor<float, 1> qs_acc_ct1(
                sycl::range<1>(512 /*QSA_HEADS * QSA_DIM*/), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, T, (max_pools + 255) / 256) *
                        sycl::range<3>(1, 1, 256),
                    sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_idx_score(
                        q, pl, g, scores, max_pools, pos0, LaneRows{},
                        qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    }
    
    strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<unsigned, 1> hist_acc_ct1(sycl::range<1>(256),
                                                       cgh);
        sycl::local_accessor<unsigned, 0> prefix_s_acc_ct1(cgh);
        sycl::local_accessor<unsigned, 0> want_s_acc_ct1(cgh);
        
        sycl::local_accessor<int, 1> s_gt_acc_ct1(
            sycl::range<1>(1024 /*QSA_SEL_THREADS*/), cgh);
        
        sycl::local_accessor<int, 1> s_eq_acc_ct1(
            sycl::range<1>(1024 /*QSA_SEL_THREADS*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                  sycl::range<3>(1, 1, QSA_SEL_THREADS),
                              sycl::range<3>(1, 1, QSA_SEL_THREADS)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                k_idx_select(
                    scores, cells, n_cells, max_pools, pos0, card, LaneRows{},
                    hist_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    prefix_s_acc_ct1, want_s_acc_ct1,
                    s_gt_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    s_eq_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    ck_launch("qsa");
}

size_t attn_scratch_floats(int) { return (size_t) ATT_ROWS * NQH * ATT_SPLITS * (HD + 2) + (size_t) ATT_ROWS * NQH * (HD + 2); }
size_t att_ring_words() { return (size_t) 2 * (NCARD - 1) * ATT_ROWS * ATT_OWN * (HD + 2); }
int attn_rows() { return ATT_ROWS; }
int kv_rows_per_card(int n_ctx) { return (n_ctx / 4 + NCARD - 1) / NCARD * 4 + 4; }

void attn(const float* qg, const uint16_t* kc16, const uint16_t* vc16, const KvPages& g, float* out, const AttRing& ar, const unsigned* tag,
          int call, const int* pos0, int T, float* scratch, void* st, const int* cells, const int* n_cells) {
    ring_call_check(call, "attn");
    float* acc = scratch;
    if (T > ATT_ROWS) throw std::runtime_error("attn: more rows than its buffers hold");
    AttPart* mz = reinterpret_cast<AttPart*>(scratch + (size_t) ATT_ROWS * NQH * ATT_SPLITS * HD);
    float* part = scratch + (size_t) ATT_ROWS * NQH * ATT_SPLITS * (HD + 2);
    const sycl::half *kc = reinterpret_cast<const sycl::half *>(kc16);
    const sycl::half *vc = reinterpret_cast<const sycl::half *>(vc16);
    dpct::queue_ptr s = (dpct::queue_ptr)st;
    {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
            
            
            sycl::local_accessor<float[12 /*ATT_MAXH*/][256 /*HD*/], 0>
                q_acc_ct1(cgh);
            
            
            sycl::local_accessor<float[32 /*ATT_CH*/][257 /*HD + 1*/], 0>
                kk_acc_ct1(cgh);
            
            
            sycl::local_accessor<float[12 /*ATT_MAXH*/][32 /*ATT_CH*/], 0>
                e_acc_ct1(cgh);
            
            sycl::local_accessor<float, 1> scale_acc_ct1(
                sycl::range<1>(12 /*ATT_MAXH*/), cgh);

            auto NQH_ct6 = NQH;
            auto NKVH_ct8 = NKVH;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(NKVH, T, ATT_SPLITS) *
                                      sycl::range<3>(1, 1, HD),
                                  sycl::range<3>(1, 1, HD)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_attn_part(
                            qg, kc, vc, g, acc, mz, NQH_ct6, 0, NKVH_ct8, 0,
                            pos0, cells, n_cells, ar.card, LaneRows{},
                            q_acc_ct1, kk_acc_ct1, e_acc_ct1,
                            scale_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    }
    strata::kprof::kp(s, __FILE__, __func__, __LINE__)->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, T, NQH) * sycl::range<3>(1, 1, HD),
                          sycl::range<3>(1, 1, HD)),
        [=](sycl::nd_item<3> item_ct1) {
            k_attn_local(acc, mz, part);
        });
    if (ar.onehop)
        strata::kprof::kp(s, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, T, ATT_OWN) *
                                              sycl::range<3>(1, 1, HD),
                                          sycl::range<3>(1, 1, HD)),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_att_merge_a2a(ar, qg, part, out, T, tag, call);
                        });
        else strata::kprof::kp(s, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, T, ATT_OWN) *
                                                   sycl::range<3>(1, 1, HD),
                                               sycl::range<3>(1, 1, HD)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 k_att_merge(ar, qg, part, out, T, tag, call);
                             });
    ck_launch("attn");
}

void attn_part_lanes(const float* qg, const uint16_t* kc16, const uint16_t* vc16, const KvPages& g, const LaneRows& lr, int T,
                     float* scratch, void* st, const int* cells, const int* n_cells, int card) {
    if (T > ATT_ROWS) throw std::runtime_error("attn: more rows than its buffers hold");
    if (!lr.pos || !lr.pt) throw std::runtime_error("attn_part_lanes: no lane rows");
    AttPart* mz = reinterpret_cast<AttPart*>(scratch + (size_t) ATT_ROWS * NQH * ATT_SPLITS * HD);
    {
        dpct::has_capability_or_fail(
            ((sycl::queue *)((dpct::queue_ptr)st))->get_device(),
            {sycl::aspect::fp16});

        strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
            
            
            sycl::local_accessor<float[12 /*ATT_MAXH*/][256 /*HD*/], 0>
                q_acc_ct1(cgh);
            
            
            sycl::local_accessor<float[32 /*ATT_CH*/][257 /*HD + 1*/], 0>
                kk_acc_ct1(cgh);
            
            
            sycl::local_accessor<float[12 /*ATT_MAXH*/][32 /*ATT_CH*/], 0>
                e_acc_ct1(cgh);
            
            sycl::local_accessor<float, 1> scale_acc_ct1(
                sycl::range<1>(12 /*ATT_MAXH*/), cgh);

            auto NQH_ct6 = NQH;
            auto NKVH_ct8 = NKVH;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(NKVH, T, ATT_SPLITS) *
                                      sycl::range<3>(1, 1, HD),
                                  sycl::range<3>(1, 1, HD)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_attn_part(
                            qg, reinterpret_cast<const sycl::half *>(kc16),
                            reinterpret_cast<const sycl::half *>(vc16), g,
                            scratch, mz, NQH_ct6, 0, NKVH_ct8, 0, nullptr,
                            cells, n_cells, card, lr, q_acc_ct1, kk_acc_ct1,
                            e_acc_ct1,
                            scale_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    }
    ck_launch("attn lanes");
}

void attn_finish(const float* qg, float* out, const AttRing& ar, const unsigned* tag, int call, int T, float* scratch, void* st) {
    ring_call_check(call, "attn_finish");
    AttPart* mz = reinterpret_cast<AttPart*>(scratch + (size_t) ATT_ROWS * NQH * ATT_SPLITS * HD);
    float* part = scratch + (size_t) ATT_ROWS * NQH * ATT_SPLITS * (HD + 2);
    dpct::queue_ptr s = (dpct::queue_ptr)st;
    strata::kprof::kp(s, __FILE__, __func__, __LINE__)->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, T, NQH) * sycl::range<3>(1, 1, HD),
                          sycl::range<3>(1, 1, HD)),
        [=](sycl::nd_item<3> item_ct1) {
            k_attn_local(scratch, mz, part);
        });
    if (ar.onehop)
        strata::kprof::kp(s, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, T, ATT_OWN) *
                                              sycl::range<3>(1, 1, HD),
                                          sycl::range<3>(1, 1, HD)),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_att_merge_a2a(ar, qg, part, out, T, tag, call);
                        });
        else strata::kprof::kp(s, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, T, ATT_OWN) *
                                                   sycl::range<3>(1, 1, HD),
                                               sycl::range<3>(1, 1, HD)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 k_att_merge(ar, qg, part, out, T, tag, call);
                             });
    ck_launch("attn finish");
}

size_t attn_prompt_block_floats(int C) { return (size_t) C * ATT_OWN * AP_W; }

void attn_prompt(const float* qg, const uint16_t* kc, const uint16_t* vc, const KvPages& g, const int* pos0, const int* cells,
                 const int* n_cells, int card, int C, int r0, int T, float* part, void* st) {
    if (T == 0) return;   // an empty half of a one-row chunk
#if !FNX_HIP && FNX_ARCH >= 80
    // FNX_PFX_ATTN_I8: the INT8 tensor-core kernel (attn_prompt_i8.cuh) on int8 K/V pages, on by default on NVIDIA sm_80+ for the IQ3_S-type files and off for the others (UD-Q4_K_XL lost agent-A8 and A10 with it); FNX_PFX_ATTN_I8=0/1 overrides; HIP keeps the f16 kernel
    static const int i8_env = (int) env_int("FNX_PFX_ATTN_I8", -1);
    const bool i8 = i8_env >= 0 ? i8_env != 0 : attn_i8_default();
    if (i8 && g.q8) {
        if (env_flag("FNX_PF_NATIVE_ATTN")) {
            strata::kernels::QsaSplitAttnLayout layout{
                g.pt, n_cells, pos0, g.page, g.rows, NKVH,
                card, NCARD, C, r0, T, QSA_MAXSEL};
            if (!strata::kernels::qsa_prompt_attn_esimd_split(qg, kc, vc, cells, layout, part, st))
                throw std::runtime_error("PP ESIMD split attention does not support this layout/device");
            return;
        }
        static bool attr[64] = {};
        int dev = 0;
        dev = dpct::get_current_device_id();
        if (dev >= 0 && dev < 64 && !attr[dev]) {
            
; // three blocks an SM
            
            attr[dev] = true;
        }
        {
            dpct::has_capability_or_fail(
                ((sycl::queue *)((dpct::queue_ptr)st))->get_device(),
                {sycl::aspect::fp16});

            strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
                    
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(sizeof(AiSmem)), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, NKVH, T) *
                                              sycl::range<3>(1, 1, AI_THREADS),
                                          sycl::range<3>(1, 1, AI_THREADS)),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                k_pf_attn_i8(
                                    qg,
                                    reinterpret_cast<const sycl::half *>(kc),
                                    reinterpret_cast<const sycl::half *>(vc), g,
                                    pos0, cells, n_cells, card, C, r0, part,
                                    dpct_local_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get());
                            });
                });
        }
        ck_launch("attn_prompt (int8)");
        return;
    }
#endif
    {
        dpct::has_capability_or_fail(
            ((sycl::queue *)((dpct::queue_ptr)st))->get_device(),
            {sycl::aspect::fp16});

        strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(AP_SMEM), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, NKVH, T) *
                                      sycl::range<3>(1, 1, AP_THREADS),
                                  sycl::range<3>(1, 1, AP_THREADS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_pf_attn(
                            qg, reinterpret_cast<const sycl::half *>(kc),
                            reinterpret_cast<const sycl::half *>(vc), g, pos0,
                            cells, n_cells, card, C, r0, part,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    }
    ck_launch("attn_prompt");
}

void attn_prompt_merge(const float* own, const float* recv, const float* qg, float* out, int C, int card, void* st) {
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, ATT_OWN, C) *
                                             sycl::range<3>(1, 1, HD),
                                         sycl::range<3>(1, 1, HD)),
                       [=](sycl::nd_item<3> item_ct1) {
                           k_pf_attn_merge(own, recv, qg, out, C, card);
                       });
    ck_launch("attn_prompt_merge");
}

#endif
#if FNX_KPART == 0
void ple_post(float* R, const float* key, const float* value, const float* nk, const float* nq, const float* nc, const float* convw,
              float* hist, float* hsnap, int T, float eps, void* st, float* scratch) {
    float* gated = scratch;   // gated, then normalized rows
    float* normd = scratch + (size_t) T * D;
    
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sh_acc_ct1(sycl::range<1>(32), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, T, HC) *
                                  sycl::range<3>(1, 1, 1024),
                              sycl::range<3>(1, 1, 1024)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                k_ple_gate(
                    R, key, value, nk, nq, nc, gated, normd, eps,
                    sh_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, (D + 255) / 256) *
                                             sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                           k_ple_conv(R, gated, normd, convw, hist, hsnap, T);
                       });
    ck_launch("ple_post");
}

// flags: [2 sets][kind: the reduce steps' (in), then the finished chunks' (ag)][block]; on n cards kinds 0..n-2 are in
// steps 0..n-2 and kinds n-1..2n-3 ag steps n-1..2n-3 (three cards: in step 0, in step 1, ag step 2, ag step 3)
constexpr int RB_BLOCKS = 240, RB_KINDS = 2 * NCARD - 2;
size_t ring_flag_words() { return (size_t) 2 * RB_KINDS * RB_BLOCKS; }

// block b's segment of a hop's chunk c: pairs [s0, s1) of the T * len (row, pair) elements, row-major
__dpct_inline__ void rb_seg(int T, int c, int &lo, int &len, int &s0, int &s1) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    lo = clo(c);
    len = chi(c) - lo;
    const int n = T * len, seg = (n + item_ct1.get_group_range(2) - 1) /
                                 item_ct1.get_group_range(2);
    s0 = sycl::min(n, (int)((int)item_ct1.get_group(2) * seg));
    s1 = sycl::min(n, s0 + seg);
}
__dpct_inline__ sycl::half2 rb_pack(sycl::float2 v, unsigned *err) {
    if (!(sycl::fabs(v.x()) <= RING_MAX && sycl::fabs(v.y()) <= RING_MAX))
        dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(
            err, 2u); // out of range (or NaN)
    return sycl::float2(v.x(), v.y())
        .convert<sycl::half, sycl::rounding_mode::rte>();
}
// the block's writes, then its flag on the next card (one fence after the barrier covers the block's writes)
__dpct_inline__ void rb_signal(unsigned *flag, unsigned tag) {
    
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0) {
        
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::system);
        strata::sys_store(flag, tag);
    }
}
__dpct_inline__ void rb_wait(const unsigned *flag, unsigned tag,
                             unsigned *err) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (item_ct1.get_local_id(2) == 0) {
        
        unsigned spins = 0;
        while (strata::sys_load(flag) != tag)
            
            if (++spins >= sycl_port::spin_limit) {
                dpct::atomic_fetch_or<
                    sycl::access::address_space::generic_space>(err, 1u);
                break;
            }
        
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::system);
    }
    
    item_ct1.barrier();
}
__dpct_inline__ sycl::float2 rb_load(const sycl::half2 *p) {
    const unsigned w = strata::sys_load(reinterpret_cast<const unsigned*>(p));
    return (*reinterpret_cast<const sycl::half2 *>(&w))
        .template convert<float, sycl::rounding_mode::automatic>();
}


void k_ring_big(Ring r, float *x, int T, const unsigned *tagp, int call,
                RingMoe m) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const unsigned tag = *tagp * 256u + (unsigned)call + 1u;
    const int g = r.card, set = tag & 1, b = item_ct1.get_group(2),
              tid = item_ct1.get_local_id(2), nt = item_ct1.get_local_range(2);
    const size_t in_set = (size_t) RSTEPS * r.rows * RCH, ag_set = (size_t) r.rows * NP;   // in half2 words
    sycl::half2 *in = reinterpret_cast<sycl::half2 *>(r.in) + set * 2 * in_set;
    sycl::half2 *nin =
        reinterpret_cast<sycl::half2 *>(r.next_in) + set * 2 * in_set;
    sycl::half2 *ag = reinterpret_cast<sycl::half2 *>(r.ag) + set * ag_set;
    sycl::half2 *nag =
        reinterpret_cast<sycl::half2 *>(r.next_ag) + set * ag_set;
    unsigned* fl = r.flags + (size_t) set * RB_KINDS * RB_BLOCKS;
    unsigned* nfl = r.next_flags + (size_t) set * RB_KINDS * RB_BLOCKS;
    auto slot = [&](sycl::half2 *base, int s, int t, int i) { return base + ((size_t) s * r.rows + t) * RCH + i;
    };
    int lo, len, s0, s1;
    {   // step 0: my part of chunk g to g+1
        rb_seg(T, g, lo, len, s0, s1);
        for (int e = s0 + tid; e < s1; e += nt) {
            const int t = e / len, i = e % len;
            *slot(nin, 0, t, i) = rb_pack(ring_x2(x, m, T, t, lo + i), r.err);
        }
        rb_signal(nfl + 0 * RB_BLOCKS + b, tag);
    }
    for (int s = 1; s < RSTEPS; ++s) {   // step s: chunk g-s arrived; add mine, on to g+1
        rb_seg(T, (g + NCARD - s) % NCARD, lo, len, s0, s1);
        rb_wait(fl + (s - 1) * RB_BLOCKS + b, tag, r.err);
        for (int e = s0 + tid; e < s1; e += nt) {
            const int t = e / len, i = e % len;
            *slot(nin, s, t, i) = rb_pack(add2(rb_load(slot(in, s - 1, t, i)), ring_x2(x, m, T, t, lo + i)), r.err);
        }
        rb_signal(nfl + s * RB_BLOCKS + b, tag);
    }
    {   // step NCARD - 1: chunk g+1 with the other cards' parts; add mine: finished. Keep it, send it on
        rb_seg(T, (g + 1) % NCARD, lo, len, s0, s1);
        rb_wait(fl + (RSTEPS - 1) * RB_BLOCKS + b, tag, r.err);
        for (int e = s0 + tid; e < s1; e += nt) {
            const int t = e / len, i = e % len;
            const sycl::half2 v =
                rb_pack(add2(rb_load(slot(in, RSTEPS - 1, t, i)),
                             ring_x2(x, m, T, t, lo + i)),
                        r.err);
            ag[(size_t) t * NP + lo + i] = v;
            nag[(size_t) t * NP + lo + i] = v;
        }
        rb_signal(nfl + RSTEPS * RB_BLOCKS + b, tag);
    }
    for (int f = 1; f < RSTEPS; ++f) {   // step NCARD - 1 + f: chunk g-f+1 arrived finished from g-1: forward it
        rb_seg(T, (g + NCARD - f + 1) % NCARD, lo, len, s0, s1);
        rb_wait(fl + (RSTEPS - 1 + f) * RB_BLOCKS + b, tag, r.err);
        for (int e = s0 + tid; e < s1; e += nt) {
            const int t = e / len, i = e % len;
            const unsigned w = strata::sys_load(reinterpret_cast<const unsigned*>(ag + (size_t) t * NP + lo + i));
            nag[(size_t)t * NP + lo + i] =
                *reinterpret_cast<const sycl::half2 *>(&w);
        }
        rb_signal(nfl + (RSTEPS + f) * RB_BLOCKS + b, tag);
    }
    // the last chunk arrives (finished on two cards, forwarded on three: the previous card's last step); then
    // everything into x. Each block waits for its own segments: every block must see every chunk, so it waits on the
    // flags of the blocks that wrote them
    rb_wait(fl + (RB_KINDS - 1) * RB_BLOCKS + b, tag, r.err);
    
    item_ct1.barrier();
}

// after k_ring_big on every block: all chunks are in ag (this card's own chunk g+1, chunk g from the finishing step's
// sender and the rest forwarded); a second launch reads them once every block of the first has passed its waits
void k_ring_big_out(Ring r, float* x, int T, const unsigned* tagp, int call) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const unsigned tag = *tagp * 256u + (unsigned)call + 1u;
    const int set = tag & 1;
    const sycl::half2 *ag =
        reinterpret_cast<const sycl::half2 *>(r.ag) + set * (size_t)r.rows * NP;
    for (int e = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         e < T * NP;
         e += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
        const sycl::float2 v = rb_load(ag + e);
        x[2 * e] = v.x();
        x[2 * e + 1] = v.y();
    }
}

size_t ring_in_words(int rows) { return (size_t) 2 * RSTEPS * rows * RCH * 2; }
size_t ring_ag_words(int rows) { return (size_t) 2 * rows * N * 2; }

void ring_sum(const Ring& r, float* x, int T, const unsigned* tag, int call, void* st, const RingMoe& m) {
    ring_call_check(call, "ring_sum");
    if (T > r.rows) throw std::runtime_error("ring_sum: more rows than the ring's buffers hold");
    if (r.flags && T > MAXT) {
        if (r.f32) throw std::runtime_error("ring_sum: the chunk ring carries fp16 only");
        const int grid = std::min(RB_BLOCKS, std::max(4, (T * NP / NCARD + 255) / 256));
        {
            dpct::has_capability_or_fail(
                ((sycl::queue *)((dpct::queue_ptr)st))->get_device(),
                {sycl::aspect::fp16});

            strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                                     sycl::range<3>(1, 1, 256),
                                                 sycl::range<3>(1, 1, 256)),
                               [=](sycl::nd_item<3> item_ct1) {
                                   k_ring_big(r, x, T, tag, call, m);
                               });
        }
        {
            dpct::has_capability_or_fail(
                ((sycl::queue *)((dpct::queue_ptr)st))->get_device(),
                {sycl::aspect::fp16});

            strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                                     sycl::range<3>(1, 1, 256),
                                                 sycl::range<3>(1, 1, 256)),
                               [=](sycl::nd_item<3> item_ct1) {
                                   k_ring_big_out(r, x, T, tag, call);
                               });
        }
        ck_launch("ring_sum (chunk)");
        return;
    }
    int grid = std::max(4, std::min(T > MAXT ? 240 : 40, (T * NP / NCARD + 255) / 256));
#if FNX_HIP
    // FNX_RING_GRID=n (screen): the one-hop window sum on n blocks of 256 threads instead of at most 40; at 16 rows the 17 blocks left each thread 3-4 elements to push and then wait for
    // one after the other, a remote round trip each
    if (T <= MAXT) {
        static const int g = (int) env_int("FNX_RING_GRID", 0);
        if (g > 0) grid = std::min(g, (T * NP + 255) / 256);
    }
#endif
    if (r.peer && T <= MAXT) {
        {
            dpct::has_capability_or_fail(
                ((sycl::queue *)((dpct::queue_ptr)st))->get_device(),
                {sycl::aspect::fp16});

            strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                                     sycl::range<3>(1, 1, 256),
                                                 sycl::range<3>(1, 1, 256)),
                               [=](sycl::nd_item<3> item_ct1) {
                                   k_ring_a2a_push(r, x, T, tag, call, m);
                               });
        }
        ck_launch("ring_sum (one hop, peer push)");
        return;
    }
    if (r.onehop && T <= MAXT) {
        {
            dpct::has_capability_or_fail(
                ((sycl::queue *)((dpct::queue_ptr)st))->get_device(),
                {sycl::aspect::fp16});

            strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                                     sycl::range<3>(1, 1, 256),
                                                 sycl::range<3>(1, 1, 256)),
                               [=](sycl::nd_item<3> item_ct1) {
                                   k_ring_a2a(r, x, T, tag, call, m);
                               });
        }
        ck_launch("ring_sum (one hop)");
        return;
    }
    if (r.f32) {
        dpct::has_capability_or_fail(
            ((sycl::queue *)((dpct::queue_ptr)st))->get_device(),
            {sycl::aspect::fp16});

        strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                                 sycl::range<3>(1, 1, 256),
                                             sycl::range<3>(1, 1, 256)),
                           [=](sycl::nd_item<3> item_ct1) {
                               k_ring<true>(r, x, T, tag, call, m);
                           });
    } else {
        dpct::has_capability_or_fail(
            ((sycl::queue *)((dpct::queue_ptr)st))->get_device(),
            {sycl::aspect::fp16});

        strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                                 sycl::range<3>(1, 1, 256),
                                             sycl::range<3>(1, 1, 256)),
                           [=](sycl::nd_item<3> item_ct1) {
                               k_ring<false>(r, x, T, tag, call, m);
                           });
    }
    ck_launch("ring_sum");
}

void top_tokens(float* logits, int n, int T, int v0, const int* pen_tok, const int* pen_cnt, const int* pen_n, int pen_stride,
                const float* pen_par, float* out_v, int* out_i, unsigned* err, void* st) {
    if (n < TOPK_N) throw std::runtime_error("top_tokens: a vocabulary slice smaller than the candidates");
    
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<unsigned, 1> hist_acc_ct1(sycl::range<1>(256),
                                                       cgh);
        sycl::local_accessor<unsigned, 0> prefix_s_acc_ct1(cgh);
        sycl::local_accessor<unsigned, 0> want_s_acc_ct1(cgh);
        
        sycl::local_accessor<unsigned long long, 1> cand_acc_ct1(
            sycl::range<1>(256 /*TOPK_CAP*/), cgh);
        sycl::local_accessor<unsigned, 0> n_cand_acc_ct1(cgh);
        
        sycl::local_accessor<unsigned, 1> warp_n_acc_ct1(
            sycl::range<1>(32 /*TOPK_THREADS / 32*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, T) *
                                  sycl::range<3>(1, 1, TOPK_THREADS),
                              sycl::range<3>(1, 1, TOPK_THREADS)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                k_topk_rows(
                    logits, n, v0, pen_tok, pen_cnt, pen_n, pen_stride, pen_par,
                    out_v, out_i, err,
                    hist_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    prefix_s_acc_ct1, want_s_acc_ct1,
                    cand_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    n_cand_acc_ct1,
                    warp_n_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    ck_launch("top_tokens");
}

void argmax(const float* logits, int rows, int T, int v0, float* best_v, int* best_i, unsigned long long* keys, unsigned* err, void* st) {
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, T, AM_BLOCKS) *
                                             sycl::range<3>(1, 1, AM_THREADS),
                                         sycl::range<3>(1, 1, AM_THREADS)),
                       [=](sycl::nd_item<3> item_ct1)
                           [[sycl::reqd_sub_group_size(32)]] {
                               k_argmax(logits, rows, keys, err);
                           });
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 32),
                                         sycl::range<3>(1, 1, 32)),
                       [=](sycl::nd_item<3> item_ct1) {
                           k_argmax_fin(keys, T, v0, best_v, best_i);
                       });
    ck_launch("argmax");
}

namespace {
void k_penalize_row(float* L, int n, const int* pen, const float* par) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int k = pen[0];
    const float repeat = par[0], freq = par[1], present = par[2];
    for (int i = item_ct1.get_local_id(2); i < k;
         i += item_ct1.get_local_range(2)) {
        const int j = pen[1 + i], cnt = pen[1 + k + i];
        if (j < 0 || j >= n || cnt <= 0) continue;
        float l = L[j];
        l = l <= 0.f ? l * repeat : l / repeat;
        L[j] = l - (float) cnt * freq - present;
    }
}
}  // namespace

void penalize_row(float* row, int n, const int* pen, const float* par, void* st) {
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                           k_penalize_row(row, n, pen, par);
                       });
    ck_launch("penalize_row");
}

namespace {
void k_copy_many(const CopyJob* jobs) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const CopyJob j = jobs[item_ct1.get_group(1)];
    sycl::float4 *d = reinterpret_cast<sycl::float4 *>(j.dst);
    const sycl::float4 *s = reinterpret_cast<const sycl::float4 *>(j.src);
    for (size_t i =
             (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < j.bytes / 16; i += (size_t)item_ct1.get_group_range(2) *
                                item_ct1.get_local_range(2)) d[i] = s[i];
}
}  // namespace

void copy_many(const CopyJob* jobs, int n, void* st) {
    if (n <= 0) return;
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, n, 8) *
                                             sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                           k_copy_many(jobs);
                       });
    ck_launch("copy_many");
}
#endif

}  // namespace fnx

#if FNX_KPART == 0
namespace fnx {
namespace {
void k_hc_write(float* R, const float* bo, const float* inj) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1),
              e = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (e >= D) return;
    const int c = e / N, i = e % N;
    R[(size_t)t * D + e] +=
        bo[(size_t)t * N + i] * 2.0f /
        (1.0f + sycl::native::exp(-inj[t * HC + c] / (float)HC));
}
}  // namespace
void hc_write(float* R, const float* bo, const float* inj, int T, void* st) {
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, T, (D + 255) / 256) *
                                             sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                           k_hc_write(R, bo, inj);
                       });
    ck_launch("hc_write");
}
}  // namespace fnx

namespace fnx {
namespace {
// row t (block y) of the window, if its position's block is this card's: both KV heads to the local index
void k_kv_append(const float *k, const float *v, sycl::half *kc, sycl::half *vc,
                 KvPages g, int card, const int *pos0p) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1), p = *pos0p + t, j = p / 4;
    if (kv_card(j) != card) return;
    const int r = kv_local_block(j) * 4 + p % 4;
    if (g.q8) {
        // a warp's 32 elements are one group: d = max |x| / 127, q = round(x / d) (llama.cpp's q8_0)
        static_assert(KV8_GROUP == 32, "a group per warp");
        int8_t *kq = reinterpret_cast<int8_t*>(kc), *vq = reinterpret_cast<int8_t*>(vc);
        const size_t at = kvq_row(g, r), sc = kvq_scale(g, r);
        for (int e = item_ct1.get_local_id(2); e < NKVH * HD;
             e += item_ct1.get_local_range(2)) {
            const float x[2] = {k[(size_t) t * NKVH * HD + e], v[(size_t) t * NKVH * HD + e]};
#pragma unroll
            for (int w = 0; w < 2; ++w) {
                const float d = wmax(sycl::fabs(x[w])) / 127.f,
                            id = d != 0.f ? 1.f / d : 0.f;
                (w ? vq : kq)[at + e] = (int8_t)sycl::round(x[w] * id);
                if ((e & 31) == 0)(w ? vc : kc)[sc + e / KV8_GROUP] =
                    sycl::vec<float, 1>(d)
                        .convert<sycl::half, sycl::rounding_mode::rte>()[0];
            }
        }
        return;
    }
    const size_t at = kvp_row(g, r);
    for (int e = item_ct1.get_local_id(2); e < NKVH * HD;
         e += item_ct1.get_local_range(2)) {
        kc[at + e] = sycl::vec<float, 1>(k[(size_t)t * NKVH * HD + e])
                         .convert<sycl::half, sycl::rounding_mode::rte>()[0];
        vc[at + e] = sycl::vec<float, 1>(v[(size_t)t * NKVH * HD + e])
                         .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    }
}

// k_attn_prep and k_kv_append in one launch: block (h, row t), HD threads; h < NQH: q head h normed and rotated; then
// the NKVH k heads, normed, rotated and appended; then the NKVH v heads, appended (a row whose position is another
// card's appends nothing). The same values as the two kernels
void k_attn_prep_append(float *qg, float *k, const float *v, const float *qn,
                        const float *kn, sycl::half *kc, sycl::half *vc,
                        KvPages g, int card, const int *pos0p, float eps,
                        Rope base, float *sh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1), h = item_ct1.get_group(2),
              i = item_ct1.get_local_id(2), p = *pos0p + t, j = p / 4;

    if (h < NQH) {
        norm_rope(qg + ((size_t) t * NQH + h) * 2 * HD, qn, p, eps, base, sh);
        return;
    }
    const bool is_k = h < NQH + NKVH;
    const int kh = h - NQH - (is_k ? 0 : NKVH), e = kh * HD + i;
    float x;
    if (is_k) {
        float* kr = k + ((size_t) t * NKVH + kh) * HD;
        norm_rope(kr, kn, p, eps, base, sh);
        
        
        item_ct1.barrier();
        x = kr[i];
    } else {
        x = v[(size_t) t * NKVH * HD + e];
    }
    if (kv_card(j) != card) return;
    const int r = kv_local_block(j) * 4 + p % 4;
    sycl::half *cache = is_k ? kc : vc;
    if (g.q8) {
        static_assert(KV8_GROUP == 32 && HD % 32 == 0, "a group per warp");
        const float d = wmax(sycl::fabs(x)) / 127.f,
                    id = d != 0.f ? 1.f / d : 0.f;
        reinterpret_cast<int8_t *>(cache)[kvq_row(g, r) + e] =
            (int8_t)sycl::round(x * id);
        if ((e & 31) == 0) cache[kvq_scale(g, r) + e / KV8_GROUP] =
            sycl::vec<float, 1>(d)
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
        return;
    }
    cache[kvp_row(g, r) + e] =
        sycl::vec<float, 1>(x)
            .convert<sycl::half, sycl::rounding_mode::rte>()[0];
}
}  // namespace
namespace {
// FNX_PREFETCH_PROBE: guess[t][k], the top 10 of logits row t (NEXP each; ties: the lower index), a warp per row
void k_probe_guess(const float* logits, int T, int* guess) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    if (t >= T) return;
    float v[NEXP / 32];
    for (int i = 0; i < NEXP / 32; ++i) v[i] = logits[(size_t) t * NEXP + lane + 32 * i];
    unsigned taken = 0;
    for (int k = 0; k < TOPK; ++k) {
        float bv = -INFINITY;
        int bi = NEXP;
        for (int i = 0; i < NEXP / 32; ++i)
            if (!(taken >> i & 1u) && v[i] > bv) bv = v[i], bi = lane + 32 * i;
        for (int o = 16; o; o >>= 1) {
            const float ov = dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), bv, o);
            const int oi = dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), bi, o);
            if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
        }
        if ((bi & 31) == lane) taken |= 1u << (bi >> 5);
        if (lane == 0) guess[t * TOPK + k] = bi;
    }
}
// the window's distinct experts picked (topi) and guessed: cnt[0] += picked, cnt[1] += guessed, cnt[2] += both,
// cnt[3] += 1 (layer-windows); with the layer's slots (slot_of, before the window's cache plan) cnt[4] += picked and
// not in a slot, cnt[5] += those guessed, cnt[6] += guessed and neither picked nor in a slot (copies a prefetch wastes)
void k_probe_score(const int* topi, const int* guess, int T, const int* slot_of, unsigned long long* cnt,
                   unsigned char *a, unsigned char *g) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    for (int e = item_ct1.get_local_id(2); e < NEXP;
         e += item_ct1.get_local_range(2)) a[e] = g[e] = 0;
    
    item_ct1.barrier();
    for (int i = item_ct1.get_local_id(2); i < T * TOPK;
         i += item_ct1.get_local_range(2)) {
        a[topi[i]] = 1;
        g[guess[i]] = 1;
    }
    
    item_ct1.barrier();
    unsigned na = 0, ng = 0, nb = 0, nm = 0, nmg = 0, nw = 0;
    for (int e = item_ct1.get_local_id(2); e < NEXP;
         e += item_ct1.get_local_range(2)) {
        na += a[e];
        ng += g[e];
        nb += a[e] & g[e];
        if (slot_of && slot_of[e] < 0) {
            nm += a[e];
            nmg += a[e] & g[e];
            nw += g[e] & !a[e];
        }
    }
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &cnt[0], (unsigned long long)na);
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &cnt[1], (unsigned long long)ng);
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &cnt[2], (unsigned long long)nb);
    if (item_ct1.get_local_id(2) == 0)
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &cnt[3], 1ull);
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &cnt[4], (unsigned long long)nm);
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &cnt[5], (unsigned long long)nmg);
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &cnt[6], (unsigned long long)nw);
}
void k_route_count(const int* topi, int n, unsigned* cnt) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < n)
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &cnt[topi[i]], 1u);
}
void k_route_trace(const int* topi, int n, int layer, unsigned* trace, unsigned long long* len, unsigned long long cap) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n) return;
    const unsigned long long at =
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            len, 1ull);
    if (at < cap) trace[at] = (unsigned) layer << 16 | (unsigned) topi[i];
}
}  // namespace
namespace {
// the plan alone (cache_plan, moe.cuh), one warp
void k_cache_plan(const Job* jobs, const int* njobs, int layer, ExpertCache ec, int* list) {
    cache_plan(jobs, njobs[0], layer, ec, list);
}
// the listed experts' matrices (gu_bytes and dn_bytes each, multiples of 16: the layer's) from their host copies to
// their slots in pool p
void k_cache_copy(ExpertCache ec, int p, const int* list, size_t gu_bytes, size_t dn_bytes) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int n = list[0];
    const size_t ug = gu_bytes / 16, per = ug + dn_bytes / 16;
    for (size_t i =
             (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < (size_t)n * per; i += (size_t)item_ct1.get_group_range(2) *
                                   item_ct1.get_local_range(2)) {
        const int k = (int) (i / per);
        const size_t o = i % per;
        const int g = list[1 + k];
        char* at = ec.pool[p] + (size_t) (ec.slot_of[g] - ec.s0[p]) * ec.slot_bytes[p];
        if (o < ug) reinterpret_cast<sycl::uint4 *>(at)[o] =
            reinterpret_cast<const sycl::uint4 *>(ec.host_gu[g])[o];
        else reinterpret_cast<sycl::uint4 *>(at + ec.dn_off[p])[o - ug] =
            reinterpret_cast<const sycl::uint4 *>(ec.host_dn[g])[o - ug];
    }
}
}  // namespace
void cache_copy(const ExpertCache& ec, int p, const int* list, void* st) {
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
        auto ec_dn_off_p_ct3 = ec.dn_off[p];
        auto ec_slot_bytes_p_ec_dn_off_p_ct4 = ec.slot_bytes[p] - ec.dn_off[p];

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 640) *
                                               sycl::range<3>(1, 1, 256),
                                           sycl::range<3>(1, 1, 256)),
                         [=](sycl::nd_item<3> item_ct1) {
                             k_cache_copy(ec, p, list, ec_dn_off_p_ct3,
                                          ec_slot_bytes_p_ec_dn_off_p_ct4);
                         });
    });
    ck_launch("cache_copy");
}
void moe_cache(int layer, size_t gu_stride, size_t dn_stride, const ExpertCache& ec, int* list, void* st, const Job* jobs, const int* njobs) {
    const int p = ec.pool_of[layer];
    if (gu_stride != ec.dn_off[p] || ec.dn_off[p] + dn_stride != ec.slot_bytes[p])
        throw std::runtime_error("moe_cache: the layer's experts are not its pool's size");
        if (jobs)strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 32),
                                             sycl::range<3>(1, 1, 32)),
                           [=](sycl::nd_item<3> item_ct1)
                               [[sycl::reqd_sub_group_size(32)]] {
                                   k_cache_plan(jobs, njobs, layer, ec, list);
                               });
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 160) *
                                             sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                           k_cache_copy(ec, p, list, gu_stride, dn_stride);
                       });
    ck_launch("moe_cache");
}
namespace {
// the arena pool's slots [from, to) lent (their experts to RAM, the slots empty and ref 3), or [to, from) given back
// (empty, ref 0: the first misses take them); then lent[p] = to and the hand past the lent ones
void k_arena_slots(ExpertCache ec, int from, int to) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int p = ec.arena_pool, s0 = ec.s0[p];
    const bool lend = to > from;
    for (int i = (lend ? from : to) +
                 item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         i < (lend ? to : from);
         i += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
        const int s = s0 + i;
        if (const int o = ec.owner[s]; o >= 0) {
            ec.slot_of[o] = -1;
            ec.ep_gu[o] = ec.host_gu[o];
            ec.ep_dn[o] = ec.host_dn[o];
        }
        ec.owner[s] = -1;
        ec.ref[s] = lend ? 3 : 0;
    }
    if (item_ct1.get_group(2) == 0 && item_ct1.get_local_id(2) == 0) {
        ec.lent[p] = to;
        if (ec.hand[p] < s0 + to) ec.hand[p] = s0 + to;
    }
}
}  // namespace
void arena_slots(const ExpertCache& ec, int from, int to, void* st) {
    if (ec.arena_pool < 0 || from < 0 || to < 0 || from > ec.n[ec.arena_pool] || to > ec.n[ec.arena_pool])
        throw std::runtime_error("arena_slots: slots out of range");
    if (from == to) return;
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 64) *
                                             sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                           k_arena_slots(ec, from, to);
                       });
    ck_launch("arena_slots");
}
void route_trace(const int* topi, int n, int layer, unsigned* trace, unsigned long long* len, unsigned long long cap, void* st) {
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, (n + 255) / 256) *
                                             sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                           k_route_trace(topi, n, layer, trace, len, cap);
                       });
    ck_launch("route_trace");
}
void route_count(const int* topi, int n, unsigned* cnt, void* st) {
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, (n + 255) / 256) *
                                             sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                           k_route_count(topi, n, cnt);
                       });
    ck_launch("route_count");
}
void probe_guess(const float* logits, int T, int* guess, void* st) {
    
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 32 * MAXT),
                                         sycl::range<3>(1, 1, 32 * MAXT)),
                       [=](sycl::nd_item<3> item_ct1)
                           [[sycl::reqd_sub_group_size(32)]] {
                               k_probe_guess(logits, T, guess);
                           });
    ck_launch("probe_guess");
}
void probe_score(const int* topi, const int* guess, int T, const int* slot_of, unsigned long long* cnt, void* st) {
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
        
        sycl::local_accessor<unsigned char, 1> a_acc_ct1(
            sycl::range<1>(512 /*NEXP*/), cgh);
        
        sycl::local_accessor<unsigned char, 1> g_acc_ct1(
            sycl::range<1>(512 /*NEXP*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, 128),
                              sycl::range<3>(1, 1, 128)),
            [=](sycl::nd_item<3> item_ct1) {
                k_probe_score(
                    topi, guess, T, slot_of, cnt,
                    a_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    g_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    ck_launch("probe_score");
}
// int8 K/V is prod's and the default; FNX_KV8=0: f16
bool kv_q8() {
    static const bool q8 = [] {
        const char* s = std::getenv("FNX_KV8");
        if (s && std::strcmp(s, "0") && std::strcmp(s, "1")) throw std::runtime_error(std::string("FNX_KV8: 0 or 1, not '") + s + "'");
        return !s || !std::strcmp(s, "1");
    }();
    return q8;
}
void kv_append(const float* k, const float* v, uint16_t* kc, uint16_t* vc, const KvPages& g, int card, int T, const int* pos0, void* st) {
    {
        dpct::has_capability_or_fail(
            ((sycl::queue *)((dpct::queue_ptr)st))->get_device(),
            {sycl::aspect::fp16});

        strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, T, 1) *
                                                 sycl::range<3>(1, 1, 256),
                                             sycl::range<3>(1, 1, 256)),
                           [=](sycl::nd_item<3> item_ct1)
                               [[sycl::reqd_sub_group_size(32)]] {
                                   k_kv_append(
                                       k, v, reinterpret_cast<sycl::half *>(kc),
                                       reinterpret_cast<sycl::half *>(vc), g,
                                       card, pos0);
                               });
    }
    ck_launch("kv_append");
}
void attn_prep_append(float* qg, float* k, const float* v, const float* q_norm, const float* k_norm, uint16_t* kc, uint16_t* vc,
                      const KvPages& g, int card, int T, const int* pos0, float eps, Rope base, void* st) {
    {
        dpct::has_capability_or_fail(
            ((sycl::queue *)((dpct::queue_ptr)st))->get_device(),
            {sycl::aspect::fp16});

        strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> sh_acc_ct1(sycl::range<1>(32), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, T, NQH + 2 * NKVH) *
                                      sycl::range<3>(1, 1, HD),
                                  sycl::range<3>(1, 1, HD)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_attn_prep_append(
                            qg, k, v, q_norm, k_norm,
                            reinterpret_cast<sycl::half *>(kc),
                            reinterpret_cast<sycl::half *>(vc), g, card, pos0,
                            eps, base,
                            sh_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    }
    ck_launch("attn_prep_append");
}
}  // namespace fnx
#endif

#if FNX_KPART == 4

namespace fnx {
static std::atomic<bool> g_gdn_tc_default{true};
void gdn_tc_set_default(bool on) { g_gdn_tc_default = on; }
static std::atomic<bool> g_attn_i8_default{false};
void attn_i8_set_default(bool on) { g_attn_i8_default = on; }
bool attn_i8_default() { return g_attn_i8_default.load(); }
size_t gdn_scratch_floats(int nv) { return (size_t) MAXT * nv * HS + (size_t) nv * MAXT * GDN_Q + (size_t) MAXT * nv; }

void gdn_core(const float* qkvz, const float* ab, float* hist, const float* convw, const float* dt, const float* a, const float* nw,
              float* S, void* yq, const int* ctl, float* scratch, float* snap, float* hsnap, int nk, int nv, const int* vmap, int T,
              float eps, void* st, float* y32) {
    dpct::queue_ptr s = (dpct::queue_ptr)st;
    block_q8_1* y = (block_q8_1*) yq;
    int8_t* tq;
    float* td;
    twin_of(yq, nv * HS, tq, td);
#define FNX_GDN(t)                                                             \
    strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {                                        \
        /*                                                                     \
        DPCT1060:730: SYCL range can only be a 1D, 2D, or 3D vector. Adjust    \
        the code.                                                              \
        */                                                                     \
        sycl::local_accessor<float[2][t][32][4], 0> xs_acc_ct1(cgh);           \
        /*                                                                     \
        DPCT1101:731: 'GDN_QC' expression was replaced with a value. Modify    \
        the code to use the original expression, provided in comments, if it   \
        is correct.                                                            \
        */                                                                     \
        sycl::local_accessor<float[t][32 /*GDN_QC*/], 0> vs_acc_ct1(cgh);      \
        /*                                                                     \
        DPCT1101:732: 'GDN_THREADS / 32' expression was replaced with a value. \
        Modify the code to use the original expression, provided in comments,  \
        if it is correct.                                                      \
        */                                                                     \
        sycl::local_accessor<float[4 /*GDN_THREADS / 32*/][2 * t], 0>          \
            red_acc_ct1(cgh);                                                  \
        sycl::local_accessor<float[t][2], 0> gates_acc_ct1(cgh);               \
        sycl::local_accessor<bool, 0> last_acc_ct1(cgh);                       \
                                                                               \
        auto qkvz_ct0 = qkvz;                                                  \
        auto ab_ct1 = ab;                                                      \
        auto hist_ct2 = hist;                                                  \
        auto convw_ct3 = convw;                                                \
        auto dt_ct4 = dt;                                                      \
        auto a_ct5 = a;                                                        \
        auto nw_ct6 = nw;                                                      \
        auto S_ct7 = S;                                                        \
        auto y_ct8 = y;                                                        \
        auto ctl_ct9 = ctl;                                                    \
        auto scratch_ct10 = scratch;                                           \
        auto snap_ct11 = snap;                                                 \
        auto hsnap_ct12 = hsnap;                                               \
        auto nk_ct13 = nk;                                                     \
        auto nv_ct14 = nv;                                                     \
        auto vmap_ct15 = vmap;                                                 \
        auto eps_ct16 = eps;                                                   \
        auto tq_ct17 = tq;                                                     \
        auto td_ct18 = td;                                                     \
        auto y32_ct19 = y32;                                                   \
        auto ct20 = GdnLanes{};                                                \
                                                                               \
        cgh.parallel_for(                                                      \
            sycl::nd_range<3>(sycl::range<3>(1, 1, nv * GDN_Q) *               \
                                  sycl::range<3>(1, 1, GDN_THREADS),           \
                              sycl::range<3>(1, 1, GDN_THREADS)),              \
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] { \
                k_gdn_core<t>(qkvz_ct0, ab_ct1, hist_ct2, convw_ct3, dt_ct4,   \
                              a_ct5, nw_ct6, S_ct7, y_ct8, ctl_ct9,            \
                              scratch_ct10, snap_ct11, hsnap_ct12, nk_ct13,    \
                              nv_ct14, vmap_ct15, eps_ct16, tq_ct17, td_ct18,  \
                              y32_ct19, ct20, xs_acc_ct1, vs_acc_ct1,          \
                              red_acc_ct1, gates_acc_ct1, last_acc_ct1);       \
            });                                                                \
    });
    switch (T) {
        case 1: FNX_GDN(1); break;
        case 2: FNX_GDN(2); break;
        case 3: FNX_GDN(3); break;
        case 4: FNX_GDN(4); break;
        case 5: FNX_GDN(5); break;
        case 6: FNX_GDN(6); break;
        case 7: FNX_GDN(7); break;
        case 8: FNX_GDN(8); break;
        case 9: FNX_GDN(9); break;
        case 10: FNX_GDN(10); break;
        case 11: FNX_GDN(11); break;
        case 12: FNX_GDN(12); break;
        case 13: FNX_GDN(13); break;
        case 14: FNX_GDN(14); break;
        case 15: FNX_GDN(15); break;
        case 16: FNX_GDN(16); break;
        default: throw std::runtime_error("gdn_core: T out of range");
    }
#undef FNX_GDN
    ck_launch("gdn_core");
}

void gdn_core_lanes(const float* qkvz, const float* ab, const float* convw, const float* dt, const float* a, const float* nw, void* yq,
                    float* scratch, int nk, int nv, const int* vmap, const GdnLanes& gl, int T, float eps, void* st) {
    if (T < 1 || T > MAXT) throw std::runtime_error("gdn_core_lanes: T out of range");
    if (!gl.S || !gl.hist || !gl.ctl) throw std::runtime_error("gdn_core_lanes: no lane rows");
    int8_t* tq;
    float* td;
    twin_of(yq, nv * HS, tq, td);
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float[2][1][32][4], 0> xs_acc_ct1(cgh);
        
        sycl::local_accessor<float[1][32 /*GDN_QC*/], 0> vs_acc_ct1(cgh);
        
        sycl::local_accessor<float[4 /*GDN_THREADS / 32*/][2 * 1], 0>
            red_acc_ct1(cgh);
        sycl::local_accessor<float[1][2], 0> gates_acc_ct1(cgh);
        sycl::local_accessor<bool, 0> last_acc_ct1(cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, T, nv * GDN_Q) *
                                  sycl::range<3>(1, 1, GDN_THREADS),
                              sycl::range<3>(1, 1, GDN_THREADS)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                k_gdn_core<1>(qkvz, ab, nullptr, convw, dt, a, nw, nullptr,
                              (block_q8_1 *)yq, nullptr, scratch, nullptr,
                              nullptr, nk, nv, vmap, eps, tq, td, nullptr, gl,
                              xs_acc_ct1, vs_acc_ct1, red_acc_ct1,
                              gates_acc_ct1, last_acc_ct1);
            });
    });
    ck_launch("gdn_core_lanes");
}

void gdn_core_segs(const float* qkvz, const float* ab, const float* convw, const float* dt, const float* a, const float* nw, void* yq,
                   float* scratch, int nk, int nv, const int* vmap, const GdnLanes& gl, int nseg, int T, float eps, void* st) {
    if (nseg < 1 || T < 1 || nseg * T > MAXT) throw std::runtime_error("gdn_core_segs: segments out of range");
    if (!gl.S || !gl.hist || !gl.ctl || !gl.row0 || !gl.snap || !gl.hsnap) throw std::runtime_error("gdn_core_segs: no segment rows");
    dpct::queue_ptr s = (dpct::queue_ptr)st;
    int8_t* tq;
    float* td;
    twin_of(yq, nv * HS, tq, td);
#define FNX_GDN(t)                                                             \
    strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {                                        \
        sycl::local_accessor<float[2][t][32][4], 0> xs_acc_ct1(cgh);           \
        /*                                                                     \
        DPCT1101:735: 'GDN_QC' expression was replaced with a value. Modify    \
        the code to use the original expression, provided in comments, if it   \
        is correct.                                                            \
        */                                                                     \
        sycl::local_accessor<float[t][32 /*GDN_QC*/], 0> vs_acc_ct1(cgh);      \
        /*                                                                     \
        DPCT1101:736: 'GDN_THREADS / 32' expression was replaced with a value. \
        Modify the code to use the original expression, provided in comments,  \
        if it is correct.                                                      \
        */                                                                     \
        sycl::local_accessor<float[4 /*GDN_THREADS / 32*/][2 * t], 0>          \
            red_acc_ct1(cgh);                                                  \
        sycl::local_accessor<float[t][2], 0> gates_acc_ct1(cgh);               \
        sycl::local_accessor<bool, 0> last_acc_ct1(cgh);                       \
                                                                               \
        auto qkvz_ct0 = qkvz;                                                  \
        auto ab_ct1 = ab;                                                      \
        float *nullptr_ct2 = nullptr;                                          \
        auto convw_ct3 = convw;                                                \
        auto dt_ct4 = dt;                                                      \
        auto a_ct5 = a;                                                        \
        auto nw_ct6 = nw;                                                      \
        float *nullptr_ct7 = nullptr;                                          \
        auto yq_ct8 = (block_q8_1 *)yq;                                        \
        const int *nullptr_ct9 = nullptr;                                      \
        auto scratch_ct10 = scratch;                                           \
        float *nullptr_ct11 = nullptr;                                         \
        float *nullptr_ct12 = nullptr;                                         \
        auto nk_ct13 = nk;                                                     \
        auto nv_ct14 = nv;                                                     \
        auto vmap_ct15 = vmap;                                                 \
        auto eps_ct16 = eps;                                                   \
        auto tq_ct17 = tq;                                                     \
        auto td_ct18 = td;                                                     \
        float *nullptr_ct19 = nullptr;                                         \
        auto gl_ct0 = gl;                                                      \
                                                                               \
        cgh.parallel_for(                                                      \
            sycl::nd_range<3>(sycl::range<3>(1, nseg, nv * GDN_Q) *            \
                                  sycl::range<3>(1, 1, GDN_THREADS),           \
                              sycl::range<3>(1, 1, GDN_THREADS)),              \
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] { \
                k_gdn_core<t>(qkvz_ct0, ab_ct1, nullptr_ct2, convw_ct3,        \
                              dt_ct4, a_ct5, nw_ct6, nullptr_ct7, yq_ct8,      \
                              nullptr_ct9, scratch_ct10, nullptr_ct11,         \
                              nullptr_ct12, nk_ct13, nv_ct14, vmap_ct15,       \
                              eps_ct16, tq_ct17, td_ct18, nullptr_ct19,        \
                              gl_ct0, xs_acc_ct1, vs_acc_ct1, red_acc_ct1,     \
                              gates_acc_ct1, last_acc_ct1);                    \
            });                                                                \
    });
    switch (T) {
        case 1: FNX_GDN(1); break;
        case 2: FNX_GDN(2); break;
        case 3: FNX_GDN(3); break;
        case 4: FNX_GDN(4); break;
        case 5: FNX_GDN(5); break;
        case 6: FNX_GDN(6); break;
        case 7: FNX_GDN(7); break;
        case 8: FNX_GDN(8); break;
        case 9: FNX_GDN(9); break;
        case 10: FNX_GDN(10); break;
        case 11: FNX_GDN(11); break;
        case 12: FNX_GDN(12); break;
        case 13: FNX_GDN(13); break;
        case 14: FNX_GDN(14); break;
        case 15: FNX_GDN(15); break;
        case 16: FNX_GDN(16); break;
        default: throw std::runtime_error("gdn_core_segs: T out of range");
    }
#undef FNX_GDN
    ck_launch("gdn_core_segs");
}

static void gdn_prompt_impl(const float* qkvz, const float* ab, float* hist, const float* convw, const float* dt, const float* a, const float* nw,
                       float* S, const int* ctl, int ng, int nk, int nv, const int* vmap, int C, float eps, float* xc, float* sq, float* y32,
                       float* chw, void* st) {
    dpct::queue_ptr s = (dpct::queue_ptr)st;
    const int ch = (2 * nk + nv) * HS;
    strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
        
        sycl::local_accessor<float, 1> red_acc_ct1(
            sycl::range<1>(4 /*HS / 32*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 2 * nk + nv, C) *
                                  sycl::range<3>(1, 1, HS),
                              sycl::range<3>(1, 1, HS)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                k_gdn_pf_conv(
                    qkvz, hist, convw, ctl, xc, nk, nv,
                    red_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    strata::kprof::kp(s, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, (ch + 255) / 256) *
                                          sycl::range<3>(1, 1, 256),
                                      sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_gdn_pf_hist(qkvz, hist, ctl, ng, C, nk, nv);
                    });
#if FNX_GDN_TC_ON
    // FNX_PFX_GDN_TC: on by default on NVIDIA sm_80+ except for the IQ3_S-type sets (FNX_PFX_GDN_TC=0/1 overrides): Strata's tensor-core chunk scan (gdn_tc_prompt.cuh); false -> this card cannot run it, the sub-chunk kernels below do
    static const int gdn_tc_env = (int) fnx::env_int("FNX_PFX_GDN_TC", -1);   // -1: the file decides (gdn_tc_set_default)
    const bool gdn_tc = gdn_tc_env >= 0 ? gdn_tc_env != 0 : g_gdn_tc_default.load();
    if (gdn_tc && gdn_tc_run(xc, ab, dt, a, S, y32, nk, nv, vmap, C, s)) {
        strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
            
            sycl::local_accessor<float, 1> red_acc_ct1(
                sycl::range<1>(4 /*HS / 32*/), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, nv, C) *
                                      sycl::range<3>(1, 1, HS),
                                  sycl::range<3>(1, 1, HS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_gdn_pf_norm_own(
                            qkvz, nw, y32, nk, nv, eps,
                            red_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
        ck_launch("gdn_prompt");
        return;
    }
#endif
    static const bool chunked = [] {
        const char* e = std::getenv("FNX_PF_GDN_CHUNK");
        if (e && std::strcmp(e, "0") && std::strcmp(e, "1")) throw std::runtime_error(std::string("FNX_PF_GDN_CHUNK: 0 or 1, not '") + e + "'");
        return !e || !std::strcmp(e, "1");
    }();
    if (chunked) {
        static thread_local bool smem_set[16] = {};
        int dev = 0;
        dev = dpct::get_current_device_id();
        if (!smem_set[dev]) {
            
            if (0 != 0)
                ck_launch("gdn_prompt: shared memory");
            smem_set[dev] = true;
        }
        // FNX_PF_GDN_CHECK=1 (debugging): the row-by-row kernel on the same inputs first, then how far the sub-chunk
        // path's outputs and end state are from its (largest difference over the largest value)
        static const bool check = std::getenv("FNX_PF_GDN_CHECK") != nullptr;
        std::vector<float> y_ref, s_ref;
        const size_t ns = (size_t) nv * HS * HS, ny = (size_t) C * nv * HS;
        if (check) {
            float *s0, *yr;
            if (DPCT_CHECK_ERROR(s0 = (float *)sycl_port::alloc_device(
                                     ns * 4, dpct::get_in_order_queue())) !=
                    0 ||
                DPCT_CHECK_ERROR(yr = (float *)sycl_port::alloc_device(
                                     ny * 4, dpct::get_in_order_queue())) != 0)
                ck_launch("gdn check: alloc");
            
            s->memcpy(s0, S, ns * 4);
            strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
                
                
                
                
                sycl::local_accessor<
                    float[2][32 /*GDN_PF_TT*/][16 /*GDN_PF_RP*/]
                         [8 /*GDN_PF_PARTS*/],
                    0>
                    xs_acc_ct1(cgh);
                
                
                sycl::local_accessor<float[32 /*GDN_PF_TT*/][32 /*GDN_QC*/], 0>
                    vs_acc_ct1(cgh);
                
                
                sycl::local_accessor<float[32 /*GDN_PF_TT*/][32 /*GDN_QC*/], 0>
                    os_acc_ct1(cgh);
                
                sycl::local_accessor<float[32 /*GDN_PF_TT*/][2], 0>
                    gates_acc_ct1(cgh);
                
                
                sycl::local_accessor<
                    float[8 /*GDN_PF_THREADS / 32*/][32 /*GDN_PF_TT*/], 0>
                    red_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, nv * GDN_Q) *
                                          sycl::range<3>(1, 1, GDN_PF_THREADS),
                                      sycl::range<3>(1, 1, GDN_PF_THREADS)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            k_gdn_pf_rec(xc, ab, dt, a, S, yr, sq, nk, nv, vmap,
                                         C, xs_acc_ct1, vs_acc_ct1, os_acc_ct1,
                                         gates_acc_ct1, red_acc_ct1);
                        });
            });
            y_ref.resize(ny), s_ref.resize(ns);
            
            s->memcpy(y_ref.data(), yr, ny * 4);
            
            s->memcpy(s_ref.data(), S, ns * 4);
            
            s->memcpy(S, s0, ns * 4);
            s->wait();
            DPCT_CHECK_ERROR(dpct::dpct_free(s0, dpct::get_in_order_queue())),
                DPCT_CHECK_ERROR(
                    dpct::dpct_free(yr, dpct::get_in_order_queue()));
        }
        {
            dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp64});

            strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
                
                
                sycl::local_accessor<float[32 /*GCH*/][129 /*HS + 1*/], 0>
                    ks_acc_ct1(cgh);
                
                
                sycl::local_accessor<float[32 /*GCH*/][129 /*HS + 1*/], 0>
                    qs_acc_ct1(cgh);
                
                
                sycl::local_accessor<float[32 /*GCH*/][33 /*GCH + 1*/], 0>
                    kk_acc_ct1(cgh);
                
                
                sycl::local_accessor<float[32 /*GCH*/][33 /*GCH + 1*/], 0>
                    pp_acc_ct1(cgh);
                
                
                sycl::local_accessor<float[32 /*GCH*/][33 /*GCH + 1*/], 0>
                    tt_acc_ct1(cgh);
                
                sycl::local_accessor<float, 1> lg_acc_ct1(
                    sycl::range<1>(32 /*GCH*/), cgh);
                
                sycl::local_accessor<float, 1> lgam_acc_ct1(
                    sycl::range<1>(32 /*GCH*/), cgh);
                
                sycl::local_accessor<float, 1> bb_acc_ct1(
                    sycl::range<1>(32 /*GCH*/), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, nv, (C + GCH - 1) / GCH) *
                            sycl::range<3>(1, 1, GCH_THREADS),
                        sycl::range<3>(1, 1, GCH_THREADS)),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_gdn_ch_prep(
                            xc, ab, dt, a, vmap, nk, nv, C, chw, ks_acc_ct1,
                            qs_acc_ct1, kk_acc_ct1, pp_acc_ct1, tt_acc_ct1,
                            lg_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            lgam_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            bb_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        }
        strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(GCH_REC_SMEM), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, nv * GCH_REC_BLOCKS_PER_HEAD) *
                        sycl::range<3>(1, 1, GCR_THREADS),
                    sycl::range<3>(1, 1, GCR_THREADS)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_gdn_ch_rec(
                        xc, chw, S, y32, nk, nv, vmap, C,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
        if (check) {
            std::vector<float> y(ny), sn(ns);
            
            s->memcpy(y.data(), y32, ny * 4);
            
            s->memcpy(sn.data(), S, ns * 4);
            s->wait();
            auto rel = [](const std::vector<float>& r, const std::vector<float>& x) {
                double md = 0, mv = 0;
                for (size_t i = 0; i < r.size(); ++i) md = std::max(md, (double) std::fabs(r[i] - x[i])), mv = std::max(mv, (double) std::fabs(r[i]));
                return md / std::max(mv, 1e-30);
            };
            std::fprintf(stderr, "gdn check: C %d, outputs %.3g, end state %.3g (largest difference over largest value)\n", C,
                         rel(y_ref, y), rel(s_ref, sn));
        }
        strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
            
            sycl::local_accessor<float, 1> red_acc_ct1(
                sycl::range<1>(4 /*HS / 32*/), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, nv, C) *
                                      sycl::range<3>(1, 1, HS),
                                  sycl::range<3>(1, 1, HS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_gdn_pf_norm_own(
                            qkvz, nw, y32, nk, nv, eps,
                            red_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    } else {
        strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
            
            
            
            sycl::local_accessor<float[2][32 /*GDN_PF_TT*/][16 /*GDN_PF_RP*/]
                                      [8 /*GDN_PF_PARTS*/],
                                 0>
                xs_acc_ct1(cgh);
            
            
            sycl::local_accessor<float[32 /*GDN_PF_TT*/][32 /*GDN_QC*/], 0>
                vs_acc_ct1(cgh);
            
            
            sycl::local_accessor<float[32 /*GDN_PF_TT*/][32 /*GDN_QC*/], 0>
                os_acc_ct1(cgh);
            
            sycl::local_accessor<float[32 /*GDN_PF_TT*/][2], 0> gates_acc_ct1(
                cgh);
            
            
            sycl::local_accessor<
                float[8 /*GDN_PF_THREADS / 32*/][32 /*GDN_PF_TT*/], 0>
                red_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nv * GDN_Q) *
                                      sycl::range<3>(1, 1, GDN_PF_THREADS),
                                  sycl::range<3>(1, 1, GDN_PF_THREADS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_gdn_pf_rec(xc, ab, dt, a, S, y32, sq, nk, nv, vmap, C,
                                     xs_acc_ct1, vs_acc_ct1, os_acc_ct1,
                                     gates_acc_ct1, red_acc_ct1);
                    });
        });
        strata::kprof::kp(s, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, nv, C) *
                                              sycl::range<3>(1, 1, HS),
                                          sycl::range<3>(1, 1, HS)),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_gdn_pf_norm(qkvz, nw, sq, y32, nk, nv, eps);
                        });
    }
    ck_launch("gdn_prompt");
}
// FNX_GDN_TIME=1 (measuring): CUDA events around every prompt GDN call of a card (conv, history, scan, norm), their GPU time summed per
// card and printed to stderr each time 36 x 32768 rows (a 32K prompt through the 36 recurrent layers) have gone through it (the events are read a call late, nothing waits on the GPU)
void gdn_prompt(const float* qkvz, const float* ab, float* hist, const float* convw, const float* dt, const float* a, const float* nw,
                float* S, const int* ctl, int ng, int nk, int nv, const int* vmap, int C, float eps, float* xc, float* sq, float* y32,
                float* chw, void* st) {
    static const bool timed = fnx::env_int("FNX_GDN_TIME", 0) != 0;
    if (!timed) {
        gdn_prompt_impl(qkvz, ab, hist, convw, dt, a, nw, S, ctl, ng, nk, nv, vmap, C, eps, xc, sq, y32, chw, st);
        return;
    }
    struct Rec { dpct::event_ptr e0, e1; int rows; };
    struct Dev { std::vector<Rec> pend; double ms = 0; long rows = 0, calls = 0, since = 0; };
    static thread_local Dev devs[16];
    int dev = 0;
    dev = dpct::get_current_device_id();
    Dev& d = devs[dev & 15];
    dpct::queue_ptr s = (dpct::queue_ptr)st;
    Rec r{nullptr, nullptr, C};
    DPCT_CHECK_ERROR(r.e0 = new sycl::event()),
        DPCT_CHECK_ERROR(r.e1 = new sycl::event());
    dpct::sync_barrier(r.e0, s);
    gdn_prompt_impl(qkvz, ab, hist, convw, dt, a, nw, S, ctl, ng, nk, nv, vmap, C, eps, xc, sq, y32, chw, st);
    dpct::sync_barrier(r.e1, s);
    d.pend.push_back(r);
    while (!d.pend.empty() &&
           (d.pend.front().e1)
                   ->get_info<sycl::info::event::command_execution_status>() ==
               sycl::info::event_command_status::complete) {
        Rec q = d.pend.front();
        d.pend.erase(d.pend.begin());
        float ms = 0;
        ms = (q.e1->get_profiling_info<
                  sycl::info::event_profiling::command_end>() -
              q.e0->get_profiling_info<
                  sycl::info::event_profiling::command_start>()) /
             1000000.0f;
        d.ms += ms, d.rows += q.rows, d.since += q.rows, ++d.calls;
        DPCT_CHECK_ERROR(dpct::destroy_event(q.e0)),
            DPCT_CHECK_ERROR(dpct::destroy_event(q.e1));
        if (d.since >= 36L * 32768) {
            std::fprintf(stderr, "gdn time: card %d: %.1f ms GPU over %ld calls, %ld rows (%.4f ms a row)\n", dev, d.ms, d.calls, d.rows, d.ms / (double) d.rows);
            d.ms = 0, d.rows = d.calls = d.since = 0;
        }
    }
}

size_t gdn_prompt_work_floats(int C, int nv) { return (size_t) ((C + GCH - 1) / GCH) * nv * GCH_PREP_FLOATS; }
}  // namespace fnx
#endif

namespace fnx {
#if FNX_KPART == 8
void router_route(const float* W, const float* x, float* logits, int T, int* topi, float* topw, Job* jobs_gu, Job* jobs_dn,
                  int* njobs, void* st, const ExpertCache* ec, int layer, int* list, const Q8S* sh_gu, int sh_rows, const void* sh_y,
                  float* sh_out) {
    dpct::queue_ptr s = (dpct::queue_ptr)st;
    ShexpUp su;
    if (sh_gu) {
        if (sh_rows < 1 || sh_rows >= 4096) throw std::runtime_error("router_route: the shared gate|up takes 1..4095 rows");
        for (int k = 0; k < Q8S_SEGS; ++k) su.w.q[k] = sh_gu->q, su.w.d[k] = sh_gu->d, su.w.out[k] = sh_out, su.w.end[k] = sh_rows, su.w.stride[k] = sh_rows;
        su.rows = sh_rows, su.cols = N, su.blocks = (sh_rows + Q8S_THREADS / 32 - 1) / (Q8S_THREADS / 32);
        su.y = (const block_q8_1*) sh_y;
        int8_t* tq;
        float* td;
        twin_of(sh_y, N, tq, td);
        su.tq = tq, su.td = td;
    }
    const int grid = (NEXP + 1 + RR_ROWS - 1) / RR_ROWS + su.blocks;
    const bool split = env_flag("FNX_ROUTER_SPLIT");
    unsigned* counter = (unsigned*) (njobs + 1);
    const ExpertCache none{};
    if (ec && (layer < 0 || layer >= NL || !list)) throw std::runtime_error("router_route: a cache plan needs its layer and list");
#define FNX_RR(t)                                                              \
    for (int pass = 0; pass < (split ? 2 : 1); ++pass) {                        \
    const int phase = split ? pass + 1 : 0;                                   \
    strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {                                        \
        /*                                                                     \
        DPCT1101:774: 'RR_THREADS / 32' expression was replaced with a value.  \
        Modify the code to use the original expression, provided in comments,  \
        if it is correct.                                                      \
        */                                                                     \
        sycl::local_accessor<float[t][8 /*RR_THREADS / 32*/], 0> rsum_acc_ct1( \
            cgh);                                                              \
        sycl::local_accessor<bool, 0> last_acc_ct1(cgh);                       \
        /*                                                                     \
        DPCT1101:775: 'MAXT * TOPK' expression was replaced with a value.      \
        Modify the code to use the original expression, provided in comments,  \
        if it is correct.                                                      \
        */                                                                     \
        sycl::local_accessor<int, 1> ex_acc_ct1(                               \
            sycl::range<1>(160 /*MAXT * TOPK*/), cgh);                         \
        /*                                                                     \
        DPCT1101:776: 'NEXP' expression was replaced with a value. Modify the  \
        code to use the original expression, provided in comments, if it is    \
        correct.                                                               \
        */                                                                     \
        sycl::local_accessor<int, 1> firstpair_acc_ct1(                        \
            sycl::range<1>(512 /*NEXP*/), cgh);                                \
        /*                                                                     \
        DPCT1101:777: 'NEXP' expression was replaced with a value. Modify the  \
        code to use the original expression, provided in comments, if it is    \
        correct.                                                               \
        */                                                                     \
        sycl::local_accessor<int, 1> slots_acc_ct1(                            \
            sycl::range<1>(512 /*NEXP*/), cgh);                                \
        /*                                                                     \
        DPCT1101:778: 'NEXP' expression was replaced with a value. Modify the  \
        code to use the original expression, provided in comments, if it is    \
        correct.                                                               \
        */                                                                     \
        sycl::local_accessor<int, 1> jobof_acc_ct1(                            \
            sycl::range<1>(512 /*NEXP*/), cgh);                                \
        /*                                                                     \
        DPCT1101:779: 'MAXT * TOPK' expression was replaced with a value.      \
        Modify the code to use the original expression, provided in comments,  \
        if it is correct.                                                      \
        */                                                                     \
        sycl::local_accessor<unsigned char, 1> first_acc_ct1(                  \
            sycl::range<1>(160 /*MAXT * TOPK*/), cgh);                         \
        sycl::local_accessor<int, 0> nj_acc_ct1(cgh);                          \
                                                                               \
        auto W_ct0 = W;                                                        \
        auto x_ct1 = x;                                                        \
        auto logits_ct2 = logits;                                              \
        auto topi_ct3 = topi;                                                  \
        auto topw_ct4 = topw;                                                  \
        auto jobs_gu_ct5 = jobs_gu;                                            \
        auto jobs_dn_ct6 = jobs_dn;                                            \
        auto njobs_ct7 = njobs;                                                \
        auto counter_ct8 = counter;                                            \
        const ExpertCache ec_ec_none_ct9 = ec ? *ec : none;                    \
        auto layer_ct10 = layer;                                               \
        auto list_ct11 = list;                                                 \
        auto su_ct0 = phase == 2 ? ShexpUp{} : su;                             \
                                                                               \
        cgh.parallel_for(                                                      \
            sycl::nd_range<3>(sycl::range<3>(1, 1, phase == 2 ? 1 : grid) *   \
                                  sycl::range<3>(1, 1, RR_THREADS),            \
                              sycl::range<3>(1, 1, RR_THREADS)),               \
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] { \
                k_router_route<t>(                                             \
                    W_ct0, x_ct1, logits_ct2, topi_ct3, topw_ct4, jobs_gu_ct5, \
                    jobs_dn_ct6, njobs_ct7, counter_ct8, ec_ec_none_ct9,       \
                    layer_ct10, list_ct11, su_ct0, rsum_acc_ct1, last_acc_ct1, \
                    ex_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()    \
                        .get(),                                                \
                    firstpair_acc_ct1                                          \
                        .get_multi_ptr<sycl::access::decorated::no>()          \
                        .get(),                                                \
                    slots_acc_ct1.get_multi_ptr<sycl::access::decorated::no>() \
                        .get(),                                                \
                    jobof_acc_ct1.get_multi_ptr<sycl::access::decorated::no>() \
                        .get(),                                                \
                    first_acc_ct1.get_multi_ptr<sycl::access::decorated::no>() \
                        .get(),                                                \
                    nj_acc_ct1, phase);                                        \
            });                                                                \
    });                                                                       \
    }
    switch (T) {
        case 1: FNX_RR(1); break;
        case 2: FNX_RR(2); break;
        case 3: FNX_RR(3); break;
        case 4: FNX_RR(4); break;
        case 5: FNX_RR(5); break;
        case 6: FNX_RR(6); break;
        case 7: FNX_RR(7); break;
        case 8: FNX_RR(8); break;
        case 9: FNX_RR(9); break;
        case 10: FNX_RR(10); break;
        case 11: FNX_RR(11); break;
        case 12: FNX_RR(12); break;
        case 13: FNX_RR(13); break;
        case 14: FNX_RR(14); break;
        case 15: FNX_RR(15); break;
        case 16: FNX_RR(16); break;
        default: throw std::runtime_error("router_route: T out of range");
    }
#undef FNX_RR
    ck_launch("router_route");
}

#endif
#if FNX_KPART == 2
namespace {
void moe_mm(bool down, const DW& A, size_t sa, const Job* jobs, const int* njobs, int njobs_max, const void* in, void* out,
            void* st, int tt = 0) {
    const int T = tt ? tt : njobs_max / TOPK;   // the most rows a job has
    const MoeW w{(const char*) A.p, sa, A.row_bytes, A.n_out, A.n_in, A.ep};
    dpct::queue_ptr s = (dpct::queue_ptr)st;
    switch (A.type) {
        case T_Q4_K:
            if (down) moe_launch<GGML_TYPE_Q4_K>(down, w, jobs, njobs, njobs_max, in, (float*) out, s, T);
            else gu_q4k_launch(T, w, jobs, njobs, njobs_max, (const block_q8_1*) in, (float*) out, s);
            break;
        case T_Q5_K: moe_launch<GGML_TYPE_Q5_K>(down, w, jobs, njobs, njobs_max, in, (float*) out, s, T); break;
        case T_Q5_1: moe_launch<GGML_TYPE_Q5_1>(down, w, jobs, njobs, njobs_max, in, (float*) out, s, T); break;
        case T_Q4_0: moe_launch<GGML_TYPE_Q4_0>(down, w, jobs, njobs, njobs_max, in, (float*) out, s, T); break;
        case T_Q8_0: moe_launch<GGML_TYPE_Q8_0>(down, w, jobs, njobs, njobs_max, in, (float*) out, s, T); break;
        case T_IQ4_NL: moe_launch<GGML_TYPE_IQ4_NL>(down, w, jobs, njobs, njobs_max, in, (float*) out, s, T); break;
        case T_IQ4_XS: moe_launch<GGML_TYPE_IQ4_XS>(down, w, jobs, njobs, njobs_max, in, (float*) out, s, T); break;
        case T_IQ3_S: moe_launch<GGML_TYPE_IQ3_S>(down, w, jobs, njobs, njobs_max, in, (float*) out, s, T); break;
        case T_IQ3_XXS: moe_launch<GGML_TYPE_IQ3_XXS>(down, w, jobs, njobs, njobs_max, in, (float*) out, s, T); break;
        case T_IQ2_S: moe_launch<GGML_TYPE_IQ2_S>(down, w, jobs, njobs, njobs_max, in, (float*) out, s, T); break;
        default: throw std::runtime_error("moe: expert type " + std::to_string(A.type) + " not handled");
    }
    ck_launch(down ? "moe_down" : "moe_gate_up");
}

void k_swiglu_stacked(const float* gu, block_q8_1* hq, int ffs, int8_t* tq, float* td) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1),
              b = item_ct1.get_group(2) * (item_ct1.get_local_range(2) / 32) +
                  (item_ct1.get_local_id(2) >> 5),
              lane = item_ct1.get_local_id(2) & 31;
    if (b >= ffs / 32) return;
    const float* g = gu + (size_t) t * 2 * ffs;
    const float xi = silu_(g[b * 32 + lane]) * g[ffs + b * 32 + lane];
    const float amax = wmax(sycl::fabs(xi)), sum = wsum(xi);
    const float d = amax / 127.0f;
    block_q8_1& y = hq[(size_t) t * (ffs / 32) + b];
    const int8_t qv = amax == 0.0f ? 0 : (int8_t)sycl::round(xi / d);
    y.qs[lane] = qv;
    if (lane == 0) y.ds = sycl::half2(
        sycl::vec<float, 1>(d)
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0],
        sycl::vec<float, 1>(sum)
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0]);
    if (tq) {
        tq[(size_t) t * ffs + b * 32 + lane] = qv;
        if (lane == 0) td[(size_t)t * (ffs / 32) + b] =
            sycl::vec<sycl::half, 1>(
                sycl::vec<float, 1>(d)
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0])
                .convert<float, sycl::rounding_mode::automatic>()[0];
    }
}
}  // namespace

void moe_gate_up(const DW& A, size_t sa, const Job* jobs, const int* njobs, int njobs_max, const void* xq, float* gu, void* st,
                 const Q8S* sh_down, int sh_ffs, const float* sh_gu, float* sh_out) {
    if (!sh_down) {
        moe_mm(false, A, sa, jobs, njobs, njobs_max, xq, gu, st);
        return;
    }
    if (sh_ffs < 32 || sh_ffs > SH_MAXF || sh_ffs % 32) throw std::runtime_error("moe_gate_up: the shared expert's FF slice");
    ShexpDown sd;
    for (int k = 0; k < Q8S_SEGS; ++k) sd.w.q[k] = sh_down->q, sd.w.d[k] = sh_down->d, sd.w.out[k] = sh_out, sd.w.end[k] = N, sd.w.stride[k] = N;
    sd.rows = N, sd.ffs = sh_ffs, sd.gu = sh_gu;
    const MoeW w{(const char*) A.p, sa, A.row_bytes, A.n_out, A.n_in, A.ep};
    const int T = njobs_max / TOPK;
    dpct::queue_ptr s = (dpct::queue_ptr)st;
    const block_q8_1* y = (const block_q8_1*) xq;
    switch (A.type) {
        case T_Q4_K: gu_q4k_launch(T, w, jobs, njobs, njobs_max, y, gu, s, sd); break;
        case T_Q5_K: gu_sd_launch<GGML_TYPE_Q5_K>(T, w, jobs, njobs, njobs_max, y, gu, s, sd); break;
        case T_Q5_1: gu_sd_launch<GGML_TYPE_Q5_1>(T, w, jobs, njobs, njobs_max, y, gu, s, sd); break;
        case T_Q4_0: gu_sd_launch<GGML_TYPE_Q4_0>(T, w, jobs, njobs, njobs_max, y, gu, s, sd); break;
        case T_Q8_0: gu_sd_launch<GGML_TYPE_Q8_0>(T, w, jobs, njobs, njobs_max, y, gu, s, sd); break;
        case T_IQ4_NL: gu_sd_launch<GGML_TYPE_IQ4_NL>(T, w, jobs, njobs, njobs_max, y, gu, s, sd); break;
        case T_IQ4_XS: gu_sd_launch<GGML_TYPE_IQ4_XS>(T, w, jobs, njobs, njobs_max, y, gu, s, sd); break;
        case T_IQ3_S: gu_sd_launch<GGML_TYPE_IQ3_S>(T, w, jobs, njobs, njobs_max, y, gu, s, sd); break;
        case T_IQ3_XXS: gu_sd_launch<GGML_TYPE_IQ3_XXS>(T, w, jobs, njobs, njobs_max, y, gu, s, sd); break;
        case T_IQ2_S: gu_sd_launch<GGML_TYPE_IQ2_S>(T, w, jobs, njobs, njobs_max, y, gu, s, sd); break;
        default: throw std::runtime_error("moe_gate_up: the shared expert rides in the launch of expert type " + std::to_string(A.type) + " only for the types moe_mm handles");
    }
    ck_launch("moe_gate_up (with the shared expert)");
}
namespace {
void moe_mm_big(bool down, const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const void* in, void* out,
                void* st) {
    const MoeW w{(const char*) A.p, sa, A.row_bytes, A.n_out, A.n_in, A.ep};
    dpct::queue_ptr s = (dpct::queue_ptr)st;
    switch (A.type) {
        case T_Q4_K:
            if (down) moe_launch_big<GGML_TYPE_Q4_K>(down, w, jobs, njobs, njobs_cap, in, (float*) out, s);
            else {
                if (w.ncols != N || w.row_a != (size_t) N / 256 * 144) throw std::runtime_error("moe gate|up q4_K: row layout");
            strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<BigJob, 0> jb_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, njobs_cap,
                                       (w.nrows + Q4K_ROWS - 1) / Q4K_ROWS) *
                            sycl::range<3>(1, 1, Q4K_THREADS),
                        sycl::range<3>(1, 1, Q4K_THREADS)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            k_moe_gu_q4k_big(w, jobs, njobs,
                                             (const block_q8_1 *)in,
                                             (float *)out, jb_acc_ct1);
                        });
            });
            }
            break;
        case T_Q5_K: moe_launch_big<GGML_TYPE_Q5_K>(down, w, jobs, njobs, njobs_cap, in, (float*) out, s); break;
        case T_Q5_1: moe_launch_big<GGML_TYPE_Q5_1>(down, w, jobs, njobs, njobs_cap, in, (float*) out, s); break;
        case T_Q4_0: moe_launch_big<GGML_TYPE_Q4_0>(down, w, jobs, njobs, njobs_cap, in, (float*) out, s); break;
        case T_Q8_0: moe_launch_big<GGML_TYPE_Q8_0>(down, w, jobs, njobs, njobs_cap, in, (float*) out, s); break;
        case T_IQ4_NL: moe_launch_big<GGML_TYPE_IQ4_NL>(down, w, jobs, njobs, njobs_cap, in, (float*) out, s); break;
        case T_IQ4_XS: moe_launch_big<GGML_TYPE_IQ4_XS>(down, w, jobs, njobs, njobs_cap, in, (float*) out, s); break;
        case T_IQ3_S: moe_launch_big<GGML_TYPE_IQ3_S>(down, w, jobs, njobs, njobs_cap, in, (float*) out, s); break;
        case T_IQ3_XXS: moe_launch_big<GGML_TYPE_IQ3_XXS>(down, w, jobs, njobs, njobs_cap, in, (float*) out, s); break;
        case T_IQ2_S: moe_launch_big<GGML_TYPE_IQ2_S>(down, w, jobs, njobs, njobs_cap, in, (float*) out, s); break;
        default: throw std::runtime_error("moe: expert type " + std::to_string(A.type) + " not handled");
    }
    ck_launch(down ? "moe_down (big jobs)" : "moe_gate_up (big jobs)");
}
}  // namespace

void moe_gate_up_big(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const void* xq, float* gu, void* st) {
    moe_mm_big(false, A, sa, jobs, njobs, njobs_cap, xq, gu, st);
}
void moe_down_big(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const float* gu, float* part, void* st) {
    moe_mm_big(true, A, sa, jobs, njobs, njobs_cap, gu, part, st);
}

void moe_gate_up_jobs(const DW& A, size_t sa, const Job* jobs, const int* njobs, int njobs_cap, const void* xq, float* gu, void* st) {
    moe_mm(false, A, sa, jobs, njobs, njobs_cap, xq, gu, st, MAXT);
}
void moe_down(const DW& A, size_t sa, const Job* jobs, const int* njobs, int njobs_max, const float* gu, float* part, void* st) {
    moe_mm(true, A, sa, jobs, njobs, njobs_max, gu, part, st);
}
void swiglu_stacked_q8_1(const float* gu, void* hq, int ffs, int T, void* st) {
    int8_t* tq;
    float* td;
    twin_of(hq, ffs, tq, td);
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, T, (ffs / 32 + 7) / 8) *
                                  sycl::range<3>(1, 1, 256),
                              sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                k_swiglu_stacked(gu, (block_q8_1 *)hq, ffs, tq, td);
            });
    ck_launch("swiglu_stacked_q8_1");
}
#endif
}  // namespace fnx

namespace fnx {
#if FNX_KPART == 6
namespace {
template <int R, bool SPLIT, bool SEG>
void q8s_launch(const Q8Segs &g, int rows, int cols, const block_q8_1 *y,
                const int8_t *tq, const float *td, int T, dpct::queue_ptr s) {
    const dpct::dim3 grid((rows + Q8S_THREADS / 32 * R - 1) /
                          (Q8S_THREADS / 32 * R));
#define FNX_Q8S(t)                                                             \
    strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {                                        \
        auto g_ct0 = g;                                                        \
        auto rows_ct1 = rows;                                                  \
        auto cols_ct2 = cols;                                                  \
        auto y_ct3 = y;                                                        \
        auto tq_ct4 = tq;                                                      \
        auto td_ct5 = td;                                                      \
                                                                               \
        cgh.parallel_for(                                                      \
            sycl::nd_range<3>(grid * sycl::range<3>(1, 1, Q8S_THREADS),        \
                              sycl::range<3>(1, 1, Q8S_THREADS)),              \
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] { \
                k_q8s<t, R, SPLIT, SEG>(g_ct0, rows_ct1, cols_ct2, y_ct3,      \
                                        tq_ct4, td_ct5);                       \
            });                                                                \
    });
    switch (T) {
        case 1: FNX_Q8S(1); break;
        case 2: FNX_Q8S(2); break;
        case 3: FNX_Q8S(3); break;
        case 4: FNX_Q8S(4); break;
        case 5: FNX_Q8S(5); break;
        case 6: FNX_Q8S(6); break;
        case 7: FNX_Q8S(7); break;
        case 8: FNX_Q8S(8); break;
        case 9: FNX_Q8S(9); break;
        case 10: FNX_Q8S(10); break;
        case 11: FNX_Q8S(11); break;
        case 12: FNX_Q8S(12); break;
        case 13: FNX_Q8S(13); break;
        case 14: FNX_Q8S(14); break;
        case 15: FNX_Q8S(15); break;
        case 16: FNX_Q8S(16); break;
        default: throw std::runtime_error("q8s_gemv: T out of range");
    }
#undef FNX_Q8S
}
template <int R, bool SEG>
void q8s_launch(const Q8Segs &g, int rows, int cols, const block_q8_1 *y, int T,
                dpct::queue_ptr s) {
    int8_t* tq;
    float* td;
    twin_of(y, cols, tq, td);
    if (tq) q8s_launch<R, true, SEG>(g, rows, cols, y, tq, td, T, s);
    else q8s_launch<R, false, SEG>(g, rows, cols, y, nullptr, nullptr, T, s);
}
template <bool SEG>
void q8s_run(const Q8Segs& g, int rows, int cols, const void* y, int T, void* st) {
    if (cols % 16) throw std::runtime_error("q8s_gemv: columns must be a multiple of 16");
    const block_q8_1* yq = (const block_q8_1*) y;
    dpct::queue_ptr s = (dpct::queue_ptr)st;
    // rows per warp: four for big matrices (activation loads shared by more rows), one when that leaves few blocks. The
    // segmented variant only for its one use, q/k/v (8192 rows and more): fewer kernels to compile
    if constexpr (SEG) {
        if (rows < 8192) throw std::runtime_error("q8s_gemv_segs: compiled for 8192 rows or more (q, k and v)");
        q8s_launch<4, SEG>(g, rows, cols, yq, T, s);
    } else {
#if FNX_27B
        // up to 4 rows by the window (the 27B's windows, bin/s27_run S27_BENCH=1: 1 row 19.4 -> 18.3 ms, 4 rows
        // 24.5 -> 22.3), past that by the matrix (5 to 16 rows: choosing by the window was no faster)
        if (T <= 2) q8s_launch<2, SEG>(g, rows, cols, yq, T, s);
        else if (T <= 4) q8s_launch<1, SEG>(g, rows, cols, yq, T, s);
        else if (rows >= 8192) q8s_launch<4, SEG>(g, rows, cols, yq, T, s);
#else
        if (rows >= 8192) q8s_launch<4, SEG>(g, rows, cols, yq, T, s);
#endif
        else if (rows >= 4096) q8s_launch<2, SEG>(g, rows, cols, yq, T, s);
        else q8s_launch<1, SEG>(g, rows, cols, yq, T, s);
    }
    ck_launch("q8s_gemv");
}
}  // namespace

void q8s_gemv(const Q8S& w, int rows, int cols, const void* y, float* out, int out_stride, int T, void* st) {
    Q8Segs g{};
    for (int s = 0; s < Q8S_SEGS; ++s) g.q[s] = w.q, g.d[s] = w.d, g.out[s] = out, g.end[s] = rows, g.stride[s] = out_stride;
    q8s_run<false>(g, rows, cols, y, T, st);
}

void q8s_gemv_segs(const Q8S* w, const int* rows, float* const* out, const int* out_stride, int n, int cols, const void* y, int T,
                   void* st) {
    if (n < 1 || n > Q8S_SEGS) throw std::runtime_error("q8s_gemv_segs: 1.." + std::to_string(Q8S_SEGS) + " matrices");
    Q8Segs g{};
    int end = 0;
    for (int s = 0; s < Q8S_SEGS; ++s) {
        const int k = std::min(s, n - 1);   // past n: the last segment again (no row reaches it)
        if (s < n) end += rows[s];
        g.q[s] = w[k].q, g.d[s] = w[k].d, g.out[s] = out[k], g.end[s] = end, g.stride[s] = out_stride[k];
    }
    q8s_run<true>(g, end, cols, y, T, st);
}

#endif
#if FNX_KPART == 7
namespace {
template <int R, bool SPLIT>
void q8s_ab_launch(const Q8Segs &g, int rows, int cols, const block_q8_1 *y,
                   const int8_t *tq, const float *td, const float *W,
                   const float *x, float *yf, int rf, int T,
                   dpct::queue_ptr s) {
    const dpct::dim3 grid(rf + (rows + Q8S_THREADS / 32 * R - 1) /
                                   (Q8S_THREADS / 32 * R));
#define FNX_QA(t)                                                              \
    strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {                                        \
        /*                                                                     \
        DPCT1101:780: 'WPR' expression was replaced with a value. Modify the   \
        code to use the original expression, provided in comments, if it is    \
        correct.                                                               \
        */                                                                     \
        sycl::local_accessor<float[t][4 /*WPR*/], 0> red_acc_ct1(cgh);         \
                                                                               \
        auto g_ct0 = g;                                                        \
        auto rows_ct1 = rows;                                                  \
        auto cols_ct2 = cols;                                                  \
        auto y_ct3 = y;                                                        \
        auto tq_ct4 = tq;                                                      \
        auto td_ct5 = td;                                                      \
        auto W_ct6 = W;                                                        \
        auto x_ct7 = x;                                                        \
        auto yf_ct8 = yf;                                                      \
        auto rf_ct9 = rf;                                                      \
                                                                               \
        cgh.parallel_for(                                                      \
            sycl::nd_range<3>(grid * sycl::range<3>(1, 1, Q8S_THREADS),        \
                              sycl::range<3>(1, 1, Q8S_THREADS)),              \
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] { \
                k_q8s_ab<t, R, SPLIT>(g_ct0, rows_ct1, cols_ct2, y_ct3,        \
                                      tq_ct4, td_ct5, W_ct6, x_ct7, yf_ct8,    \
                                      rf_ct9, red_acc_ct1);                    \
            });                                                                \
    });
    switch (T) {
        case 1: FNX_QA(1); break;
        case 2: FNX_QA(2); break;
        case 3: FNX_QA(3); break;
        case 4: FNX_QA(4); break;
        case 5: FNX_QA(5); break;
        case 6: FNX_QA(6); break;
        case 7: FNX_QA(7); break;
        case 8: FNX_QA(8); break;
        case 9: FNX_QA(9); break;
        case 10: FNX_QA(10); break;
        case 11: FNX_QA(11); break;
        case 12: FNX_QA(12); break;
        case 13: FNX_QA(13); break;
        case 14: FNX_QA(14); break;
        case 15: FNX_QA(15); break;
        case 16: FNX_QA(16); break;
        default: throw std::runtime_error("q8s_gemv_f32: T out of range");
    }
#undef FNX_QA
}
template <int R>
void q8s_ab_launch(const Q8Segs &g, int rows, int cols, const block_q8_1 *y,
                   const float *W, const float *x, float *yf, int rf, int T,
                   dpct::queue_ptr s) {
    int8_t* tq;
    float* td;
    twin_of(y, cols, tq, td);
    if (tq) q8s_ab_launch<R, true>(g, rows, cols, y, tq, td, W, x, yf, rf, T, s);
    else q8s_ab_launch<R, false>(g, rows, cols, y, nullptr, nullptr, W, x, yf, rf, T, s);
}
}  // namespace

void q8s_gemv_f32(const Q8S& w, int rows, int cols, const void* y, float* out, const float* W, const float* x, float* yf, int rf, int T,
                  void* st) {
    if (cols % 16) throw std::runtime_error("q8s_gemv_f32: columns must be a multiple of 16");
    if (rf < 1 || rf > 160) throw std::runtime_error("q8s_gemv_f32: the f32 matrix takes 1..160 rows (a block a row)");
    Q8Segs g{};
    for (int s = 0; s < Q8S_SEGS; ++s) g.q[s] = w.q, g.d[s] = w.d, g.out[s] = out, g.end[s] = rows, g.stride[s] = rows;
    const block_q8_1* yq = (const block_q8_1*) y;
    dpct::queue_ptr s = (dpct::queue_ptr)st;
    // compiled for its one use, qkvz, as q8s_gemv alone would run them: 4096 to 8191 rows two a warp; the 27B's 8192
    // and more (on two cards) four
#if FNX_27B
    if (rows < 4096) throw std::runtime_error("q8s_gemv_f32: compiled for 4096 rows or more (qkvz)");
    if (rows >= 8192) q8s_ab_launch<4>(g, rows, cols, yq, W, x, yf, rf, T, s);
    else q8s_ab_launch<2>(g, rows, cols, yq, W, x, yf, rf, T, s);
#else
    if (rows < 4096 || rows >= 8192) throw std::runtime_error("q8s_gemv_f32: compiled for 4096..8191 rows (qkvz)");
    q8s_ab_launch<2>(g, rows, cols, yq, W, x, yf, rf, T, s);
#endif
    ck_launch("q8s_gemv_f32");
}
#endif
}  // namespace fnx

#if FNX_KPART == 0
namespace fnx {
namespace {
// one load per 32-byte sector (L2 fills sectors, not whole lines)
void k_l2_prefetch(const char* p, size_t sectors, unsigned* sink) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    unsigned acc = 0;
    for (size_t i =
             item_ct1.get_group(2) * (size_t)item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < sectors; i += (size_t)item_ct1.get_group_range(2) *
                           item_ct1.get_local_range(2)) {
        unsigned v;
#if FNX_HIP
        v = __ldcg(reinterpret_cast<const unsigned*>(p + i * 32));
#else
        v = *((uint32_t *)(uintptr_t)(p + i * 32));
#endif
        acc ^= v;
    }
    if (acc == 0x9e3779b9u) *sink = acc;   // keeps the loads; never true in practice
}
}  // namespace
void l2_prefetch(const void *p, size_t bytes, void *st) try {
    const size_t sectors = (bytes + 31) / 32;
    static unsigned* sink = nullptr;
    if (!sink && DPCT_CHECK_ERROR(sink = (unsigned int *)sycl_port::alloc_device(
                                      4, dpct::get_in_order_queue())) !=
                     0) throw std::runtime_error("l2_prefetch: sink");
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
        auto sink_ct2 = sink;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1,
                                             (unsigned)std::min<size_t>(
                                                 160, (sectors + 255) / 256)) *
                                  sycl::range<3>(1, 1, 256),
                              sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) {
                k_l2_prefetch((const char *)p, sectors, sink_ct2);
            });
    });
    ck_launch("l2_prefetch");
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
}  // namespace fnx

namespace fnx {
namespace {
// y[r] = x[r] / sqrt(mean(x[r]^2) + eps) * w, a block per row
void k_rms_norm_rows(const float* x, const float* w, float* y, int n, float eps,
                     float *sh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const float *xr = x + (size_t)item_ct1.get_group(2) * n;
    float s = 0.f;
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) s += xr[i] * xr[i];
    const float r = sycl::rsqrt(block_sum(s, sh) / (float)n + eps);
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2))
        y[(size_t)item_ct1.get_group(2) * n + i] = xr[i] * r * w[i];
}
// R[t][c][i] = h[t][c][i] + e[t][i]
void k_add_streams(const float* h, const float* e, float* R) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1),
              j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (j >= D) return;
    R[(size_t) t * D + j] = h[(size_t) t * D + j] + e[(size_t) t * N + j % N];
}
}  // namespace

void rms_norm_rows(const float* x, const float* w, float* y, int rows, int n, float eps, void* st) {
    
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sh_acc_ct1(sycl::range<1>(32), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, rows) *
                                  sycl::range<3>(1, 1, 1024),
                              sycl::range<3>(1, 1, 1024)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                k_rms_norm_rows(
                    x, w, y, n, eps,
                    sh_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    ck_launch("rms_norm_rows");
}
void add_streams(const float* h, const float* e, float* R, int T, void* st) {
    strata::kprof::kp((dpct::queue_ptr)st, __FILE__, __func__, __LINE__)->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, T, (D + 255) / 256) *
                                             sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                           k_add_streams(h, e, R);
                       });
    ck_launch("add_streams");
}
}  // namespace fnx
#endif
