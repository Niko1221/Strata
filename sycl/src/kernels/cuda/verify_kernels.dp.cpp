// src/kernels/cuda/verify_kernels.cu - see include/strata/kernels/verify_kernels.hpp.
//
// The per-token arithmetic of every kernel here is transcribed from its single-token original (fused_gdn.cu,
// elementwise.cu) with the same operation order, so a verify window reproduces plain decode bit for bit.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/sycl_doorbell.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/resident_plan_mirror.hpp"
#include "strata/kernels/dp4a.hpp"

#include <cstdio>
#include <cstdlib>

#ifndef STRATA_PLAN_LOCAL
#define STRATA_PLAN_LOCAL 1   // 0: the original one-thread plan kernel (A/B)
#endif
namespace strata::kernels {
namespace {

constexpr int S = 128;          // GDN state size
constexpr int RG = 4;
constexpr int RPG = S / RG;

void check(const char* what) {
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
}

__dpct_inline__ void gdn_conv_l2_multi_kernel(const float *__restrict__ hist,
                                              const float *__restrict__ qkv,
                                              const float *__restrict__ w,
                                              float *__restrict__ h, int C,
                                              int qk_heads, float eps,
                                              int t_begin) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S / 32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = t_begin + item_ct1.get_group(1);
    const int c = item_ct1.get_group(2) * S + item_ct1.get_local_id(2);
    // the window of token t: [hist0, hist1, hist2, x_0, ..., x_t], its last four entries
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = t + j;          // index into [hist(3) | x...]
        win[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t) t * C + c];
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    float y = sum / (1.0f + sycl::native::exp(-sum));
    if ((int)item_ct1.get_group(2) < qk_heads) {
        float sq = y * y;
        /*
DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sq +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                sq, o);
        if ((item_ct1.get_local_id(2) & 31) == 0)
            part[item_ct1.get_local_id(2) >> 5] = sq;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= sycl::rsqrt(ss + eps);
    }
    h[(size_t) t * C + c] = y;
}

__dpct_inline__ void
gdn_conv_commit_kernel(float *__restrict__ hist, const float *__restrict__ qkv,
                       int C, const int32_t *__restrict__ n_keep) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= C) return;
    const int n = *n_keep;
    if (n <= 0) return;
    float seq[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = n + j;          // the last three of [hist(3) | x_0..x_{n-1}]
        seq[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    hist[c * 3] = seq[0];
    hist[c * 3 + 1] = seq[1];
    hist[c * 3 + 2] = seq[2];
}

/*
DPCT1110: The total declared local variable size in device function
gdn_ab_multi_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void gdn_ab_multi_kernel(
    const float *__restrict__ x, const uint16_t *__restrict__ wa,
    const uint16_t *__restrict__ wb, const float *__restrict__ dt,
    const float *__restrict__ ssm_a, float *__restrict__ gate,
    float *__restrict__ beta, int n, int h_v, int T) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * 8 + (item_ct1.get_local_id(2) >> 5),
              lane = item_ct1.get_local_id(2) & 31;
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const sycl::uint4 *w4 = reinterpret_cast<const sycl::uint4 *>(
        (is_beta ? wb : wa) + (size_t)r * n);
    float acc[kVerifyMaxT];
#pragma unroll
    for (int t = 0; t < kVerifyMaxT; ++t) acc[t] = 0.0f;
    for (int j = lane; j < n / 8; j += 32) {
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const sycl::uint4 wv = *(w4 + j);
#pragma unroll
        for (int t = 0; t < kVerifyMaxT; ++t) {
            if (t >= T) break;
            const float* xt = x + (size_t) t * n;
            const sycl::float4 xa =
                *reinterpret_cast<const sycl::float4 *>(xt + j * 8);
            const sycl::float4 xb =
                *reinterpret_cast<const sycl::float4 *>(xt + j * 8 + 4);
            float a = acc[t];
            a = sycl::fma(sycl::bit_cast<float>(wv.x() << 16), (float)(xa.x()),
                          a);
                a = sycl::fma(sycl::bit_cast<float>(wv.x() & 0xffff0000u),
                              (float)(xa.y()), a);
            a = sycl::fma(sycl::bit_cast<float>(wv.y() << 16), (float)(xa.z()),
                          a);
                a = sycl::fma(sycl::bit_cast<float>(wv.y() & 0xffff0000u),
                              (float)(xa.w()), a);
            a = sycl::fma(sycl::bit_cast<float>(wv.z() << 16), (float)(xb.x()),
                          a);
                a = sycl::fma(sycl::bit_cast<float>(wv.z() & 0xffff0000u),
                              (float)(xb.y()), a);
            a = sycl::fma(sycl::bit_cast<float>(wv.w() << 16), (float)(xb.z()),
                          a);
                a = sycl::fma(sycl::bit_cast<float>(wv.w() & 0xffff0000u),
                              (float)(xb.w()), a);
            acc[t] = a;
        }
    }
#pragma unroll
    for (int t = 0; t < kVerifyMaxT; ++t) {
        if (t >= T) break;
        float a = acc[t];
        /*
DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                a, o);
        if (lane != 0) continue;
        if (is_beta) {
            beta[(size_t)t * h_v + r] = 1.0f / (1.0f + sycl::native::exp(-a));
        } else {
            const float v = a + dt[r];
            const float sp = v > 20.0f ? v : sycl::log1p(sycl::native::exp(v));
            gate[(size_t) t * h_v + r] = sp * ssm_a[r];
        }
    }
}

/*
DPCT1110: The total declared local variable size in device function
gdn_step_norm_multi_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
__dpct_inline__ void gdn_step_norm_multi_kernel(
    float *__restrict__ state, const float *__restrict__ hbuf, int C,
    const float *__restrict__ gate, const float *__restrict__ beta,
    const float *__restrict__ z, const float *__restrict__ gamma, float eps,
    float *__restrict__ y, int h_k, int h_v, int T,
    const int32_t *__restrict__ n_keep, int t_out_begin) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sk = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sq = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[RG][S]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &wsum = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[S * RG / 32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int head = item_ct1.get_group(2);
    const int col = item_ct1.get_local_id(2);
    const int rg = item_ct1.get_local_id(1);
    const int tid = rg * S + col;
    const int qh = head % h_k;
    const int qk = S * h_k;             // q at [0, qk), k at [qk, 2qk), v at [2qk, ...)
    const int value_dim = S * h_v;
    const int n = n_keep ? *n_keep : T;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); // the previous token is done with sk/sq/red/wsum
        if (tid < S) { sk[tid] = ht[qk + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
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
        const float g = sycl::native::exp(gate[(size_t)t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
        red[rg][col] = kv;
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
        const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = sycl::fma((float)g, s[r], sk[rg * RPG + r] * delta);
            o = sycl::fma(s[r], sq[rg * RPG + r], o);
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
        red[rg][col] = o;
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
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) *
                 sycl::rsqrt((float)S);
            sq_part = oc * oc;
        }
        if (t < t_out_begin) continue;   // a replayed token: its state update is needed, its output is not
        /*
DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
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
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = sycl::rsqrt(ss / (float)S + eps);
            const float zz = z[(size_t) t * value_dim + head * S + col];
            y[(size_t)t * value_dim + head * S + col] =
                oc * scale * gamma[col] *
                (1.0f / (1.0f + sycl::native::exp(-zz)));
        }
    }
    if (n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}

__dpct_inline__ void embedding_gather_dev_kernel(
    const uint8_t *__restrict__ codes, const float *__restrict__ scales,
    const float *__restrict__ offsets, const int32_t *__restrict__ tokens,
    int64_t n, int code_bits, int code_bias, int group_elems,
    unsigned long long row_codes, unsigned long long row_groups,
    float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    const unsigned long long token = (unsigned long long) tokens[t];
    const uint8_t* c = codes + token * row_codes;
    const float* sc = scales + token * row_groups;
    const float* of = offsets ? offsets + token * row_groups : nullptr;
    const int per_byte = 8 / code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    const int code = (c[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const int64_t group = i / group_elems;
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    const float product = (float)(code + code_bias) * sc[group];
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    out[(size_t)t * n + i] = product + (of ? of[group] : 0.0f);
}

__dpct_inline__ void broadcast_streams_kernel(const float *__restrict__ x,
                                              float *__restrict__ R, int64_t n,
                                              int hc) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = x[(size_t) t * n + i % n];
}

__dpct_inline__ void copy_indexed_kernel(float *__restrict__ dst,
                                         const float *__restrict__ src,
                                         int64_t stride,
                                         const int32_t *__restrict__ index,
                                         int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int idx = *index;
    if (idx < 0) return;
#pragma unroll
    for (int64_t i =
             (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < n; i += (int64_t)item_ct1.get_group_range(2) *
                     item_ct1.get_local_range(2))
        dst[i] = src[(size_t) idx * stride + i];
}

__dpct_inline__ void
fetch_blobs_kernel(const unsigned long long *__restrict__ src,
                   const int32_t *__restrict__ n, sycl::uint4 *__restrict__ dst,
                   long long per) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long total = (long long)*n * per;
    for (long long i =
             (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < total; i += (long long)item_ct1.get_group_range(2) *
                         item_ct1.get_local_range(2)) {
        const long long k = i / per, off = i - k * per;
        dst[i] = ((const sycl::uint4 *)src[k])[off];
    }
}

__dpct_inline__ void rebase_ptrs_kernel(unsigned long long *ptr,
                                        const int32_t *n,
                                        unsigned long long base,
                                        long long bytes) {
    const int k =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (k < *n) ptr[k] = base + (unsigned long long) k * (unsigned long long) bytes;
}

__dpct_inline__ void add_streams_broadcast_kernel(const float *__restrict__ h,
                                                  const float *__restrict__ e,
                                                  float *__restrict__ R,
                                                  int64_t n, int hc) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = h[(size_t) t * n * hc + i] + e[(size_t) t * n + i % n];
}

__dpct_inline__ void ident_hits_kernel(const int32_t *__restrict__ ids, int n,
                                       int32_t *__restrict__ slot,
                                       int32_t *__restrict__ dst,
                                       int32_t *__restrict__ count) {
    const int i =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (i < n) { slot[i] = ids[i]; dst[i] = i; }
    if (i == 0) *count = n;
}

// E = the widest element the row size divides into (16, 4 or 1 bytes): a Q6_K head row of 2560 values is 2100 bytes
template <typename E>
__dpct_inline__ void gather_rows_kernel(const E *__restrict__ src,
                                        long long row_e,
                                        const int32_t *__restrict__ ids,
                                        long long n, E *__restrict__ dst) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long total = n * row_e;
    for (long long i =
             (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < total; i += (long long)item_ct1.get_group_range(2) *
                         item_ct1.get_local_range(2)) {
        const long long r = i / row_e, o = i - r * row_e;
        dst[i] = src[(long long) ids[r] * row_e + o];
    }
}

__dpct_inline__ void map_ids_kernel(int32_t *ids,
                                    const int32_t *__restrict__ table, int n) {
    const int i =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (i < n) ids[i] = table[ids[i]];
}

__dpct_inline__ void row_top_prob_kernel(const float *__restrict__ logits,
                                         int n_vocab,
                                         const int32_t *__restrict__ ids,
                                         float *__restrict__ probs) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_group(2);
    const float* l = logits + (size_t) t * n_vocab;
    const float m = l[ids[t]];
    float s = 0.0f;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n_vocab;
         i += item_ct1.get_local_range(2)) s += sycl::native::exp(l[i] - m);
    /*
DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), s,
            o);
    if ((item_ct1.get_local_id(2) & 31) == 0)
        part[item_ct1.get_local_id(2) >> 5] = s;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (item_ct1.get_local_id(2) == 0) {
        float tot = 0.0f;
#pragma unroll
        for (int w = 0; w < (int)(item_ct1.get_local_range(2) >> 5); ++w) tot +=
            part[w];
        probs[t] = 1.0f / tot;
    }
}

__dpct_inline__ void
mtp_select_kernel(const float *__restrict__ R_src, int64_t stride,
                  const int32_t *__restrict__ ids,
                  const int32_t *__restrict__ row_dev,
                  float *__restrict__ R_dst, int32_t *__restrict__ tok_dst,
                  int32_t *out, int j, const float *probs, float *out_p) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = *row_dev;
#pragma unroll
    for (int64_t i =
             (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < stride; i += (int64_t)item_ct1.get_group_range(2) *
                          item_ct1.get_local_range(2))
        R_dst[i] = R_src[(size_t) row * stride + i];
    if (item_ct1.get_group(2) == 0 && item_ct1.get_local_id(2) == 0) {
        const int32_t tok = ids[row];
        *tok_dst = tok;
        if (out != nullptr) ((volatile int32_t*) out)[j] = tok;
        if (probs != nullptr && out_p != nullptr) ((volatile float*) out_p)[j] = probs[row];
    }
}

__dpct_inline__ void dense_steps_kernel(const int32_t *__restrict__ cells,
                                        int n, int32_t *__restrict__ steps) {
    const int i =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (i >= n) return;
    const int c = cells[i];
    steps[i * 4 + 0] = c;
    steps[i * 4 + 1] = c + 1;
    steps[i * 4 + 2] = (c + 1) / 4;
    steps[i * 4 + 3] = c + 1;
}

}  // namespace

void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream) {
    if (cap <= 0) return;
    if (blob_bytes % 16 != 0) { std::fprintf(stderr, "fetch_blobs: blob size must be a multiple of 16\n"); std::exit(1); }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {
                auto blob_bytes_ct3 = (long long)(blob_bytes / 16);

                cgh.parallel_for<
                    dpct_kernel_name<class fetch_blobs_kernel_14b6f5>>(
                    sycl::nd_range<3>(sycl::range(1, 1, 48 * 8) *
                                          sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        fetch_blobs_kernel(src, n, (sycl::uint4 *)dst,
                                           blob_bytes_ct3);
                    });
            });
    }
    check("fetch_blobs");
}

void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class rebase_ptrs_kernel_436f27>>(
                sycl::nd_range<3>(sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    rebase_ptrs_kernel(ptr, n, (unsigned long long)base,
                                       (long long)blob_bytes);
                });
    }
    check("rebase_ptrs");
}

void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class add_streams_broadcast_kernel_7de8f5>>(
                sycl::nd_range<3>(
                    sycl::range(1, (unsigned)n_tok,
                                (unsigned)((n_embd * hc + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    add_streams_broadcast_kernel(h, e, R, n_embd, hc);
                });
    }
    check("add_streams_broadcast");
}

void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 1024) { std::fprintf(stderr, "ident_hits: n out of range\n"); std::exit(1); }
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class ident_hits_kernel_287489>>(
                sycl::nd_range<3>(sycl::range(1, 1, 1024),
                                  sycl::range(1, 1, 1024)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    ident_hits_kernel(ids, n, slot, dst, count);
                });
    }
    check("ident_hits");
}

void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class mtp_select_kernel_4611fe>>(
                sycl::nd_range<3>(sycl::range(1, 1, 16) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    mtp_select_kernel(R_src, R_stride, ids, row_dev, R_dst,
                                      tok_dst, out, j, probs, out_p);
                });
    }
    check("mtp_select");
}

void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream) {
    dpct::queue_ptr s = strata::q_of(stream);
    if (row_bytes % 16 == 0)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            long long row_bytes_ct1 = row_bytes / 16;

            cgh.parallel_for<
                dpct_kernel_name<class gather_rows_kernel_d4a04b, sycl::uint4>>(
                sycl::nd_range<3>(sycl::range(1, 1, 48 * 8) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    gather_rows_kernel((const sycl::uint4 *)src, row_bytes_ct1,
                                       ids, n, (sycl::uint4 *)dst);
                });
        });
    } else if (row_bytes % 4 == 0)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            long long row_bytes_ct1 = row_bytes / 4;

            cgh.parallel_for<
                dpct_kernel_name<class gather_rows_kernel_7960b2, uint32_t>>(
                sycl::nd_range<3>(sycl::range(1, 1, 48 * 8) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    gather_rows_kernel((const uint32_t *)src, row_bytes_ct1,
                                       ids, n, (uint32_t *)dst);
                });
        });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->parallel_for<
            dpct_kernel_name<class gather_rows_kernel_6148af, uint8_t>>(
            sycl::nd_range<3>(sycl::range(1, 1, 48 * 8) *
                                  sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                gather_rows_kernel(src, row_bytes, ids, n, dst);
            });
    }
    check("gather_rows");
}

void map_ids(int32_t* ids, const int32_t* table, int n, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class map_ids_kernel_43a113>>(
                sycl::nd_range<3>(sycl::range(1, 1, 64), sycl::range(1, 1, 64)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    map_ids_kernel(ids, table, n);
                });
    }
    check("map_ids");
}

void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream) {
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class row_top_prob_kernel_6adbe2>>(
                sycl::nd_range<3>(sycl::range(1, 1, n_rows) *
                                      sycl::range(1, 1, 1024),
                                  sycl::range(1, 1, 1024)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        row_top_prob_kernel(logits, n_vocab, ids, probs);
                    });
    }
    check("row_top_prob");
}

namespace {
__dpct_inline__ void window_ids_kernel(int32_t *steps, int window, int32_t *ids,
                                       long long stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int q = item_ct1.get_group(1);
    int32_t* st = steps + q * 4;
    const int n_kv = st[1];
    const int start = n_kv > window ? n_kv - window : 0;
    const int width = n_kv - start;
#pragma unroll
    for (int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         j < width;
         j += item_ct1.get_group_range(2) * item_ct1.get_local_range(2))
        ids[q * stride + j] = start + j;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_group(2) == 0 && item_ct1.get_local_id(2) == 0) st[3] =
        width;
}
}  // namespace

void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class window_ids_kernel_581536>>(
                sycl::nd_range<3>(sycl::range(1, (unsigned)n, 8) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    window_ids_kernel(steps, window, ids,
                                      (long long)ids_stride);
                });
    }
    check("window_ids");
}

void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class dense_steps_kernel_fd7245>>(
                sycl::nd_range<3>(sycl::range(1, 1, 64), sycl::range(1, 1, 64)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    dense_steps_kernel(cells, n, steps);
                });
    }
    check("dense_steps");
}

void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_conv_l2_multi: invalid arguments\n");
        std::exit(1);
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class gdn_conv_l2_multi_kernel_b3f737>>(
                sycl::nd_range<3>(
                    sycl::range(1, (unsigned)n_tok, (unsigned)(channels / S)) *
                        sycl::range(1, 1, S),
                    sycl::range(1, 1, S)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_conv_l2_multi_kernel(history, qkv, conv_w, h,
                                                 channels, qk_heads, eps,
                                                 t_begin);
                    });
    }
    check("gdn_conv_l2_multi");
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class gdn_conv_commit_kernel_dc1d7f>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((channels + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    gdn_conv_commit_kernel(history, qkv, channels, n_keep);
                });
    }
    check("gdn_conv_commit");
}

void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream) {
    if (n_embd % 8 != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_ab_multi: invalid arguments\n");
        std::exit(1);
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class gdn_ab_multi_kernel_34dcce>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((2 * h_v + 7) / 8)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_ab_multi_kernel(x, w_alpha, w_beta, dt, ssm_a, gate,
                                            beta, n_embd, h_v, n_tok);
                    });
    }
    check("gdn_ab_multi");
}

void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin) {
    if (!state || !h || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k || n_tok < 1 ||
        n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_step_norm_multi: invalid arguments\n");
        std::exit(1);
    }
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class gdn_step_norm_multi_kernel_42ae26>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)h_v) *
                                      sycl::range(1, RG, S),
                                  sycl::range(1, RG, S)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        gdn_step_norm_multi_kernel(
                            state, h, conv_channels, gate, beta, z, gamma, eps,
                            y, h_k, h_v, n_tok, n_keep, t_out_begin);
                    });
    }
    check("gdn_step_norm_multi");
}

namespace {
__dpct_inline__ void wait_flag_ge_kernel(const volatile uint32_t *flag,
                                         uint32_t value) {
    for (uint32_t spin = 0; spin < strata::kSpinMax && strata::sys_load(flag) < value; ++spin) strata_spin_pause();
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}
}  // namespace

namespace {
inline bool plan_parallel() {   // STRATA_PLAN_PARALLEL=0: thread 0 groups the entries alone (the old way)
    static const bool v = std::getenv("STRATA_PLAN_PARALLEL") == nullptr || std::atoi(std::getenv("STRATA_PLAN_PARALLEL")) != 0;
    return v;
}
__dpct_inline__ void resident_plan_kernel(
    const int32_t *__restrict__ ids, int n, int k,
    const int32_t *__restrict__ res, int n_expert, const uint8_t *cache_base,
    const unsigned long long *slot_off, long long blob,
    int32_t *__restrict__ pl, long long capx, uint32_t *skip, uint32_t ring,
    const unsigned long long *__restrict__ mir, bool par) {
#if STRATA_PLAN_LOCAL
    // SYCL port: the host's exact loop, but over a local copy of the ids and their slots. One thread reading global
    // memory for every compare (n^2 of them) took 87 us per layer on the B70 - 4% of a decode round; a work-group
    // of 64 loads the n <= 60 entries once, then thread 0 groups them from local memory.
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &s_ids = *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[64]>(item.get_group());
    auto &s_res = *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[64]>(item.get_group());
    auto &s_bad = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(item.get_group());
    auto &s_mir = *sycl::ext::oneapi::group_local_memory_for_overwrite<unsigned long long[64]>(item.get_group());
    const int tid = (int) item.get_local_id(2);
    if (tid == 0) s_bad = 0;
    item.barrier(sycl::access::fence_space::local_space);
    for (int i = tid; i < n && i < 64; i += 64) {
        const int32_t e = ids[i];
        const bool valid = e >= 0 && e < n_expert;
        const int32_t sl = valid ? res[e] : -1;
        const unsigned long long ma = (valid && sl < 0 && mir != nullptr) ? mir[e] : 0ull;   // mirrored in host memory
        s_ids[i] = e;
        s_res[i] = sl;
        s_mir[i] = ma;
        if (!valid || (sl < 0 && ma == 0)) s_bad = 1;
    }
    item.barrier(sycl::access::fence_space::local_space);
    if (s_bad || n > 64) { if (tid == 0) *skip = 0; return; }   // n > 64 never happens (kVerifyMaxT * 10 = 60); refuse rather than read past
    if (par) {
        // SYCL port: the grouping in parallel, one thread per entry (unitrace: thread 0 alone took 73 us per layer,
        // ~3.5 ms of a decode round). The plan is the host loop's exactly: groups in order of first occurrence,
        // entries in a group in index order. Entry i: L = its expert's first index; its group = the number of
        // first occurrences before L; the group starts at the number of entries whose leader is before L; its
        // place in the group = the number of earlier entries with the same leader.
        auto &s_lead = *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[64]>(item.get_group());
        if (tid < n) {
            int L = tid;
            for (int j = 0; j < tid; ++j) if (s_ids[j] == s_ids[tid]) { L = j; break; }
            s_lead[tid] = L;
        }
        item.barrier(sycl::access::fence_space::local_space);
        int32_t* counts = pl;
        int32_t* start = pl + 4;
        int32_t* dst = start + capx + 1;
        int32_t* tok = dst + capx;
        const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
        unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
        int32_t* start2 = pl + ptr_off + 4 * capx;
        if (tid < n) {
            const int L = s_lead[tid];
            int g = 0, gstart = 0, pos = 0;
            for (int j = 0; j < n; ++j) {
                const int Lj = s_lead[j];
                if (j < L && Lj == j) ++g;           // first occurrences before L
                if (Lj < L) ++gstart;                // entries of earlier groups
                if (j < tid && Lj == L) ++pos;       // earlier entries of this group
            }
            dst[gstart + pos] = tid;
            tok[gstart + pos] = tid / k;
            if (L == tid) {
                const int32_t slot = s_res[tid];
                ptr[g] = slot >= 0 ? (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob))
                                   : s_mir[tid];
                start[g] = gstart;
            }
        }
        item.barrier(sycl::access::fence_space::global_and_local);
        if (tid != 0) return;
        int groups = 0;
        for (int i = 0; i < n; ++i) groups += s_lead[i] == i;
        start[groups] = n;
        start2[0] = n;
        counts[0] = groups;
        counts[1] = n;
        counts[2] = 0;
        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
        strata::sys_store(skip, ring);
        return;
    }
    if (tid != 0) return;
#else
    // the original: one thread, the host's exact loop over global memory
    for (int i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || e >= n_expert || res[e] < 0) { *skip = 0; return; }
    }
#endif
#if !STRATA_PLAN_LOCAL
#define s_ids ids
#define S_RES(i) res[ids[i]]
#else
#define S_RES(i) s_res[i]
#endif
    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;
    int groups = 0, entries = 0;
    for (int i0 = 0; i0 < n; ++i0) {
        bool first = true;
        for (int j = 0; j < i0; ++j) if (s_ids[j] == s_ids[i0]) {
            first = false; break;
        }
        if (!first) continue;
        const int32_t slot = S_RES(i0);
#if STRATA_PLAN_LOCAL
        ptr[groups] = slot >= 0 ? (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob))
                                : s_mir[i0];   // not in VRAM: its pinned host mirror, read by the expert kernels over PCIe
#else
        ptr[groups] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
#endif
        start[groups] = entries;
        for (int i = i0; i < n; ++i)
            if (s_ids[i] == s_ids[i0]) {
                // an entry belongs to i0's group when its first occurrence is i0: the same expert id
                dst[entries] = i;
                tok[entries] = i / k;
                ++entries;
            }
        ++groups;
    }
    start[groups] = entries;
    start2[0] = entries;
    counts[0] = groups;
    counts[1] = entries;
    counts[2] = 0;
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    strata::sys_store(skip, ring);
#undef s_ids
#undef S_RES
}
__dpct_inline__ void wait_flag_ge_or_kernel(const volatile uint32_t *flag,
                                            uint32_t value,
                                            const volatile uint32_t *skip) {
    if (strata::sys_load(skip) == value) return;
    for (uint32_t spin = 0; spin < strata::kSpinMax && strata::sys_load(flag) < value; ++spin) strata_spin_pause();
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}
__dpct_inline__ void copy_i32_unless_kernel(int32_t *__restrict__ dst,
                                            const volatile int32_t *src, int n,
                                            const uint32_t *skip,
                                            uint32_t value) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (strata::sys_load(skip) == value) return;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) dst[i] = src[i];
}
/*
DPCT1052: SYCL does not support the member access for a volatile qualified
vector type. The volatile qualifier was removed. You may need to rewrite the
code.
*/
__dpct_inline__ void copy_or_zero_kernel(sycl::float4 *__restrict__ dst,
                                         const sycl::float4 *src, long long n4,
                                         const uint32_t *skip, uint32_t value) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const bool zero = *skip == value;
#pragma unroll
    for (long long i =
             (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < n4; i += (long long)item_ct1.get_group_range(2) *
                      item_ct1.get_local_range(2))
        dst[i] = zero ? sycl::float4(0.f, 0.f, 0.f, 0.f)
                      : const_cast<const sycl::float4 *>(src)[i];
}
}  // namespace

namespace {
const int32_t* g_mirror_res = nullptr;
const unsigned long long* g_mirror_table = nullptr;
}
void resident_plan_set_mirror(const int32_t* d_res, const unsigned long long* mirror_table) {
    g_mirror_res = d_res;
    g_mirror_table = mirror_table;
}
void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                   long long capx, uint32_t* skip, uint32_t ring, void* stream) {
    const unsigned long long* mir = nullptr;   // SYCL port: the layer's slice of the host-mirror table, if any
    if (g_mirror_table != nullptr && g_mirror_res != nullptr && res_layer >= g_mirror_res)
        mir = g_mirror_table + (res_layer - g_mirror_res);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        const bool par = plan_parallel();
        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class resident_plan_kernel_db6b7a>>(
                sycl::nd_range<3>(sycl::range(1, 1, STRATA_PLAN_LOCAL ? 64 : 1), sycl::range(1, 1, STRATA_PLAN_LOCAL ? 64 : 1)),
                exp_props, [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                    resident_plan_kernel(ids, n_entries, k, res_layer, n_expert,
                                         cache_base, slot_off, blob, plan, capx,
                                         skip, ring, mir, par);
                });
    }
    check("resident_plan");
}
void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class wait_flag_ge_or_kernel_2b2de3>>(
                sycl::nd_range<3>(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    wait_flag_ge_or_kernel(flag, value, skip);
                });
    }
    check("wait_flag_ge_or");
}
void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* stream) {
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class copy_i32_unless_kernel_d9d761>>(
                sycl::nd_range<3>(sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    copy_i32_unless_kernel(dst, (const volatile int32_t *)src,
                                           (int)n, skip, value);
                });
    }
    check("copy_i32_from_mapped_unless");
}
void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream) {
    if (n <= 0) return;
    const long long n4 = n / 4;
    const int blocks = (int) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class copy_or_zero_kernel_5ef081>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    copy_or_zero_kernel((sycl::float4 *)dst,
                                        (const sycl::float4 *)src, n4,
                                        skip, value);
                });
    }
    check("copy_or_zero_from_mapped");
}

void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class wait_flag_ge_kernel_d7debf>>(
                sycl::nd_range<3>(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    wait_flag_ge_kernel(flag, value);
                });
    }
    check("wait_flag_ge");
}

void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class embedding_gather_dev_kernel_600016>>(
                sycl::nd_range<3>(sycl::range(1, (unsigned)n_tok,
                                              (unsigned)((n + 255) / 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    embedding_gather_dev_kernel(
                        codes, scales, offsets, tokens, n, code_bits, code_bias,
                        group_elems, row_codes, row_groups, out);
                });
    }
    check("embedding_gather_dev");
}

void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class broadcast_streams_kernel_aa4f6f>>(
                sycl::nd_range<3>(
                    sycl::range(1, (unsigned)n_tok,
                                (unsigned)((n_embd * hc + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    broadcast_streams_kernel(x, R, n_embd, hc);
                });
    }
    check("broadcast_streams");
}

void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream) {
    const unsigned blocks = (unsigned) ((n + 255) / 256 < 64 ? (n + 255) / 256 : 64);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class copy_indexed_kernel_b12e23>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    copy_indexed_kernel(dst, src, stride, index, n);
                });
    }
    check("copy_indexed");
}

// a GPU timestamp (ns, %globaltimer) into buf[i] - the verify window's stage profiler
namespace {
    __dpct_inline__ void gpu_stamp_kernel(unsigned long long *buf, int i) {
    unsigned long long t;
#if defined(__HIPCC__)
    t = wall_clock64() * 10ull;   // gfx10.3 / gfx11 / gfx12: a constant 100 MHz counter, in ns
#else
    /*
    DPCT1053: Migration of device assembly code is not supported.
    */
    t = 0;   // SYCL: no %globaltimer equivalent; the stage profiler is inert on this backend
#endif
    buf[i] = t;
} }
void gpu_stamp(unsigned long long* buf, int i, void* stream) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    strata::q_of(stream)
        ->parallel_for<dpct_kernel_name<class gpu_stamp_kernel_76ad79>>(
            sycl::nd_range<3>(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                gpu_stamp_kernel(buf, i);
            });
}

}  // namespace strata::kernels
