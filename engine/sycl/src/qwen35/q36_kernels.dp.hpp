// SYCL counterpart of src/qwen35/q36_kernels.cuh. Preserve the q8_1 ABI,
// the 32-lane XOR trees and the reference reduction order. STRATA_Q36_VERIFY
// checks these fused paths against the shared SYCL kernels on real activations;
// cross-backend floating-point bit equality is not assumed here.
#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include "strata/sycl_queue.hpp"
#include <cstddef>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace q36k {

struct QB {   // a q8_1 activation block (what native_quantize_q8_1 writes)
    sycl::half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(QB) == 36, "q8_1 block");
static_assert(offsetof(QB, qs) == 4, "q8_1 values start after the half2 scale/sum");

__dpct_inline__ float wsum(float x) {
#pragma unroll
    /*
    DPCT1096:156: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    for (int o = 16; o > 0; o >>= 1) x += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), x, o);
    return x;
}
__dpct_inline__ float wmax(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x = sycl::fmax(
        x, dpct::permute_sub_group_by_xor(
               sycl::ext::oneapi::this_work_item::get_sub_group(), x, o));
    return x;
}
__dpct_inline__ float sigm(float v) {
    return 1.0f / (1.0f + sycl::native::exp(-v));
}

// Match the existing SYCL reference paths: finite q8_1 scales use ordinary
// float division; native router/SwiGLU use native::exp. CUDA PTX approximation
// instructions are intentionally not part of the Intel backend.
__dpct_inline__ float fdiv_fm(float a, float b) { return a / b; }
__dpct_inline__ float frcp_fm(float a) { return 1.0f / a; }
__dpct_inline__ float fexp_fm(float a) { return sycl::native::exp(a); }

/// One lane's element `v` of a 32-element q8_1 block (the warp is the block): native_quantize_q8_1_kernel's body.
/// `y` points at the row's blocks, `i` is the element index within the row.
__dpct_inline__ void q8_store(QB *__restrict__ y, int i, float v, int lane) {
    const float amax = wmax(sycl::fabs(v));
    const float sum = wsum(v);
    const float d = strata::kernels::q8_1_finite(fdiv_fm(amax, 127.0f));
    int8_t q = 0;   // strata::kernels::q8_1_quant with the division of the fast-math TU
    if (amax != 0.0f) {
        const float qf = sycl::round(fdiv_fm(v, d));
        q = (int8_t) (qf > 127.0f ? 127.0f : qf < -127.0f ? -127.0f : qf);
    }
    y[i >> 5].qs[lane] = q;
    if (lane == 0) y[i >> 5].ds = strata::kernels::q8_1_ds(d, sum);
}

/// A Q8_0 row against NC activation columns. Unpacked matrices use the current
/// SYCL native Wide32Q8 contract (16 integer products before scaling, two lanes
/// per block). Packed matrices retain their registered legacy contract (eight
/// integer products, four warp partials combined in order, then the XOR tree).
/// These layouts are mathematically equivalent but are not generally bitwise
/// interchangeable for non-dyadic half scales. res[c] is replicated in all lanes.
template <int NC, int NJ, bool PK>
/*
DPCT1110:9: The total declared local variable size in device function q8_row_dot
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void q8_row_dot(const uint8_t *__restrict__ w,
                                const sycl::half *__restrict__ dpl,
                                const QB *__restrict__ x, int row, int nb,
                                int lane, float (&res)[NC]) {
    if constexpr (!PK) {
        // The current SYCL native_q8_0_mmvq defaults to Wide32Q8: two
        // lanes per 32-value block, sixteen integer products per lane.
        // Applying the scale after only eight products (the CUDA layout
        // below) changes FP32 rounding with ordinary, non-dyadic scales.
        // Keep the same kb order and apply() expression as that real path.
#pragma unroll
        for (int c = 0; c < NC; ++c) res[c] = 0.0f;
        const int half = lane & 1;
        for (int kb = lane >> 1; kb < nb; kb += 16) {
            const size_t blk = (size_t)row * nb + kb;
            const uint16_t* b = reinterpret_cast<const uint16_t*>(w + blk * 34);
            const float d0 = sycl::vec<sycl::half, 1>(
                sycl::bit_cast<sycl::half, unsigned short>(b[0]))
                .convert<float, sycl::rounding_mode::automatic>()[0];
            const uint16_t* qs = b + 1 + half * 8;
            int v[4];
#pragma unroll
            for (int k = 0; k < 4; ++k)
                v[k] = (int)qs[k * 2] | ((int)qs[k * 2 + 1] << 16);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const QB* xb = x + (size_t)c * nb + kb;
                const int* u = reinterpret_cast<const int*>(xb->qs) + half * 4;
                int sumi = 0;
#pragma unroll
                for (int k = 0; k < 4; ++k)
                    sumi = STRATA_DP4A(v[k], u[k], sumi);
                const float d1 = xb->ds[0];
                // F::apply() in the legacy kernel returns a rounded float
                // before the caller adds it. Preserve that FP32 boundary;
                // otherwise inlining the fused loop can contract this multiply
                // with the previous block's accumulator into an FMA.
                volatile const float term = d0 * d1 * float(sumi);
                res[c] += term;
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) res[c] = wsum(res[c]);
        return;
    }
    const int kqs = 2 * (lane & 3);
    const int kb0 = lane >> 2;
    int v0[4][NJ], v1[4][NJ];
    float d0[4][NJ];
    bool ok[4][NJ];
#pragma unroll
    for (int p = 0; p < 4; ++p)
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int kbx = kb0 + 8 * p + 32 * j;
            ok[p][j] = kbx < nb;
            if (!ok[p][j]) continue;
            const size_t blk = (size_t) row * nb + kbx;
            if (PK) {
                const int* q = reinterpret_cast<const int*>(w + blk * 32) + kqs;
                v0[p][j] = q[0];
                v1[p][j] = q[1];
                d0[p][j] =
                    sycl::vec<sycl::half, 1>(dpl[blk])
                        .convert<float, sycl::rounding_mode::automatic>()[0];
            } else {
                const uint16_t* b = reinterpret_cast<const uint16_t*>(w + blk * 34);
                d0[p][j] =
                    sycl::vec<sycl::half, 1>(
                        sycl::bit_cast<sycl::half, unsigned short>(b[0]))
                        .convert<float, sycl::rounding_mode::automatic>()[0];
                v0[p][j] = (int) b[1 + 2 * kqs] | ((int) b[2 + 2 * kqs] << 16);
                v1[p][j] = (int) b[3 + 2 * kqs] | ((int) b[4 + 2 * kqs] << 16);
            }
        }
    float P[NC][4];
#pragma unroll
    for (int c = 0; c < NC; ++c)
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            float t = 0.f;
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                if (!ok[p][j]) continue;
                const QB* xb = x + (size_t) c * nb + kb0 + 8 * p + 32 * j;
                const int* u = reinterpret_cast<const int*>(xb->qs);
                int sumi = 0;
                sumi = STRATA_DP4A(v0[p][j], u[kqs], sumi);
                sumi = STRATA_DP4A(v1[p][j], u[kqs + 1], sumi);
                const float d1 = xb->ds[0];
                t += d0[p][j] * d1 * float(sumi);
            }
            P[c][p] = t;
        }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float t = P[c][0];
        t += P[c][1];
        t += P[c][2];
        t += P[c][3];
        res[c] = wsum(t);
    }
}

/// y[c][row] = W[row] . x[c], NC columns, 8 rows (warps) per 256-thread block.
template <int NC, int NJ, bool PK>
void k_q8_rows(const uint8_t *__restrict__ w,
               const sycl::half *__restrict__ dpl, const QB *__restrict__ x,
               float *__restrict__ y, int nb, int n_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int stride =
        item_ct1.get_group_range(2) *
        (item_ct1.get_local_range(2) >>
         5); // a grid-stride loop over the rows (see launch_q8_rows)
    for (int row =
             item_ct1.get_group(2) * (item_ct1.get_local_range(2) >> 5) + warp;
         row < n_out; row += stride) {
        float res[NC];
        q8_row_dot<NC, NJ, PK>(w, dpl, x, row, nb, lane, res);
        if (lane == 0)
#pragma unroll
            for (int c = 0; c < NC; ++c) y[(size_t) c * n_out + row] = res[c];
    }
}

template <int NC, int NJ, bool PK>
inline void launch_q8_rows_t(const void *w, const sycl::half *dpl,
                             const void *x, float *y, int nb, int n_out,
                             int warps, int max_warps, dpct::queue_ptr s) {
    auto* queue = strata::q_of(s);
    const int rows = n_out < max_warps ? n_out : max_warps;
    /*
    DPCT1049:10: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        dpct::has_capability_or_fail(queue->get_device(), {sycl::aspect::fp16});

        queue->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, (unsigned)((rows + warps - 1) / warps)) *
                    sycl::range<3>(1, 1, warps * 32),
                sycl::range<3>(1, 1, warps * 32)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                k_q8_rows<NC, NJ, PK>(static_cast<const uint8_t *>(w), dpl,
                                      static_cast<const QB *>(x), y, nb, n_out);
            });
    }
}

template <int NC, int NJ, bool PK>
/// At most 1024 warps in the grid, a grid-stride loop over the rows beyond: a
/// 2048-row product took 21 us on gfx1201 as 2048 warps and 5.2 us as 1024 (a
/// grid above 32768 threads is several times slower there; the same rows, the
/// same sums).
inline void launch_q8_rows(const void *w, const sycl::half *dpl, const void *x,
                           float *y, int nb, int n_out, dpct::queue_ptr s) {
    launch_q8_rows_t<NC, NJ, PK>(w, dpl, x, y, nb, n_out, 8, 1024, s);
}

/// The shared expert's gate and up products and SwiGLU + q8_1 in one launch: 32 rows (one q8_1 block of h) per 1024-thread
/// block, warp w computing gate row and up row 32 b + w, then warp 0 applying native_swiglu_quantize_q8_1_kernel's body to
/// the 32 pairs.  Column c of the h image is at hq + c * (n_out / 32).  n_out % 32 == 0.
template <int NC, int NJ, bool PK>
void k_sh_gu(const uint8_t *__restrict__ wg, const sycl::half *__restrict__ dg,
             const uint8_t *__restrict__ wu, const sycl::half *__restrict__ du,
             const QB *__restrict__ x, QB *__restrict__ hq, int nb, int n_out,
             float g[NC][32], float u[NC][32]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int row = item_ct1.get_group(2) * 32 + warp;
    float rg[NC], ru[NC];
    q8_row_dot<NC, NJ, PK>(wg, dg, x, row, nb, lane, rg);
    q8_row_dot<NC, NJ, PK>(wu, du, x, row, nb, lane, ru);
    if (lane == 0)
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            g[c][warp] = rg[c];
            u[c][warp] = ru[c];
        }
    /*
    DPCT1065:44: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (warp == 0) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float gi = g[c][lane];
            /*
            DPCT1013:43: The rounding mode could not be specified and the
            generated code may have different accuracy than the original code.
            Verify the correctness. SYCL math built-in function rounding mode is
            aligned with OpenCL C 1.2 standard.
            */
            const float xi = gi / (1.0f + sycl::native::exp(-gi)) * u[c][lane];
            q8_store(hq + (size_t)c * (n_out >> 5),
                     item_ct1.get_group(2) * 32 + lane, xi, lane);
        }
    }
}

/// k_add_rms (x += y; h = rms(x) * w, one block of 256 per row) and the q8_1 image of h in the same launch.
inline void k_add_rms_q(float* x, const float* y, const float* w, float* h, QB* hq, int n, float eps,
                                        float *sh) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    float *xr = x + (size_t)item_ct1.get_group(2) * n;
    const float *yr = y + (size_t)item_ct1.get_group(2) * n;
    float *hr = h + (size_t)item_ct1.get_group(2) * n;
    QB *qr = hq + (size_t)item_ct1.get_group(2) * (n >> 5);
    float s = 0.f;
    for (int i = item_ct1.get_local_id(2); i < n; i += 256) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
    sh[item_ct1.get_local_id(2)] = s;
    /*
    DPCT1065:45: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int o = 128; o > 0; o >>= 1) {
        if ((int)item_ct1.get_local_id(2) < o) sh[item_ct1.get_local_id(2)] +=
            sh[item_ct1.get_local_id(2) + o];
        /*
        DPCT1118:11: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:46: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    const float r = sycl::rsqrt(sh[0] / (float)n + eps);
    for (int i = item_ct1.get_local_id(2); i < n; i += 256) {
        /*
        DPCT1013:47: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        const float hv = xr[i] * r * w[i]; // no FMA with the quantizer's sums:
                                           // the old h was stored first
        hr[i] = hv;
        q8_store(
            qr, i, ((volatile float *)hr)[i],
            item_ct1.get_local_id(2) &
                31); // the stored value, as the standalone quantizer reads it
    }
}

/// k_moe_out for T <= 8 rows (decode, verify windows) with n <= 2048 and n % 256 == 0: 1024 threads per row instead of 256
/// (the elements' loads in parallel), the squares summed per old thread (element t, t + 256, ... in order) and the
/// 256-wide tree as before: the same bits.  x += sum_k w_k part_k (+ gate * shared); h = rms(x) * wn; with hq the q8_1
/// image of h as well (the next layer's input).
inline void k_moe_out_small(float* x, const float* parts, const int32_t* prow, const float* wts,
                                                        int K_, const float* shared, const float* sgate, int sg_stride,
                                                        const float* wn, float* h, int n, float eps,
                                                        float *vv, float *sh,
                                                        const float **pk, QB* hq = nullptr) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int t = item_ct1.get_group(2);
    float* xr = x + (size_t) t * n;
    if (item_ct1.get_local_id(2) < K_) pk[item_ct1.get_local_id(2)] =
        parts + (size_t)(prow ? prow[t * K_ + item_ct1.get_local_id(2)]
                              : t * K_ + item_ct1.get_local_id(2)) *
                    n;
    /*
    DPCT1065:48: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const float* wr = wts + (size_t) t * K_;
    const float gs = sigm(sgate[(size_t) t * sg_stride]);
    for (int i = item_ct1.get_local_id(2); i < n; i += 1024) {
        float acc = wr[0] * pk[0][i];
        for (int k = 1; k < K_; ++k)
            acc = sycl::fma((float)(wr[k]), (float)(pk[k][i]), acc);
        acc += gs * shared[(size_t) t * n + i];
        const float v = xr[i] + acc;
        xr[i] = v;
        vv[i] = v;
    }
    if (wn == nullptr) return;
    /*
    DPCT1065:49: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) < 256) {
        float s = 0.f;
        for (int i = item_ct1.get_local_id(2); i < n; i += 256) s +=
            vv[i] * vv[i];
        sh[item_ct1.get_local_id(2)] = s;
    }
    /*
    DPCT1065:50: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int s = 128; s > 0; s >>= 1) {
        if ((int)item_ct1.get_local_id(2) < s) sh[item_ct1.get_local_id(2)] +=
            sh[item_ct1.get_local_id(2) + s];
        /*
        DPCT1118:12: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:51: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    const float r = sycl::rsqrt(sh[0] / (float)n + eps);
    for (int i = item_ct1.get_local_id(2); i < n; i += 1024) {
        /*
        DPCT1013:52: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        const float hv = vv[i] * r * wn[i];
        h[(size_t) t * n + i] = hv;
        if (hq) q8_store(hq + (size_t)t * (n >> 5), i,
                         ((volatile float *)h)[(size_t)t * n + i],
                         item_ct1.get_local_id(2) & 31);
    }
}

/// k_group / k_group_hy for n_ent <= 64 entries: one block of 64 threads ranks each entry by (expert id, entry) in
/// parallel instead of one thread walking all the experts; the same group / entry arrays.  HY: only the entries whose
/// expert has a VRAM slot (slot_of >= 0), the group's blob at blobs + slot * blob.
template <bool HY>
void k_group_small(const int32_t* ids, int n_ent, int K_, const int32_t* slot_of, const uint8_t* blobs,
                              unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups,
                              int32_t* ent_dst, int32_t* ent_tok, int *sid) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int e = item_ct1.get_local_id(2);
    const int id = e < n_ent ? ids[e] : 0;
    const bool valid = e < n_ent && (!HY || slot_of[id] >= 0);
    int rank = 0;
    if (valid) {
        for (int e2 = 0; e2 < n_ent; ++e2) {
            const int id2 = ids[e2];
            if (HY && slot_of[id2] < 0) continue;
            if (id2 < id || (id2 == id && e2 < e)) ++rank;
        }
        ent_dst[rank] = e;
        ent_tok[rank] = e / K_;
        sid[rank] = id;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    int nv = 0;
    for (int e2 = 0; e2 < n_ent; ++e2) nv += (!HY || slot_of[ids[e2]] >= 0) ? 1 : 0;
    if (valid && (rank == 0 || sid[rank - 1] != sid[rank])) {
        int g = 0;
        for (int q = 1; q < rank; ++q) g += sid[q - 1] != sid[q] ? 1 : 0;
        g += rank > 0 ? 1 : 0;
        grp_ptr[g] = (unsigned long long) (blobs + (size_t) (HY ? slot_of[id] : id) * blob);
        grp_start[g] = rank;
    }
    if (e == 0) {
        int G = nv > 0 ? 1 : 0;
        for (int q = 1; q < nv; ++q) G += sid[q - 1] != sid[q] ? 1 : 0;
        grp_start[G] = nv;
        *n_groups = G;
    }
}

/// route_topk<256, 8> (native_router.cu, copied: the same expressions) for n_tok <= 8 tokens (one warp each) and, in the same
/// launch, k_group_small's grouping of the picks.  block (32, 8).  HY as k_group_small.
template <bool HY>
/*
DPCT1110:13: The total declared local variable size in device function
k_route_group exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void k_route_group(const float *__restrict__ logits, int32_t *__restrict__ ids,
                   float *__restrict__ weights, int n_tok, int ld,
                   const int32_t *slot_of, const uint8_t *blobs,
                   unsigned long long blob, unsigned long long *grp_ptr,
                   int32_t *grp_start, int32_t *n_groups, int32_t *ent_dst,
                   int32_t *ent_tok, int *sids, int *sid) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int NE = 256, K = 8, V = NE / 32;

    const int tk = item_ct1.get_local_id(1);
    const int lane = item_ct1.get_local_id(2);
    if (tk < n_tok) {
        const float* lg = logits + (size_t) tk * ld;
        float values[V];
#pragma unroll
        for (int i = 0; i < V; ++i) values[i] = lg[lane + i * 32];
        float maximum = -INFINITY;
#pragma unroll
        for (int i = 0; i < V; ++i) maximum = sycl::max(maximum, values[i]);
        maximum = wmax(maximum);
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < V; ++i) {
            values[i] = fexp_fm(values[i] - maximum);
            sum += values[i];
        }
        const float reciprocal = frcp_fm(wsum(sum));
#pragma unroll
        for (int i = 0; i < V; ++i) {
            values[i] *= reciprocal;
            if (sycl::isnan(values[i])) values[i] = -FLT_MAX;
        }
        float selected = 0.0f, selected_sum = 0.0f;
        for (int rank = 0; rank < K; ++rank) {
            float best = values[0];
            int expert = lane;
#pragma unroll
            for (int i = 1; i < V; ++i) {
                if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
            }
#pragma unroll
            for (int mask = 16; mask; mask >>= 1) {
                const float other = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), best,
                    mask);
                const int other_id = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), expert,
                    mask);
                if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
            }
            if ((expert & 31) == lane) {
                values[expert / 32] = -INFINITY;
                ids[(size_t) tk * K + rank] = expert;
                sids[tk * K + rank] = expert;
                selected_sum += best;
            }
            if (rank == lane) selected = best;
        }
        selected_sum = sycl::max(wsum(selected_sum), 6.103515625e-5f);
        const float inverse_selected_sum = frcp_fm(selected_sum);
        if (lane < K) weights[(size_t) tk * K + lane] = selected * inverse_selected_sum;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const int n_ent = n_tok * K;
    const int e = tk * 32 + lane;   // 0 .. 255: the entries are the first n_ent threads
    const int id = e < n_ent ? sids[e] : 0;
    const bool valid = e < n_ent && (!HY || slot_of[id] >= 0);
    int rank = 0;
    if (valid) {
        for (int e2 = 0; e2 < n_ent; ++e2) {
            const int id2 = sids[e2];
            if (HY && slot_of[id2] < 0) continue;
            if (id2 < id || (id2 == id && e2 < e)) ++rank;
        }
        ent_dst[rank] = e;
        ent_tok[rank] = e / K;
        sid[rank] = id;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    int nv = 0;
    for (int e2 = 0; e2 < n_ent; ++e2) nv += (!HY || slot_of[sids[e2]] >= 0) ? 1 : 0;
    if (valid && (rank == 0 || sid[rank - 1] != sid[rank])) {
        int g = 0;
        for (int q = 1; q < rank; ++q) g += sid[q - 1] != sid[q] ? 1 : 0;
        g += rank > 0 ? 1 : 0;
        grp_ptr[g] = (unsigned long long) (blobs + (size_t) (HY ? slot_of[id] : id) * blob);
        grp_start[g] = rank;
    }
    if (e == 0) {
        int G = nv > 0 ? 1 : 0;
        for (int q = 1; q < nv; ++q) G += sid[q - 1] != sid[q] ? 1 : 0;
        grp_start[G] = nv;
        *n_groups = G;
    }
}

}  // namespace q36k
