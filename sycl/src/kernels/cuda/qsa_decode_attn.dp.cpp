// src/kernels/cuda/qsa_decode_attn.cu - see include/strata/kernels/qsa_decode_attn.hpp.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace strata::kernels {
namespace {

constexpr int HD = 256;          // head_dim
constexpr int G = 12;            // query heads per KV head (24 / 2)
constexpr int CHUNK = 64;        // cells per block
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

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
__dpct_inline__ float warp_max(float v) {
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) v = sycl::fmax(
        v, dpct::experimental::permute_sub_group_by_xor(
               0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
               v, o));
    return v;
}

// 8 consecutive values of one cell's key or value row for KV head `kvh`, dimensions [d0, d0+8).
// Per-format bodies; `load8` below is the KV_MODE dispatcher. value=false is the K side, true the V side.
__dpct_inline__ void load8_f16(const QsaAttnPools &p, bool value, long long row,
                               int d0, float *out) {
    const uint16_t* base = (value ? p.v_pool : p.k_pool) + row * HD + d0;
    const sycl::uint4 raw = *reinterpret_cast<const sycl::uint4 *>(base);
    const sycl::half2 *h2 = reinterpret_cast<const sycl::half2 *>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const sycl::float2 f =
            h2[j].template convert<float, sycl::rounding_mode::automatic>();
        out[2 * j] = f.x();
        out[2 * j + 1] = f.y();
    }
}
__dpct_inline__ void load8_q8(const QsaAttnPools &p, bool value, long long row,
                              int d0, float *out) {
    const int8_t* codes = (value ? p.v_q : p.k_q) + row * HD + d0;
    const uint16_t sbits = (value ? p.v_scale : p.k_scale)[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP];
    const float sc = sycl::vec<sycl::half, 1>(
                         sycl::bit_cast<sycl::half, unsigned short>(sbits))
                         .convert<float, sycl::rounding_mode::automatic>()[0];
    const sycl::uint2 raw = *reinterpret_cast<const sycl::uint2 *>(codes);
    const int8_t* c = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = (float) c[j] * sc;
}
__dpct_inline__ void load8_q4(const QsaAttnPools &p, bool value, long long row,
                              int d0, float *out) {
    constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
    const int b = d0 / QK4_0;
    const int rem = d0 % QK4_0;
    const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((value ? p.v_q4 : p.k_q4) + row * bytes_per_head) + b;
    const float d = sycl::vec<sycl::half, 1>(
                        sycl::bit_cast<sycl::half, unsigned short>(blk->d))
                        .convert<float, sycl::rounding_mode::automatic>()[0];
    const int j = (rem == 0 || rem == 16) ? 0 : 8;
    const uint8_t* bytes = blk->qs + j;
    if (rem < 16) {
#pragma unroll
        for (int k = 0; k < 8; ++k) out[k] = (float) ((int)(bytes[k] & 0x0F) - 8) * d;
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) out[k] = (float) ((int)(bytes[k] >> 4) - 8) * d;
    }
}
template <int KV_MODE>
__dpct_inline__ void load8(const QsaAttnPools &p, bool value, long long row,
                           int d0, float *out) {
    if constexpr (KV_MODE == 0) load8_f16(p, value, row, d0, out);
    else if constexpr (KV_MODE == 1) load8_q8(p, value, row, d0, out);
    else if constexpr (KV_MODE == 3) {
        // hybrid K8V4: both sides are defined - K unrotated INT8, V rotated Q4_0 - so a value=true call
        // reads the Q4_0 pool instead of dereferencing the null v_q (no call site does today; PR review)
        if (value) load8_q4(p, value, row, d0, out);
        else load8_q8(p, value, row, d0, out);
    } else load8_q4(p, value, row, d0, out);
}

template <int KV_MODE>
/*
DPCT1110: The total declared local variable size in device function
attn_chunk_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
attn_chunk_kernel(const float *__restrict__ q, QsaAttnPools p,
                  const int32_t *__restrict__ ids,
                  const int32_t *__restrict__ step, int n_kv_heads,
                  int page_size, float scale, float *__restrict__ part_acc,
                  float *__restrict__ part_m, float *__restrict__ part_l,
                  int n_chunks, int cap = 0, long long scratch_stride = 0) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    q += (size_t)item_ct1.get_group(0) * (size_t)(n_kv_heads * G) * HD;
    ids += (size_t)item_ct1.get_group(0) * (size_t)cap;
    step += (size_t)item_ct1.get_group(0) * kStepCount;
    part_acc += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    part_m += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    part_l += (size_t)item_ct1.get_group(0) * (size_t)scratch_stride;
    auto &sq =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][HD]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // 12 KB: this KV head's query heads
    auto &sp =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][CHUNK]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // scores, then probabilities
    auto &srow =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<long long[CHUNK]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // pool row of each cell (page, kv head, slot)
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    const int n_ids = *(step + kStepWidth);
    const int chunk = item_ct1.get_group(2), kvh = item_ct1.get_group(1);
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = sycl::min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
#pragma unroll
    for (int i = t; i < G * HD; i += THREADS)
        sq[i / HD][i % HD] = q[(size_t)(kvh * G) * HD + i];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            // a block the KV streaming could not make resident keeps page -1 (ctl[3]); its cells are masked
            // (score -FLT_MAX, weight 0) instead of being read from before the pool.
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.
    for (int c = warp; c < CHUNK; c += WARPS) {
        if (c >= n_here || srow[c] < 0) {
            if (lane < G) sp[lane][c] = -FLT_MAX;
            continue;
        }
        float k8[8];
        load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const sycl::float4 qa =
                *reinterpret_cast<const sycl::float4 *>(&sq[h][lane * 8]);
            const sycl::float4 qb =
                *reinterpret_cast<const sycl::float4 *>(&sq[h][lane * 8 + 4]);
            float s = k8[0] * qa.x() + k8[1] * qa.y() + k8[2] * qa.z() +
                      k8[3] * qa.w() + k8[4] * qb.x() + k8[5] * qb.y() +
                      k8[6] * qb.z() + k8[7] * qb.w();
            s = warp_sum(s);
            if (lane == 0) sp[h][c] = s * scale;
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max(sycl::fmax(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0)
                             ? sycl::native::exp(a - m)
                             : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0)
                             ? sycl::native::exp(b - m)
                             : 0.0f;
        sp[h][lane] = ea;
        sp[h][lane + 32] = eb;
        const float l = warp_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // values: thread t owns dimension t for all 12 heads.
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    for (int c = 0; c < n_here; ++c) {
        if (srow[c] < 0) continue;   // masked above, weight 0
        float v;
        if constexpr (KV_MODE == 0) {
            v = sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_pool[srow[c] * HD + t]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
        } else if constexpr (KV_MODE == 1) {
            const float sc =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        p.v_scale[srow[c] * (HD / KV_Q8_GROUP) +
                                  t / KV_Q8_GROUP]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            v = (float) p.v_q[srow[c] * HD + t] * sc;
        } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
            constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
            const int b = t / QK4_0;
            const int rem = t % QK4_0;
            const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + srow[c] * bytes_per_head) + b;
            const float d =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(blk->d))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            const int j = rem < 16 ? rem : (rem - 16);
            const uint8_t byte = blk->qs[j];
            const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
            v = (float) nibble * d;
        }
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = sycl::fma(sp[h][c], v, acc[h]);
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

__dpct_inline__ void attn_merge_kernel(const float *__restrict__ part_acc,
                                       const float *__restrict__ part_m,
                                       const float *__restrict__ part_l,
                                       int n_chunks, float *__restrict__ attn,
                                       long long scratch_stride = 0) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    part_acc += (size_t)item_ct1.get_group(1) * (size_t)scratch_stride;
    part_m += (size_t)item_ct1.get_group(1) * (size_t)scratch_stride;
    part_l += (size_t)item_ct1.get_group(1) * (size_t)scratch_stride;
    attn += (size_t)item_ct1.get_group(1) *
            (size_t)item_ct1.get_group_range(2) * HD;
    const int h = item_ct1.get_group(2); // global query head
    const int kvh = h / G, hl = h % G;
    const int d = item_ct1.get_local_id(2);
    float M = -FLT_MAX;
#pragma unroll
    for (int c = 0; c < n_chunks; ++c)
        M = sycl::fmax(M, part_m[(kvh * n_chunks + c) * G + hl]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const int slot = kvh * n_chunks + c;
        const float m = part_m[slot * G + hl];
        if (m == -FLT_MAX) continue;
        const float w = sycl::native::exp(m - M);
        L = sycl::fma((float)(part_l[slot * G + hl]), (float)w, L);
        acc = sycl::fma((float)(part_acc[((size_t)slot * G + hl) * HD + d]),
                        (float)w, acc);
    }
    attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
}

}  // namespace

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || n_q > 65535) {
        std::fprintf(stderr, "qsa_decode_attn_batch: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    // per query: [acc: n_chunks*n_head*HD][m: n_chunks*n_head][l: n_chunks*n_head], all offsets from one stride
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dpct::dim3 grid((unsigned)n_chunks, (unsigned)s.n_head_kv,
                          (unsigned)n_q);
    dpct::queue_ptr st = strata::q_of(stream);
    if (kv_mode == 3)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->parallel_for<dpct_kernel_name<class attn_chunk_kernel_dfacfc,
                                          dpct_kernel_scalar<3>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<3>(q, pools, ids, steps, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, (int)cap, stride);
            });
    } else if (kv_mode == 2)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->parallel_for<dpct_kernel_name<class attn_chunk_kernel_b0d448,
                                          dpct_kernel_scalar<2>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<2>(q, pools, ids, steps, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, (int)cap, stride);
            });
    } else if (kv_mode == 1)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->parallel_for<dpct_kernel_name<class attn_chunk_kernel_2ad0a4,
                                          dpct_kernel_scalar<1>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<1>(q, pools, ids, steps, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, (int)cap, stride);
            });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->parallel_for<dpct_kernel_name<class attn_chunk_kernel_5c04f7,
                                          dpct_kernel_scalar<0>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<0>(q, pools, ids, steps, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, (int)cap, stride);
            });
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class attn_merge_kernel_2cdaf7>>(
            sycl::nd_range<3>(
                sycl::range(1, (unsigned)n_q, (unsigned)s.n_head) *
                    sycl::range(1, 1, HD),
                sycl::range(1, 1, HD)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                attn_merge_kernel(part_acc, part_m, part_l, n_chunks, attn,
                                  stride);
            });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
}

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) s.n_head * (HD + 2) + 64;
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !step ||
        !pools.page_table) {
        std::fprintf(stderr, "qsa_decode_attn: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    if (kv_mode == 3 ? (!pools.k_scale || !pools.v_q4)
                     : (kv_mode == 2 ? (!pools.v_q4) : (kv_mode == 1 ? (!pools.v_q || !pools.k_scale || !pools.v_scale)
                                                                     : (!pools.k_pool || !pools.v_pool)))) {
        std::fprintf(stderr, "qsa_decode_attn: incomplete KV pools\n");
        std::exit(1);
    }
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dpct::dim3 grid((unsigned)n_chunks, (unsigned)s.n_head_kv);
    dpct::queue_ptr st = strata::q_of(stream);
    if (kv_mode == 3)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->parallel_for<dpct_kernel_name<class attn_chunk_kernel_f262e6,
                                          dpct_kernel_scalar<3>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<3>(q, pools, ids, step, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, 0, 0);
            });
    } else if (kv_mode == 2)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->parallel_for<dpct_kernel_name<class attn_chunk_kernel_4fd923,
                                          dpct_kernel_scalar<2>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<2>(q, pools, ids, step, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, 0, 0);
            });
    } else if (kv_mode == 1)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->parallel_for<dpct_kernel_name<class attn_chunk_kernel_f6855e,
                                          dpct_kernel_scalar<1>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<1>(q, pools, ids, step, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, 0, 0);
            });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->parallel_for<dpct_kernel_name<class attn_chunk_kernel_7c8e09,
                                          dpct_kernel_scalar<0>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<0>(q, pools, ids, step, (int)s.n_head_kv,
                                     (int)s.page_size, scale, part_acc, part_m,
                                     part_l, n_chunks, 0, 0);
            });
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class attn_merge_kernel_4bb902>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)s.n_head) *
                                  sycl::range(1, 1, HD),
                              sycl::range(1, 1, HD)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                attn_merge_kernel(part_acc, part_m, part_l, n_chunks, attn, 0);
            });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
}

}  // namespace strata::kernels
