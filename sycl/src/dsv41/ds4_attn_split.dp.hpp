// Split-KV sparse attention: Strata (dsv41 CUDA lab), used with the DeepSeek attention of Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// (the surrounding types, AttnRows and the packed-row formats, are theirs; see third_party/project-maya/NOTICE.md).
// src/dsv41/ds4_attn_split.cuh - split-KV ("flash-decoding") version of DeepSeek's sparse attention for a few queries
// (decode T = 1, DSpark verify T = 2..5): 64 heads x d 512, MQA (K = V), window rows (BF16) + selected compressed rows (kKvFp4:
// E2M1 codes + E4M3 scales per 16) + a learned sink logit per head.  Same inputs and outputs as strata::ds4::sparse_attn
// (AttnRows gen 0, ckv kKvFp4); the shipped kernel runs one 256-thread block per (head, query) = 64 blocks on 48 SMs and takes
// ~125 us per query; this one splits the keys over R-row chunks and groups G heads per block, which share the decoded rows.
//   phase 1: block (head group, chunk, query): decode R rows to BF16 in shared memory (exact: window rows are BF16, packed
//            rows have <= 7 significant bits), scores for G heads (warp per row, lanes 16 values, shuffle reduction), local
//            softmax, partial output (thread = 2 output values) -> scratch {m, l, o[512]} per (query, head, chunk)
//   phase 2: block (head, query): merge the chunks and the sink.
// Plain C++ (no PTX), shuffles and shared memory only: the HIP port is a rename.
#pragma once
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "ds4_launch.h" // strata_ds4_launch: <<<>>> on CUDA and HIP, a dispatch on Metal
#include "ds4_kernels.dp.hpp"
#include "ds4_numerics.dp.hpp"

namespace strata::ds4 {
namespace attnsplit {

__dpct_inline__ float bf2f_(uint16_t b) {
    return sycl::bit_cast<float>((uint32_t)b << 16);
}
__dpct_inline__ uint16_t f2bf_exact(float f) {
    return (uint16_t)(sycl::bit_cast<unsigned int>(f) >> 16);
} // exact values only
__dpct_inline__ float e2m1_x2p14(uint32_t c) {
    return sycl::vec<sycl::half, 1>(
               sycl::bit_cast<sycl::half, unsigned short>(
                   (unsigned short)(((c & 8u) << 12) | ((c & 7u) << 9))))
        .convert<float, sycl::rounding_mode::automatic>()[0];
}
__dpct_inline__ float wsum(float v) {
#pragma unroll
    /*
    DPCT1108:122: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) v +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v,
            o);
    return v;
}
__dpct_inline__ float wmax(float v) {
#pragma unroll
    /*
    DPCT1108:123: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) v = sycl::fmax(
        v, dpct::experimental::permute_sub_group_by_xor(
               0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
               v, o));
    return v;
}

// d = 512 only (a thread owns 2 output values, a lane 16 values of a row)
template <int G, int R>
/*
DPCT1110:124: The total declared local variable size in device function
attn_part exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
attn_part(const float *__restrict__ q, const uint16_t *__restrict__ win,
          const uint8_t *__restrict__ ckv, size_t ckv_row_bytes,
          const int *__restrict__ sel, int sel_stride, AttnRows ar, int heads,
          float scale, float *__restrict__ part, float *__restrict__ ml,
          int nsplit) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int D = 512;
    auto &rows =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<uint16_t[R][D]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sc =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][R]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &pr = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        sycl::float4[R][(G + 3) / 4]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &mloc = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &lloc = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int hg = item_ct1.get_group(2), chunk = item_ct1.get_group(1),
              t = item_ct1.get_group(0);
    const int n_win = ar.nwin[t], n_sel = ar.nsel[t], total = n_win + n_sel;
    const int j0 = chunk * R;
    if (j0 >= total) return;
    const int nr = sycl::min(R, total - j0);
    const int lane = item_ct1.get_local_id(2) & 31,
              wp = item_ct1.get_local_id(2) >> 5;
    // q of this block's heads: lane's 16 values per head in registers (issued first: its latency overlaps the row loads)
    float qv[G][16];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const sycl::float4 *qp =
            (const sycl::float4 *)(q + ((size_t)t * heads + hg * G + g) * D +
                                   lane * 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const sycl::float4 v = qp[i]; qv[g][4 * i] = v.x();
            qv[g][4 * i + 1] = v.y();
            qv[g][4 * i + 2] = v.z();
            qv[g][4 * i + 3] = v.w();
        }
    }
    // ---- stage rows: warp per row, lane = 16 values; all of a warp's row loads are issued before any is converted
    constexpr int RPW = (R + 7) / 8;
    sycl::uint4 wa[RPW][2];
    sycl::uint2 cwv[RPW];
    uint32_t sbv[RPW];
    int rowoff[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        const int jj = wp + 8 * i, j = j0 + jj;
        rowoff[i] = -1;
        if (jj < nr) {
            if (j < n_win) {
                const sycl::uint4 *src =
                    (const sycl::uint4 *)(win + (size_t)(ar.win0[t] + j) * D +
                                          lane * 16);
                wa[i][0] = src[0];
                wa[i][1] = src[1];
            } else {
                rowoff[i] = sel[(size_t) t * sel_stride + (j - n_win)];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < RPW; ++i)
        if (rowoff[i] >= 0) {
            const uint8_t* r = ckv + (size_t) rowoff[i] * ckv_row_bytes;
            cwv[i] = *(const sycl::uint2 *)(r + lane * 8);
            sbv[i] = r[D / 2 + lane];
        }
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        const int jj = wp + 8 * i;
        if (jj >= nr) continue;
        sycl::uint4 *dst = (sycl::uint4 *)&rows[jj][lane * 16];
        if (rowoff[i] < 0) {
            dst[0] = wa[i][0];
            dst[1] = wa[i][1];
        } else {
            const float s = e4m3_value((uint8_t) sbv[i]) * 16384.0f;   // (the nibbles decode to value * 2^-14)
            uint32_t o[8];
#pragma unroll
            for (int w = 0; w < 2; ++w) {
                const uint32_t c = w ? cwv[i].y() : cwv[i].x();
#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    const uint32_t b = c >> (8 * x);
                    const uint16_t lo = f2bf_exact(e2m1_x2p14(b) * s), hi = f2bf_exact(e2m1_x2p14(b >> 4) * s);
                    o[4 * w + x] = (uint32_t) lo | ((uint32_t) hi << 16);
                }
            }
            dst[0] = sycl::uint4(o[0], o[1], o[2], o[3]);
            dst[1] = sycl::uint4(o[4], o[5], o[6], o[7]);
        }
    }
    /*
    DPCT1065:407: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // ---- scores
    for (int jj = wp; jj < nr; jj += 8) {
        const sycl::uint4 a = *(const sycl::uint4 *)&rows[jj][lane * 16],
                          b = *(const sycl::uint4 *)&rows[jj][lane * 16 + 8];
        float kv[16];
        const uint32_t w[8] = {a.x(), a.y(), a.z(), a.w(),
                               b.x(), b.y(), b.z(), b.w()};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            kv[2 * i] = sycl::bit_cast<float>(w[i] << 16);
            kv[2 * i + 1] = sycl::bit_cast<float>(w[i] & 0xffff0000u);
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float s = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) s += qv[g][i] * kv[i];
            s = wsum(s);
            if (lane == 0) sc[g][jj] = s * scale;
        }
    }
    /*
    DPCT1065:408: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // ---- local softmax: warp g handles head g
    if (wp < G) {
        float m = -INFINITY;
#pragma unroll
        for (int j = lane; j < nr; j += 32) m = sycl::fmax(m, sc[wp][j]);
        m = wmax(m);
        float l = 0.f;
        for (int j = lane; j < nr; j += 32) {
            const float p = sycl::native::exp(sc[wp][j] - m); sc[wp][j] = p;
            l += p;
        }
        l = wsum(l);
        if (lane == 0) { mloc[wp] = m; lloc[wp] = l; }
    }
    /*
    DPCT1065:409: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = item_ct1.get_local_id(2); i < nr * G; i += 256) {
        const int j = i / G, g = i % G; ((float *)&pr[j][0])[g] = sc[g][j];
    }
    /*
    DPCT1065:410: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // ---- partial output: thread = values 2 tid, 2 tid + 1 of every head
    float o0[G], o1[G];
#pragma unroll
    for (int g = 0; g < G; ++g) o0[g] = o1[g] = 0.f;
    const int c = item_ct1.get_local_id(2) * 2;
    for (int j = 0; j < nr; ++j) {
        const uint32_t kw = *(const uint32_t*) &rows[j][c];
        const float k0 = sycl::bit_cast<float>(kw << 16),
                    k1 = sycl::bit_cast<float>(kw & 0xffff0000u);
        const float* p = (const float*) &pr[j][0];
#pragma unroll
        for (int g = 0; g < G; ++g) { o0[g] += p[g] * k0; o1[g] += p[g] * k1; }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const size_t bi = ((size_t) t * heads + hg * G + g) * nsplit + chunk;
        *(sycl::float2 *)(part + bi * D + c) = sycl::float2(o0[g], o1[g]);
        if (item_ct1.get_local_id(2) == 0) {
            ml[bi * 2] = mloc[g]; ml[bi * 2 + 1] = lloc[g];
        }
    }
}

__dpct_inline__ void attn_merge(const float *__restrict__ part,
                                const float *__restrict__ ml,
                                const float *__restrict__ sink, int heads,
                                int nsplit, int R, AttnRows ar,
                                float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int D = 512;
    const int h = item_ct1.get_group(2), t = item_ct1.get_group(1);
    const int total = ar.nwin[t] + ar.nsel[t];
    const int ns = (total + R - 1) / R;
    const size_t base = ((size_t) t * heads + h) * nsplit;
    float M = sink[h];
#pragma unroll
    for (int s = 0; s < ns; ++s) M = sycl::fmax(M, ml[(base + s) * 2]);
    float den = sycl::native::exp(sink[h] - M);
#pragma unroll
    for (int s = 0; s < ns; ++s) den +=
        ml[(base + s) * 2 + 1] * sycl::native::exp(ml[(base + s) * 2] - M);
    const int c = item_ct1.get_local_id(2) * 2;
    float a0 = 0.f, a1 = 0.f;
    for (int s = 0; s < ns; ++s) {
        const float w = sycl::native::exp(ml[(base + s) * 2] - M);
        const sycl::float2 v =
            *(const sycl::float2 *)(part + (base + s) * D + c);
        a0 += w * v.x();
        a1 += w * v.y();
    }
    *(sycl::float2 *)(out + ((size_t)t * heads + h) * D + c) =
        sycl::float2(a0 / den, a1 / den);
}

struct AttnSplitScratch {
    float* part = nullptr;   // [T][heads][nsplit][512]
    float* ml = nullptr;     // [T][heads][nsplit][2]
    int nsplit = 0, max_t = 0;
};

// total_max = the largest nwin + nsel of the T queries
template <int G, int R>
void sparse_attn_split(const float *q, const uint16_t *win, const uint8_t *ckv,
                       size_t ckv_row_bytes, const int *sel, int sel_stride,
                       const AttnRows &ar, const float *sink, int heads, int T,
                       float scale, float *out, const AttnSplitScratch &sc,
                       int total_max, dpct::queue_ptr s) {
    const int nsplit = (total_max + R - 1) / R;
    {
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->parallel_for(
            sycl::nd_range<3>(sycl::range(T, nsplit, heads / G) *
                                  sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attn_part<G, R>(q, win, ckv, ckv_row_bytes, sel, sel_stride, ar,
                                heads, scale, sc.part, sc.ml, sc.nsplit);
            });
    }
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, T, heads) * sycl::range(1, 1, 256),
                          sycl::range(1, 1, 256)),
        [=](sycl::nd_item<3> item_ct1) {
            attn_merge(sc.part, sc.ml, sink, heads, sc.nsplit, R, ar, out);
        });
}

}  // namespace attnsplit
}  // namespace strata::ds4
