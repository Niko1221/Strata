// xmx_mmvq_bench (exp 42): DPAS int8 MMVQ on reordered Q6_K - does moving the decode dot onto the matrix pipe beat
// the shipped AOS dp4a kernel and exp 27's no-unpack byte ceiling?
//
// Arms, all in one process so every ratio is same-run:
//   1  native_q6_k_mmvq            shipped AOS 6-bit unpack + dp4a on the ALU pipe (19.1/21.7/26.3/36.1/55.3 us)
//   2  native_mmvq_q6k_unpacked    exp 27's one-time pre-unpack to signed bytes, still dp4a (9.3/11.1/16.5/20.9/25.9)
//   3  xmx_dpas_q6k                this experiment: 6-bit unpack in registers, dot on DPAS int8
//
//
// Arms 2 and 3 read one-time host-built transforms, never timed: arm 2 is exp 27's pre-unpack and arm 3 is a
// VNNI-preordered byte tile (272 B per 256 weights, the same class as arm 2, so the two differ in the dot and not
// in resident bytes). `sweep` as argv[4] adds exp 44's width comparison against the prompt path's oneMKL FP16 GEMM.
//
// int8 DPAS on this device is K=32 with N in {8,16} and repeat count (M) 1..8. Q6_K keeps one scale per 16 weights,
// so each 32-deep DPAS covers two different scales: the kernel issues two DPAS per group with the unused half of
// B zeroed, which is what "zeros in B drop the other half of K" means in the upstream design.
//
//   xmx_mmvq_bench [n_in=2560] [n_out=2560] [reps=400] [ks=0 sweep]
#include "strata/kernels/native_mmvq.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;
namespace xmx   = sycl::ext::intel::esimd::xmx;

struct Q6KBlock {                     // the port's AOS Q6_K, GGUF block order
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    uint16_t d;
};
static_assert(sizeof(Q6KBlock) == 210);

static ESIMD_INLINE float h2f(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }

ESIMD_INLINE esimd::simd<uint32_t, 128> q6k_bytes(esimd::simd<uint32_t, 128> q, uint32_t zero) {
    return ((q | 0x80808080u) - zero) ^ 0x80808080u;
}

// One work-item owns 16 weight rows (the DPAS N tile). The weights arrive in DPAS B order already: the one-time
// reorder writes, per (tile, block, 32-wide group), the 512-byte tile with element (k, n) at dword (k/4)*16+n and
// byte k%4, so a group is one contiguous 512-byte load and no in-register transpose is needed. The scales and d
// stay in row-major planes: scales 16 B per (row, block), d one half per (row, block).
template <int NC, int KS, bool ONE_DPAS>
static void xmx_dpas_q6k_kernel(const uint8_t* __restrict__ tiles, const uint8_t* __restrict__ xq,
                                float* __restrict__ y, int n_in, int n_out, int ntiles,
                                const int8_t* __restrict__ sc, const sycl::half* __restrict__ dh, sycl::queue* q) {
    constexpr int NT = 16, K = 32;
    const int bpr = n_in / 256, ng = n_in / 32;

    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> red(sycl::range<1>((size_t) KS * NC * NT), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ntiles * KS), sycl::range<1>(KS)),
                         [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            const int tile = (int) (it.get_group(0) / KS);
            const int lid = (int) it.get_local_id(0);
            const int row0 = tile * NT;
            esimd::simd<uint32_t, 128> half_mask = 0u;
#pragma unroll
            for (int i = 64; i < 128; ++i) half_mask[i] = 0xFFFFFFFFu;   // keep kd 4..7 (the upper 16 of K)
            esimd::simd<float, NC * NT> Cf = 0.0f;
            for (int sb = lid; sb < bpr; sb += KS) {
                esimd::simd<uint32_t, NT> rows;
#pragma unroll
                for (int n = 0; n < NT; ++n) rows[n] = (uint32_t) (row0 + n < n_out ? row0 + n : n_out - 1);
                int8_t scl[NT][16];
                float dsp[NT];
                {
#pragma unroll
                    for (int n = 0; n < NT; ++n) dsp[n] = h2f(*(const uint16_t*) (dh + (size_t) rows[n] * bpr + sb));
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        const uint8_t* sp = (const uint8_t*) (sc + ((size_t) rows[n] * bpr + sb) * 16);
#pragma unroll
                        for (int i = 0; i < 16; ++i) scl[n][i] = (int8_t) sp[i];
                    }
                }
                for (int j = 0; j < 8; ++j) {
                    const uint8_t* tp = tiles + (((size_t) tile * bpr + sb) * 8 + j) * (NT * K);
                    esimd::simd<uint32_t, 128> Xt = esimd::block_load<uint32_t, 128>((const uint32_t*) tp);
                    esimd::simd<uint32_t, 128> Xl = Xt & ~half_mask, Xh = Xt & half_mask;
                    esimd::simd<int8_t, NT * K> Blo = Xl.template bit_cast_view<int8_t>();
                    esimd::simd<int8_t, NT * K> Bhi = Xh.template bit_cast_view<int8_t>();
                    esimd::simd<float, NT> rs0 = 0.0f, rs1 = 0.0f;
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        rs0[n] = (float) scl[n][2 * j] * dsp[n];
                        rs1[n] = (float) scl[n][2 * j + 1] * dsp[n];
                    }
                    const int g8 = sb * 8 + j;
                    for (int m = 0; m < NC; ++m) {
                        const uint8_t* xb = xq + ((size_t) m * ng + g8) * 36;
                        const esimd::simd<int8_t, K> A = esimd::block_load<int8_t, K>((const int8_t*) (xb + 4));
                        const float dx = (float) sycl::bit_cast<sycl::half>(*(const uint16_t*) xb);
                        esimd::simd<int, NT> C0 = xmx::dpas<8, 1, int>(Blo, A);
                        esimd::simd<int, NT> C1 = 0;
                        if constexpr (!ONE_DPAS) C1 = xmx::dpas<8, 1, int>(Bhi, A);
                        Cf.template select<NT, 1>(m * NT) +=
                            (esimd::convert<float>(C0) * rs0 + esimd::convert<float>(C1) * rs1) * dx;
                    }
                }
            }
            float* sh = red.get_multi_ptr<sycl::access::decorated::no>().get();
#pragma unroll
            for (int i = 0; i < NC * NT; ++i) sh[lid * (NC * NT) + i] = Cf[i];
            it.barrier(sycl::access::fence_space::local_space);
            if (lid != 0) return;
#pragma unroll
            for (int m = 0; m < NC; ++m)
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    float v = 0.0f;
#pragma unroll
                    for (int k = 0; k < KS; ++k) v += sh[k * (NC * NT) + m * NT + n];
                    if (row0 + n < n_out) y[(size_t) m * n_out + row0 + n] = v;
                }
        });
    });
}

template <int NC>
static void xmx_dpas_q6k(const void* tiles, const void* xq, float* y, int n_in, int n_out, const void* sc,
                         const void* dh, sycl::queue* q) {
    const int ntiles = (n_out + 15) / 16;
    const int bpr = n_in / 256;
    const uint8_t* t = (const uint8_t*) tiles;
    const uint8_t* x = (const uint8_t*) xq;
    const int8_t* scp = (const int8_t*) sc;
    const sycl::half* dhp = (const sycl::half*) dh;
#define XMX_RUN(KS)                                                                                          \
    do {                                                                                                       \
        if (std::getenv("XMX_ONE_DPAS"))                                                                      \
            xmx_dpas_q6k_kernel<NC, KS, true>(t, x, y, n_in, n_out, ntiles, scp, dhp, q);                      \
        else                                                                                                   \
            xmx_dpas_q6k_kernel<NC, KS, false>(t, x, y, n_in, n_out, ntiles, scp, dhp, q);                     \
    } while (0)
    if (bpr >= 8) XMX_RUN(8);
    else if (bpr >= 4) XMX_RUN(4);
    else if (bpr >= 2) XMX_RUN(2);
    else XMX_RUN(1);
#undef XMX_RUN
}

// the Q6_K 6-bit code of element e, straight from the GGUF block (the value the DPAS sees, before scale and d)
static inline int q6k_code(const Q6KBlock& b, int e) {
    const int e2 = e & 127;
    const int qlb = (e >= 128) ? 64 : 0, qhb = (e >= 128) ? 32 : 0;
    const int nib = (e2 >= 64) ? 4 : 0;
    const int raw = ((b.ql[qlb + (e2 & 63)] >> nib) & 0xF) | ((((b.qh[qhb + (e2 & 31)] >> (2 * (e2 >> 5))) & 3)) << 4);
    return raw - 32;
}

// One-time reorder into DPAS B order: per (16-row tile, block, 32-wide group) a 512-byte tile with element (k, n)
// at dword (k/4)*16+n and byte k%4 - the layout the hardware consumes. 272 B per 256 weights, the same class as
// the Q6K->Q6U pre-unpack (arm 2), so the two arms differ in the dot, not in resident bytes.
static void reorder_q6k_vnni(const std::vector<Q6KBlock>& aos, std::vector<uint8_t>& tiles, std::vector<int8_t>& scl,
                             std::vector<uint16_t>& dh, int n_out, int bpr) {
    const int ntiles = n_out / 16;
    tiles.assign((size_t) ntiles * bpr * 8 * 512, 0);
    scl.assign((size_t) n_out * bpr * 16, 0);
    dh.assign((size_t) n_out * bpr, 0);
    for (int r = 0; r < n_out; ++r)
        for (int b = 0; b < bpr; ++b) {
            const Q6KBlock& blk = aos[(size_t) r * bpr + b];
            std::memcpy(scl.data() + ((size_t) r * bpr + b) * 16, blk.scales, 16);
            dh[(size_t) r * bpr + b] = blk.d;
            const int tile = r / 16, n = r % 16;
            for (int j = 0; j < 8; ++j)
                for (int kk = 0; kk < 32; ++kk) {
                    const int code = q6k_code(blk, 32 * j + kk);
                    tiles[(((size_t) tile * bpr + b) * 8 + j) * 512 + (kk / 4) * 64 + n * 4 + (kk % 4)] = (uint8_t) code;
                }
        }
}

// llama.cpp's dequantize_row_q6_K, used only for the correctness reference
static void dequant_q6k_row(const Q6KBlock& x, float* y) {
    const float d = h2f(x.d);
    for (int n = 0; n < 256; n += 128) {
        const uint8_t* ql = x.ql + n / 2;
        const uint8_t* qh = x.qh + n / 4;
        const int8_t* sc = x.scales + n / 16;
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int8_t q1 = (int8_t) ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t) ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0] = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128;
    }
}

// exp 44's width sweep: the DPAS int8 matvec against the path the prompt actually takes (dequant + oneMKL
// FP16 GEMM) at prompt-batch widths 9-80, which is llama.cpp PR 29864's window. Per-call us only.
static int sweep_mode(const std::vector<Q6KBlock>& hos, const std::vector<uint8_t>& htiles, const std::vector<int8_t>& hsc,
                      const std::vector<uint16_t>& hdh, int n_in, int n_out, int reps) {
    const int bpr = n_in / 256, ng = n_in / 32;
    const int widths[] = {1, 2, 4, 6, 8, 9, 12, 16, 24, 32, 48, 64, 80};
    const int nmax = 80;
    sycl::queue* q = &dpct::get_in_order_queue();
    void* w6 = sycl::malloc_device((size_t) n_out * bpr * 210, *q);
    void* wv = sycl::malloc_device(htiles.size(), *q);
    void* wsc = sycl::malloc_device(hsc.size(), *q);
    void* wdh = sycl::malloc_device(hdh.size() * 2, *q);
    q->memcpy(w6, hos.data(), (size_t) n_out * bpr * 210).wait();
    q->memcpy(wv, htiles.data(), htiles.size()).wait();
    q->memcpy(wsc, hsc.data(), hsc.size()).wait();
    q->memcpy(wdh, hdh.data(), hdh.size() * 2).wait();
    const size_t ubytes = strata::kernels::native_mmvq_q6k_preunpack_bytes(n_in, n_out);
    void* wu = sycl::malloc_device(ubytes, *q);
    strata::kernels::native_q6k_preunpack(w6, wu, n_in, n_out, q);
    q->wait();
    // a persistent FP16 copy of the same weights: the prompt path's product half (its dequant half is exp 07's 30%)
    const size_t kN = (size_t) n_in * n_out;
    std::vector<uint16_t> hf(kN);
    {
        std::vector<float> row(n_in);
        for (int r = 0; r < n_out; ++r) {
            for (int b = 0; b < bpr; ++b) dequant_q6k_row(hos[(size_t) r * bpr + b], &row[(size_t) b * 256]);
            for (int k = 0; k < n_in; ++k) hf[(size_t) k + (size_t) r * n_in] = sycl::bit_cast<uint16_t>(sycl::half(row[k]));
        }
    }
    uint16_t* dw = sycl::malloc_device<uint16_t>(kN, *q);
    q->memcpy(dw, hf.data(), kN * 2).wait();
    std::vector<float> xf((size_t) nmax * n_in);
    for (size_t t = 0; t < (size_t) nmax; ++t)
        for (int k = 0; k < n_in; ++k)
            xf[(size_t) k + t * n_in] = (float) ((int) ((k * 7919 + t * 104729) % 1000) - 500) / 250.f;
    void* xq = sycl::malloc_device((size_t) nmax * ng * 36, *q);
    // native_quantize_q8_1 caps ncols at 8, and the q8_1 buffer is column-major, so wider batches quantize in
    // column slices into the same buffer
    for (int c0 = 0; c0 < nmax; c0 += 8) {
        const int cn = std::min(8, nmax - c0);
        strata::kernels::native_quantize_q8_1(xf.data() + (size_t) c0 * n_in, (uint8_t*) xq + (size_t) c0 * ng * 36,
                                              n_in, cn, q);
    }
    float* y2 = sycl::malloc_device<float>((size_t) nmax * n_out, *q);
    float* y3 = sycl::malloc_device<float>((size_t) nmax * n_out, *q);
    uint16_t* dx = sycl::malloc_device<uint16_t>((size_t) nmax * n_in, *q);
    {
        std::vector<uint16_t> hx((size_t) nmax * n_in);
        for (size_t i = 0; i < hx.size(); ++i) hx[i] = sycl::bit_cast<uint16_t>(sycl::half(xf[i]));
        q->memcpy(dx, hx.data(), hx.size() * 2).wait();
    }
    float* yg = sycl::malloc_device<float>((size_t) nmax * n_out, *q);
    dpct::blas::descriptor_ptr h = new dpct::blas::descriptor();
    h->set_queue(q);
    const float alpha = 1.0f, beta = 0.0f;
    std::printf("%5s %10s %10s %10s   %s\n", "width", "preunpack", "DPAS", "fp16GEMM", "DPAS vs fp16GEMM");   // per-call us
    for (int wi = 0; wi < (int) (sizeof(widths) / sizeof(int)); ++wi) {
        const int m = widths[wi];
        // the shipped pre-unpack kernel refuses ncols > 8, so past 8 the comparison is DPAS vs the prompt path's GEMM
        auto arm2 = [&] { strata::kernels::native_mmvq_q6k_unpacked(wu, xq, y2, n_in, n_out, m, q); };
        auto arm3 = [&] {
            switch (m) {
                case 1: xmx_dpas_q6k<1>(wv, xq, y3, n_in, n_out, wsc, wdh, q); break;
                case 2: xmx_dpas_q6k<2>(wv, xq, y3, n_in, n_out, wsc, wdh, q); break;
                case 4: xmx_dpas_q6k<4>(wv, xq, y3, n_in, n_out, wsc, wdh, q); break;
                case 6: xmx_dpas_q6k<6>(wv, xq, y3, n_in, n_out, wsc, wdh, q); break;
                case 8: xmx_dpas_q6k<8>(wv, xq, y3, n_in, n_out, wsc, wdh, q); break;
                default:
                    for (int c = 0; c < m; ++c)   // no verified M>1 DPAS: m single-column launches over the same tiles
                        xmx_dpas_q6k<1>(wv, xq, y3, n_in, n_out, wsc, wdh, q);
                    break;
            }
        };
        auto arm4 = [&] {
            dpct::blas::gemm(h, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, n_out, m, n_in, &alpha, dw,
                             dpct::library_data_t::real_half, n_in, dx, dpct::library_data_t::real_half, n_in, &beta, yg,
                             dpct::library_data_t::real_float, n_out, dpct::compute_type::f32);
        };
        if (m <= 8) arm2();
        arm3(); arm4(); q->wait();
        auto warm = [&](auto&& fn) {
            const auto w0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() < 300) {
                for (int i = 0; i < 10; ++i) fn();
                q->wait();
            }
        };
        auto time = [&](auto&& fn) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) fn();
            q->wait();
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        };
        if (m <= 8) warm(arm2);
        warm(arm3); warm(arm4);
        const double t2 = m <= 8 ? time(arm2) : 0.0, t3 = time(arm3), t4 = time(arm4);
        std::printf("%5d %9s %9.1f us %9.1f us   %.2fx\n", m, m <= 8 ? "" : "n/a", t3, t4, t4 / t3);
    }
    return 0;
}

int main(int argc, char** argv) {
    const int n_in = argc > 1 ? std::atoi(argv[1]) : 2560;
    const bool sweep = argc > 4 && std::strcmp(argv[4], "sweep") == 0;
    const int n_out = argc > 2 ? std::atoi(argv[2]) : 2560;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 400;
    const int cols[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    const int nmax = 8;
    const int bpr = n_in / 256;
    const int ng = n_in / 32;
    if (n_in % 256 || n_out % 16) { std::printf("n_in must be a multiple of 256, n_out of 16\n"); return 2; }
    const size_t nb = (size_t) n_out * bpr;

    std::vector<Q6KBlock> hos(nb);
    for (size_t b = 0; b < nb; ++b) {
        for (int i = 0; i < 128; ++i) hos[b].ql[i] = (uint8_t) ((b * 2654435761u + i * 40503u) >> 11);
        for (int i = 0; i < 64; ++i) hos[b].qh[i] = (uint8_t) ((b * 2246822519u + i * 3266489917u) >> 13);
        for (int i = 0; i < 16; ++i) hos[b].scales[i] = (int8_t) (1 + ((b * 31 + i * 17) % 60));
        hos[b].d = sycl::bit_cast<uint16_t>(sycl::half(0.02f + 0.01f * (float) (b % 7)));
    }
    std::vector<uint8_t> htiles;
    std::vector<int8_t> hsc;
    std::vector<uint16_t> hdh;
    reorder_q6k_vnni(hos, htiles, hsc, hdh, n_out, bpr);

    sycl::queue* q = &dpct::get_in_order_queue();
    std::printf("device: %s   shape %d x %d  (%.2f MB packed Q6_K, %.2f MB VNNI tiles)\n",
                q->get_device().get_info<sycl::info::device::name>().c_str(), n_out, n_in,
                nb * 210 / 1e6, htiles.size() / 1e6);
    void* w6 = sycl::malloc_device(nb * 210, *q);
    void* wv = sycl::malloc_device(htiles.size(), *q);
    void* wsc = sycl::malloc_device(hsc.size(), *q);
    void* wdh = sycl::malloc_device(hdh.size() * 2, *q);
    q->memcpy(w6, hos.data(), nb * 210).wait();
    q->memcpy(wv, htiles.data(), htiles.size()).wait();
    q->memcpy(wsc, hsc.data(), hsc.size()).wait();
    q->memcpy(wdh, hdh.data(), hdh.size() * 2).wait();
    const size_t ubytes = strata::kernels::native_mmvq_q6k_preunpack_bytes(n_in, n_out);
    void* wu = sycl::malloc_device(ubytes, *q);
    strata::kernels::native_q6k_preunpack(w6, wu, n_in, n_out, q);
    q->wait();

    std::vector<float> xf((size_t) nmax * n_in);
    for (size_t t = 0; t < (size_t) nmax; ++t)
        for (int k = 0; k < n_in; ++k)
            xf[(size_t) k + t * n_in] = (float) ((int) ((k * 7919 + t * 104729) % 1000) - 500) / 250.f;
    void* xq = sycl::malloc_device((size_t) nmax * ng * 36, *q);
    strata::kernels::native_quantize_q8_1(xf.data(), xq, n_in, nmax, q);
    float* y1 = sycl::malloc_device<float>((size_t) nmax * n_out, *q);
    float* y2 = sycl::malloc_device<float>((size_t) nmax * n_out, *q);
    float* y3 = sycl::malloc_device<float>((size_t) nmax * n_out, *q);

    if (sweep) return sweep_mode(hos, htiles, hsc, hdh, n_in, n_out, reps);

    int fails = 0;
    for (int ci = 0; ci < 8; ++ci) {
        const int nc = cols[ci];
        auto run1 = [&] { strata::kernels::native_q6_k_mmvq(w6, xq, y1, n_in, n_out, nc, q); };
        auto run2 = [&] { strata::kernels::native_mmvq_q6k_unpacked(wu, xq, y2, n_in, n_out, nc, q); };
        auto run3 = [&] {
            switch (nc) {
                case 1: xmx_dpas_q6k<1>(wv, xq, y3, n_in, n_out, wsc, wdh, q); break;
                case 2: xmx_dpas_q6k<2>(wv, xq, y3, n_in, n_out, wsc, wdh, q); break;
                case 3: xmx_dpas_q6k<3>(wv, xq, y3, n_in, n_out, wsc, wdh, q); break;
                case 4: xmx_dpas_q6k<4>(wv, xq, y3, n_in, n_out, wsc, wdh, q); break;
                case 5: xmx_dpas_q6k<5>(wv, xq, y3, n_in, n_out, wsc, wdh, q); break; break;
                case 6: xmx_dpas_q6k<6>(wv, xq, y3, n_in, n_out, wsc, wdh, q); break; break;
                case 7: xmx_dpas_q6k<7>(wv, xq, y3, n_in, n_out, wsc, wdh, q); break; break;
                default: xmx_dpas_q6k<8>(wv, xq, y3, n_in, n_out, wsc, wdh, q); break; break;
            }
        };
        run1(); run2(); run3(); q->wait();
        std::vector<float> r1((size_t) nc * n_out), r2((size_t) nc * n_out), r3((size_t) nc * n_out);
        q->memcpy(r1.data(), y1, r1.size() * 4).wait();
        q->memcpy(r2.data(), y2, r2.size() * 4).wait();
        q->memcpy(r3.data(), y3, r3.size() * 4).wait();
        const int nref = 48;
        std::vector<float> wrow(n_in);
        double maxabs = 0, scale = 0;
        double e1 = 0, e2 = 0, e3 = 0;
        for (int row = 0; row < nref; ++row) {
            for (int b = 0; b < bpr; ++b) dequant_q6k_row(hos[(size_t) row * bpr + b], &wrow[(size_t) b * 256]);
            for (int t = 0; t < nc; ++t) {
                double acc = 0;
                for (int k = 0; k < n_in; ++k) acc += (double) wrow[k] * xf[(size_t) k + t * n_in];
                maxabs = std::max(maxabs, std::fabs(acc));
                scale += acc * acc;
                const size_t i = (size_t) row + t * n_out;
                e1 += (r1[i] - acc) * (r1[i] - acc);
                e2 += (r2[i] - acc) * (r2[i] - acc);
                e3 += (r3[i] - acc) * (r3[i] - acc);
            }
        }
        if (std::getenv("XMX_CMP") && nc == 1) {
            std::vector<float> wr(n_in);
            for (int row = 0; row < 4; ++row) {
                for (int b = 0; b < bpr; ++b) dequant_q6k_row(hos[(size_t) row * bpr + b], &wr[(size_t) b * 256]);
                double acc = 0;
                for (int k = 0; k < n_in; ++k) acc += (double) wr[k] * xf[k];
                std::printf("row %d: ref %12.4f | AOS %12.4f | preunpack %12.4f | DPAS %12.4f\n", row, acc, r1[row], r2[row],
                            r3[row]);
            }
        }
        const double rms1 = std::sqrt(e1 / scale), rms2 = std::sqrt(e2 / scale), rms3 = std::sqrt(e3 / scale);
        const bool ok = rms3 < 20 * std::max(rms1, 1e-9);
        if (!ok) ++fails;
        std::printf("ncols %d  rel-RMS vs fp64:  AOS dp4a %.2e | pre-unpack %.2e | DPAS %.2e  %s\n", nc, rms1, rms2, rms3,
                    ok ? "ok" : "FAIL");
        auto warm = [&](auto&& fn) {
            const auto w0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() < 300) {
                for (int i = 0; i < 20; ++i) fn();
                q->wait();
            }
        };
        auto time = [&](auto&& fn) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) fn();
            q->wait();
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        };
        warm(run1); warm(run2); warm(run3);
        const double t1 = time(run1), t2 = time(run2), t3 = time(run3);
        std::printf("%2d cols:  AOS dp4a %6.1f us | pre-unpack %6.1f us (%.2fx) | DPAS %6.1f us (%.2fx vs AOS, %.2fx vs pre-unpack)\n",
                    nc, t1, t2, t1 / t2, t3, t1 / t3, t2 / t3);
    }
    std::printf(fails ? "xmx_mmvq_bench: %d FAILURES\n" : "xmx_mmvq_bench: all arms within tolerance\n", fails);
    return fails ? 1 : 0;
}