// xmx_int8_bench: an int8 XMX (DPAS) GEMM straight from IQ4_NL weights against the engine's dequant + oneMKL FP16 path,
// on one random Coder down-projection expert (K 640, n_out 2560). Standalone prototype - nothing in the engine calls it.
//   xmx_int8_bench [reps=50] [M...=16 32 64 80 128 256 512]
// IQ4_NL is 32-element blocks of (fp16 d, 16 bytes of nibble indices into an int8 codebook), so one K32 block is one
// int8 DPAS: activations are quantized per (row, 32-block) to int8 with a float scale, each block's int32 tile is scaled
// by d_x[row] * d_w[col] into float accumulators.
// Result (2026-10-03, B70): 0.98x at M 32, 0.55x at M 256, 0.38x at M 512 against dequant + oneMKL - not worth an engine
// port. Each K32 block's DPAS is followed by its rescale on the vector units, so XMX idles between blocks; oneMKL runs
// one uninterrupted K loop on the pre-dequantized FP16 matrix, and at 2560 x 640 the dequant is only ~20 us.
#include "strata/kernels/iq_kernels.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
namespace jm = sycl::ext::oneapi::experimental::matrix;
namespace jmi = sycl::ext::intel::experimental::matrix;
static float h2f(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }
static uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
static const int8_t kv_host[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
constexpr int SG = 16, NSG = 8, WG = SG * NSG, QB = 18;   // QB: bytes per IQ4_NL block

// X (fp16 [M][K]) -> Xq (int8 [Mp][K]) + sx ([Mp][K/32]); rows M..Mp-1 zero. One sub-group of 32 lanes per block.
static void quant_x(sycl::queue& q, const uint16_t* X, int M, int Mp, int K, int8_t* Xq, float* sx) {
    const int nkb = K / 32;
    q.parallel_for(sycl::nd_range<1>((size_t) Mp * nkb * 32, 32), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int blk = (int) it.get_group(0), l = (int) it.get_local_id(0);
        const int m = blk / nkb, b = blk % nkb;
        const float v = m < M ? (float) sycl::bit_cast<sycl::half>(X[(size_t) m * K + b * 32 + l]) : 0.0f;
        const float amax = sycl::reduce_over_group(it.get_sub_group(), sycl::fabs(v), sycl::maximum<float>());
        const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
        Xq[(size_t) m * K + b * 32 + l] = (int8_t) sycl::rint(v * id);
        if (l == 0) sx[blk] = d;
    });
}

// Y[Mp][n_out] = Xq . W^T; NB B tiles (NB * 16 features) per work-group, the whole K strip decoded once into local memory
template <int NB>
static void gemm_i8(sycl::queue& q, const uint8_t* W, int K, int n_out, const int8_t* Xq, const float* sx, int Mp, float* Y) {
    constexpr int NF = NB * 16;
    const int nkb = K / 32, groups = n_out / NF;
    const size_t bytes = (size_t) NF * K + (size_t) NF * nkb * 4;
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(bytes), cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) groups * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            constexpr int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
            auto sg = it.get_sub_group();
            const int t = (int) it.get_local_id(0), sgid = (int) sg.get_group_id()[0];
            const int f0 = (int) it.get_group(0) * NF;
            int8_t* Bp = (int8_t*) slm.get_multi_ptr<sycl::access::decorated::no>().get();
            float* sw = (float*) (Bp + (size_t) NF * K);
            auto gptr = [](auto* p) { return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p); };
            auto lptr = [](auto* p) { return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p); };
            // decode: B tile nb is K x 16 int8 in the packed (VNNI-4) layout, element (k, n) at (k / 4) * 64 + n * 4 + k % 4
            for (int i = t; i < NF * nkb * 16; i += WG) {
                const int j = i & 15, rb = i >> 4, r = rb / nkb, b = rb - r * nkb;
                const uint8_t* blk = W + ((size_t) (f0 + r) * nkb + b) * QB;
                const uint8_t qb = blk[2 + j];
                int8_t* tile = Bp + (size_t) (r >> 4) * 16 * K;
                const int n = r & 15, k0 = b * 32 + j, k1 = k0 + 16;
                tile[(k0 >> 2) * 64 + n * 4 + (k0 & 3)] = kv[qb & 15];
                tile[(k1 >> 2) * 64 + n * 4 + (k1 & 3)] = kv[qb >> 4];
                if (j == 0) sw[r * nkb + b] = h2f((uint16_t) (blk[0] | (blk[1] << 8)));
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (int mt = sgid; mt < Mp / 8; mt += NSG) {
                float acc[NB][8];
#pragma unroll
                for (int nb = 0; nb < NB; ++nb)
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[nb][i] = 0.0f;
                const int8_t* xa = Xq + (size_t) mt * 8 * K;
                const float* sxa = sx + (size_t) mt * 8 * nkb;
                for (int b = 0; b < nkb; ++b) {
                    jm::joint_matrix<sycl::sub_group, int8_t, jm::use::a, 8, 32, jm::layout::row_major> A;
                    jm::joint_matrix_load(sg, A, gptr(xa + b * 32), (size_t) K);
#pragma unroll
                    for (int nb = 0; nb < NB; ++nb) {
                        jm::joint_matrix<sycl::sub_group, int8_t, jm::use::b, 32, 16, jm::layout::ext_intel_packed> B;
                        jm::joint_matrix_load(sg, B, lptr(Bp + (size_t) nb * 16 * K + b * 8 * 64), (size_t) 64);
                        jm::joint_matrix<sycl::sub_group, int32_t, jm::use::accumulator, 8, 16> C;
                        jm::joint_matrix_fill(sg, C, 0);
                        jm::joint_matrix_mad(sg, C, A, B, C);
                        int e = 0;   // (sub-group shuffles of register-held scales here measured slower: 0.25x at M 512)
                        jmi::joint_matrix_apply(sg, C, [&](int32_t& x, size_t row, size_t col) {
                            acc[nb][e++] += (float) x * sxa[row * nkb + b] * sw[(nb * 16 + col) * nkb + b];
                        });
                    }
                }
#pragma unroll
                for (int nb = 0; nb < NB; ++nb) {
                    jm::joint_matrix<sycl::sub_group, int32_t, jm::use::accumulator, 8, 16> C;
                    jm::joint_matrix_fill(sg, C, 0);
                    int e = 0;
                    jmi::joint_matrix_apply(sg, C, [&](int32_t&, size_t row, size_t col) {
                        Y[(size_t) (mt * 8 + row) * n_out + f0 + nb * 16 + col] = acc[nb][e++];
                    });
                }
            }
        });
    });
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 50;
    std::vector<int> Ms;
    for (int i = 2; i < argc; ++i) Ms.push_back(std::atoi(argv[i]));
    if (Ms.empty()) Ms = {16, 32, 64, 80, 128, 256, 512};
    const int ty = 20, K = 640, n_out = 2560, nkb = K / 32;
    std::mt19937 rng(7);
    // random IQ4_NL weights with realistic block scales
    std::vector<uint8_t> hw((size_t) n_out * nkb * QB);
    std::uniform_real_distribution<float> ud(0.002f, 0.02f);
    for (size_t o = 0; o < hw.size(); o += QB) {
        const uint16_t d = f2h(ud(rng));
        hw[o] = d & 0xFF; hw[o + 1] = d >> 8;
        for (int j = 0; j < 16; ++j) hw[o + 2 + j] = (uint8_t) rng();
    }
    // fp32 host copy of the weights for the exact reference
    std::vector<float> wf((size_t) n_out * K);
    for (int n = 0; n < n_out; ++n)
        for (int b = 0; b < nkb; ++b) {
            const uint8_t* blk = &hw[((size_t) n * nkb + b) * QB];
            const float d = h2f((uint16_t) (blk[0] | (blk[1] << 8)));
            for (int j = 0; j < 16; ++j) {
                wf[(size_t) n * K + b * 32 + j] = d * kv_host[blk[2 + j] & 15];
                wf[(size_t) n * K + b * 32 + j + 16] = d * kv_host[blk[2 + j] >> 4];
            }
        }
    sycl::queue* q = &dpct::get_in_order_queue();
    std::printf("device: %s\n", q->get_device().get_info<sycl::info::device::name>().c_str());
    const int Mmax = *std::max_element(Ms.begin(), Ms.end()), Mpmax = (Mmax + 7) / 8 * 8;
    uint8_t* dW = sycl::malloc_device<uint8_t>(hw.size(), *q);
    uint16_t* dX = sycl::malloc_device<uint16_t>((size_t) Mpmax * K, *q);
    uint16_t* dWh = sycl::malloc_device<uint16_t>((size_t) n_out * K, *q);
    int8_t* dXq = sycl::malloc_device<int8_t>((size_t) Mpmax * K, *q);
    float* dsx = sycl::malloc_device<float>((size_t) Mpmax * nkb, *q);
    float* y_ref = sycl::malloc_device<float>((size_t) Mpmax * n_out, *q);
    float* y_i8 = sycl::malloc_device<float>((size_t) Mpmax * n_out, *q);
    q->memcpy(dW, hw.data(), hw.size()).wait();
    dpct::blas::descriptor_ptr h = new dpct::blas::descriptor();
    h->set_queue(q);
    auto time = [&](auto&& fn) {
        for (int i = 0; i < 3; ++i) fn();
        q->wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) fn();
        q->wait();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    };
    const double t_empty = time([&] { q->parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {}); });
    std::printf("empty kernel launch: %.3f ms\n", t_empty);
    int fails = 0;
    for (const int M : Ms) {
        const int Mp = (M + 7) / 8 * 8;
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<uint16_t> hx((size_t) M * K);
        for (auto& x : hx) x = f2h(nd(rng));
        q->memcpy(dX, hx.data(), hx.size() * 2).wait();
        auto dequant = [&] { strata::kernels::iq_dequant_f16(ty, dW, (int64_t) n_out * K, dWh, q); };
        auto mkl = [&] {
            const float alpha = 1.0f, beta = 0.0f;
            dpct::blas::gemm(h, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, n_out, M, K, &alpha, dWh,
                             dpct::library_data_t::real_half, K, dX, dpct::library_data_t::real_half, K, &beta, y_ref,
                             dpct::library_data_t::real_float, n_out, dpct::compute_type::f32);
        };
        auto quant = [&] { quant_x(*q, dX, M, Mp, K, dXq, dsx); };
        auto i8 = [&](int nb) {
            if (nb == 1) gemm_i8<1>(*q, dW, K, n_out, dXq, dsx, Mp, y_i8);
            else if (nb == 2) gemm_i8<2>(*q, dW, K, n_out, dXq, dsx, Mp, y_i8);
            else gemm_i8<4>(*q, dW, K, n_out, dXq, dsx, Mp, y_i8);
        };
        dequant(); mkl(); quant(); i8(2); q->wait();
        std::vector<float> r((size_t) M * n_out), x((size_t) M * n_out);
        q->memcpy(r.data(), y_ref, r.size() * 4).wait();
        q->memcpy(x.data(), y_i8, x.size() * 4).wait();
        // exact fp32 reference on the fp16 inputs; relative RMS error of both paths
        double e_ref = 0, e_i8 = 0, s = 0;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < n_out; ++n) {
                double acc = 0;
                const float* w = &wf[(size_t) n * K];
                for (int k = 0; k < K; ++k) acc += (double) h2f(hx[(size_t) m * K + k]) * w[k];
                const size_t i = (size_t) m * n_out + n;
                s += acc * acc; e_ref += (r[i] - acc) * (r[i] - acc); e_i8 += (x[i] - acc) * (x[i] - acc);
            }
        const double rel_ref = std::sqrt(e_ref / s), rel_i8 = std::sqrt(e_i8 / s);
        if (!(rel_i8 < 0.02)) ++fails;
        const double t_dq = time(dequant), t_mkl = time(mkl), t_q = time(quant);
        double best = 1e9; int best_nb = 0;
        double t_nb[3];
        for (int k = 0; k < 3; ++k) {
            const int nb = 1 << k;
            t_nb[k] = time([&] { i8(nb); });
            if (t_nb[k] < best) { best = t_nb[k]; best_nb = nb; }
        }
        const double base = t_dq + t_mkl, ours = t_q + best;
        std::printf("M %4d | dequant %.3f + mkl %.3f = %.3f ms | quant %.3f + i8 nb1 %.3f nb2 %.3f nb4 %.3f -> %.3f ms (nb%d) | "
                    "%.2fx | rel err mkl %.2e i8 %.2e\n",
                    M, t_dq, t_mkl, base, t_q, t_nb[0], t_nb[1], t_nb[2], ours, best_nb, base / ours, rel_ref, rel_i8);
    }
    return fails ? 1 : 0;
}
