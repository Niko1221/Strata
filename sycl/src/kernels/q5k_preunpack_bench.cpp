// q5k_preunpack_bench: P0 #5 (exp 30) - time the pre-unpacked Q5_K decode (Q5U, no-bit-unpack) against the packed
// native_q5_k_mmvq at the engine's ncols.  Mirrors q6k_preunpack_bench: Q5_K is the min-offset analog of Q6_K.
//   q5k_preunpack_bench [n_in=2560] [n_out=2560] [reps=400]
#include "strata/kernels/native_mmvq.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

int main(int argc, char** argv) {
    const int n_in = argc > 1 ? std::atoi(argv[1]) : 2560;
    const int n_out = argc > 2 ? std::atoi(argv[2]) : 2560;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 400;
    const int cols[] = { 1, 2, 4, 6, 8 };
    const size_t w5bytes = (size_t) n_out * (n_in / 256) * 176;
    sycl::queue* q = &dpct::get_in_order_queue();

    std::vector<uint8_t> hw5(w5bytes);
    for (size_t i = 0; i < w5bytes; ++i) hw5[i] = (uint8_t) (i * 2654435761u >> 13);
    for (size_t b = 0; b + 175 < w5bytes; b += 176) { hw5[b] = 0x00; hw5[b + 1] = 0x3c; hw5[b + 2] = 0x00; hw5[b + 3] = 0x3c; }
    void* dw = sycl::malloc_device(w5bytes, *q);
    q->memcpy(dw, hw5.data(), w5bytes).wait();
    const size_t u5bytes = strata::kernels::native_mmvq_q5k_preunpack_bytes(n_in, n_out);
    void* du = sycl::malloc_device(u5bytes, *q);
    strata::kernels::native_q5k_preunpack(dw, du, n_in, n_out, q);
    q->wait();

    const int nmax = cols[4];
    std::vector<float> xf((size_t) nmax * n_in);
    for (size_t t = 0; t < (size_t) nmax; ++t)
        for (size_t k = 0; k < (size_t) n_in; ++k) xf[k + t * n_in] = (float) ((int) ((k * 7919 + t * 104729) % 1000) - 500) / 250.f;
    void* xq = sycl::malloc_device((size_t) nmax * (n_in / 32) * 36, *q);
    float* y5 = sycl::malloc_device<float>((size_t) nmax * n_out, *q);
    float* yu = sycl::malloc_device<float>((size_t) nmax * n_out, *q);

    for (int ci = 0; ci < 5; ++ci) {
        const int ncols = cols[ci];
        strata::kernels::native_quantize_q8_1(xf.data(), xq, n_in, ncols, q);
        auto q5 = [&]() { strata::kernels::native_q5_k_mmvq(dw, xq, y5, n_in, n_out, ncols, q); };
        auto qu = [&]() { strata::kernels::native_mmvq_q5k_unpacked(du, xq, yu, n_in, n_out, ncols, q); };
        q5(); qu(); q->wait();
        std::vector<float> u((size_t) ncols * n_out);
        q->memcpy(u.data(), yu, u.size() * 4).wait();
        int bad = 0; for (float v : u) if (!std::isfinite(v)) ++bad;
        auto warm = [&](auto&& fn) { const auto s = std::chrono::steady_clock::now();
            while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count() < 300) { for (int i = 0; i < 20; ++i) fn(); q->wait(); } };
        warm(q5); warm(qu);
        auto time = [&](auto&& fn) { const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) fn(); q->wait();
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps; };
        const double t5 = time(q5), tu = time(qu);
        std::printf("%2d cols:  Q5_K packed %6.1f us (%5.0f GB/s) | Q5U pre-unpacked %6.1f us (%5.0f GB/s) %.2fx  [%s]\n",
                    ncols, t5, w5bytes / t5 / 1e3, tu, u5bytes / tu / 1e3, t5 / tu,
                    (bad == 0 ? "finite" : ("NONFINITE " + std::to_string(bad)).c_str()));
    }
    return 0;
}