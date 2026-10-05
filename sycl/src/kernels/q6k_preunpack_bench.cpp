// q6k_preunpack_bench: P0 #2 avenue 1 ceiling - how much does the Q6_K decode's in-kernel 6-bit unpack actually cost?
//
// exp 26/ISA (on this card dp4a runs on the same ALU int pipe as the unpack): the a2 Q6_K wide kernel's pipe work is
// dominated by the unpack/gather (bfn 74, xor 65, shl/and/shr/or 63, plus ~120 per-byte mov) against only 32 dp4a.
// The ql/qh byte-alignment mismatch forces the per-byte gather.  The decisive test is to time the *no-unpack* ceiling
// with the SAME decode shape but signed-byte weights - which is exactly the shipped Q8_0 wide32 kernel (d + raw int8,
// load + dp4a only).  If a one-time *pre-unpack* of Q6_K rows to signed bytes (`wide32` on the re-encoded rows) is
// meaningfully faster at the engine's ncols, that is the avenue-1 win (persistent unpacked weight buffer); if it is a
// wash, the unpack is not the pipe cost and avenue 1 is null (the mov/add accumulate + addressing dominate).
//
// The Q8_0 path reads 272 B per 256 weights vs Q6_K's 210 (1.30x memory), so per-call **us** is the honest metric
// (the card has ~2x bandwidth headroom, exp 22/23: Send 0%).  Both kernels are production paths; no correctness gate
// is asserted here beyond clean dispatch and finite outputs (native_mmvq wide32 correctness is covered by the parity
// tests).
//
//   q6k_preunpack_bench [n_in=2560] [n_out=2560] [reps=400]
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
    const size_t w6bytes = (size_t) n_out * (n_in / 256) * 210;
    const size_t w8bytes = (size_t) n_out * (n_in / 32) * 34;

    sycl::queue* q = &dpct::get_in_order_queue();

    std::vector<uint8_t> hw6(w6bytes);                 // Q6_K rows, d=1.0 (like mmvq_bench)
    for (size_t i = 0; i < w6bytes; ++i) hw6[i] = (uint8_t) (i * 2654435761u >> 13);
    for (size_t b = 0; b < w6bytes / 210; ++b) { hw6[b * 210 + 208] = 0x00; hw6[b * 210 + 209] = 0x3c; }
    std::vector<uint8_t> hw8(w8bytes);                 // Q8_0 rows: {half d=1.0, 32 int8}
    for (size_t b = 0; b + 33 < w8bytes; b += 34) { hw8[b] = 0x00; hw8[b + 1] = 0x3c; for (int i = 0; i < 32; ++i) hw8[b + 2 + i] = (uint8_t) ((b * 2654435761u >> 13) + i); }
    void* dw6 = sycl::malloc_device(w6bytes, *q);
    void* dw8 = sycl::malloc_device(w8bytes, *q);
    q->memcpy(dw6, hw6.data(), w6bytes).wait();
    q->memcpy(dw8, hw8.data(), w8bytes).wait();
    const size_t u6bytes = strata::kernels::native_mmvq_q6k_preunpack_bytes(n_in, n_out);
    void* du6 = sycl::malloc_device(u6bytes, *q);          // the deployed pre-unpacked Q6_K decode
    strata::kernels::native_q6k_preunpack(dw6, du6, n_in, n_out, q);
    q->wait();

    const int nmax = cols[4];
    std::vector<float> xf((size_t) nmax * n_in);
    for (size_t t = 0; t < (size_t) nmax; ++t)
        for (size_t k = 0; k < (size_t) n_in; ++k) xf[k + t * n_in] = (float) ((int) ((k * 7919 + t * 104729) % 1000) - 500) / 250.f;
    void* xq = sycl::malloc_device((size_t) nmax * (n_in / 32) * 36, *q);
    float* y6 = sycl::malloc_device<float>((size_t) nmax * n_out, *q);
    float* y8 = sycl::malloc_device<float>((size_t) nmax * n_out, *q);

    for (int ci = 0; ci < 5; ++ci) {
        const int ncols = cols[ci];
        strata::kernels::native_quantize_q8_1(xf.data(), xq, n_in, ncols, q);
        auto q6 = [&]() { strata::kernels::native_q6_k_mmvq(dw6, xq, y6, n_in, n_out, ncols, q); };
        auto q8 = [&]() { strata::kernels::native_q8_0_mmvq(dw8, xq, y8, n_in, n_out, ncols, q); };
        auto qu = [&]() { strata::kernels::native_mmvq_q6k_unpacked(du6, xq, y6, n_in, n_out, ncols, q); };  // deployed routed path
        q6(); q8(); qu(); q->wait();
        std::vector<float> r6((size_t) ncols * n_out), r8((size_t) ncols * n_out);
        q->memcpy(r6.data(), y6, r6.size() * 4).wait();
        q->memcpy(r8.data(), y8, r8.size() * 4).wait();
        int bad = 0; for (float v : r8) if (!std::isfinite(v)) ++bad;
        const double w0 = std::chrono::steady_clock::now().time_since_epoch().count();
        auto warm = [&](auto&& fn) { (void) w0; const auto s = std::chrono::steady_clock::now();
            while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count() < 300) { for (int i = 0; i < 20; ++i) fn(); q->wait(); } };
        warm(q6); warm(q8); warm(qu);
        auto time = [&](auto&& fn) { const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) fn(); q->wait();
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps; };
        const double t6 = time(q6), t8 = time(q8), tu = time(qu);
        std::printf("%2d cols:  Q6_K unpack %6.1f us | Q6K-unpacked(routed) %6.1f us %.2fx | Q8_0 ceiling %6.1f us %.2fx  [%s]\n",
                    ncols, t6, tu, t6 / tu, t8, t6 / t8,
                    (bad == 0 ? "finite" : ("NONFINITE " + std::to_string(bad)).c_str()));
    }
    return 0;
}