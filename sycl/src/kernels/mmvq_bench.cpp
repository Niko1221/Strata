// mmvq_bench [n_in] [n_out] [ncols], or --selftest for parity without timing.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace {

bool compare(const char* label, const std::vector<float>& ref, const std::vector<float>& got,
             double atol = 1e-3, double rtol = 1e-5) {
    size_t bad = 0, first = 0;
    double worst = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double diff = std::fabs(double(ref[i]) - got[i]);
        if (!std::isfinite(ref[i]) || !std::isfinite(got[i]) || diff > atol + rtol * std::fabs(ref[i])) {
            if (bad++ == 0) first = i;
        }
        if (diff > worst) worst = diff;
    }
    std::printf("%s: %s (%zu/%zu off, max abs %.3e)\n", label, bad ? "FAIL" : "ok", bad, ref.size(), worst);
    if (bad) std::fprintf(stderr, "  first mismatch [%zu]: tag 14 %g, candidate %g\n", first, ref[first], got[first]);
    return bad == 0;
}

bool check_dequant(sycl::queue& s, const void* w, const void* wp, int n_in, int n_out) {
    const size_t count = size_t(n_in) * n_out;
    float* df = sycl::malloc_device<float>(count, s);
    uint16_t* dh = sycl::malloc_device<uint16_t>(count, s);
    if (!df || !dh) throw std::bad_alloc();
    bool ok = true;
    for (int row0 : {0, 2}) {
        const int rows = n_out - row0;
        std::vector<float> ref(size_t(rows) * n_in), got(ref.size());
        std::vector<uint16_t> h(ref.size());
        for (int format = 0; format < 3; ++format) {
            for (int packed = 0; packed < 2; ++packed) {
                auto& out = packed ? got : ref;
                const int tag = packed ? strata::kernels::kNativeQ6KStride224 : 14;
                const void* weights = packed ? wp : w;
                if (format == 0) {
                    s.fill(df, std::numeric_limits<float>::quiet_NaN(), out.size());
                    strata::kernels::dequant_f32(tag, weights, row0, rows, n_in, df, &s);
                    s.memcpy(out.data(), df, out.size() * sizeof(float)).wait_and_throw();
                } else {
                    s.fill(dh, uint16_t(0x7fff), h.size());
                    if (format == 1)
                        strata::kernels::dequant_bf16(tag, weights, row0, rows, n_in, dh, &s);
                    else
                        strata::kernels::dequant_f16(tag, weights, row0, rows, n_in, dh, &s);
                    s.memcpy(h.data(), dh, h.size() * sizeof(uint16_t)).wait_and_throw();
                    for (size_t i = 0; i < h.size(); ++i) {
                        if (format == 1) {
                            const uint32_t bits = uint32_t(h[i]) << 16;
                            std::memcpy(&out[i], &bits, sizeof(float));
                        } else {
                            sycl::half value;
                            std::memcpy(&value, &h[i], sizeof(value));
                            out[i] = float(value);
                        }
                    }
                }
            }
            std::printf("  row0=%d rows=%d ", row0, rows);
            ok = compare(format == 0 ? "dequant f32" : format == 1 ? "dequant bf16" : "dequant f16",
                         ref, got, 0, 0) && ok;
        }
    }
    sycl::free(df, s);
    sycl::free(dh, s);
    return ok;
}

bool run_case(int n_in, int n_out, int maxcols, bool timing) {
    auto& s = dpct::get_in_order_queue();
    const size_t wbytes = strata::kernels::native_mmvq_weight_bytes(14, n_in, n_out);
    std::vector<uint8_t> hw(wbytes), hp(strata::kernels::native_mmvq_weight_bytes(
        strata::kernels::kNativeQ6KStride224, n_in, n_out));
    for (size_t i = 0; i < wbytes; ++i) hw[i] = uint8_t(i * 2654435761u >> 13);
    for (size_t b = 0; b < wbytes / 210; ++b) {
        const uint16_t d = uint16_t(0x1c00 + (b % 4) * 0x400);  // finite scales: 2^-8 through 2^-5
        hw[b * 210 + 208] = uint8_t(d);
        hw[b * 210 + 209] = uint8_t(d >> 8);
    }
    strata::kernels::native_mmvq_pack(strata::kernels::kNativeQ6KStride224, hw.data(), n_in, n_out, hp.data());
    std::vector<float> hx(size_t(maxcols) * n_in);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = float((i * 7919) % 1000) / 500.f - 1.f;
    void* w = sycl::malloc_device(hw.size(), s);
    void* wp = sycl::malloc_device(hp.size(), s);
    float* x = sycl::malloc_device<float>(hx.size(), s);
    void* xq = sycl::malloc_device(strata::kernels::native_q8_1_bytes(n_in, maxcols), s);
    float* y = sycl::malloc_device<float>(size_t(maxcols) * n_out, s);
    if (!w || !wp || !x || !xq || !y) throw std::bad_alloc();
    s.memcpy(w, hw.data(), hw.size());
    s.memcpy(wp, hp.data(), hp.size());
    s.memcpy(x, hx.data(), hx.size() * sizeof(float)).wait_and_throw();
    strata::kernels::native_quantize_q8_1(x, xq, n_in, maxcols, &s);
    bool ok = true;
    for (int ncols = timing ? maxcols : 1; ncols <= maxcols; ++ncols) {
        std::printf("Q6_K %d x %d, %d cols\n", n_in, n_out, ncols);
        std::vector<float> ref(size_t(ncols) * n_out), got(ref.size());
        auto result = [&](int tag, const void* weights, std::vector<float>& out) {
            s.fill(y, std::numeric_limits<float>::quiet_NaN(), out.size());
            strata::kernels::native_mmvq(tag, weights, xq, y, n_in, n_out, ncols, &s);
            s.memcpy(out.data(), y, out.size() * sizeof(float)).wait_and_throw();
        };
        strata::kernels::native_mmvq_set_q6k_wide(false);
        result(14, w, ref);
        strata::kernels::native_mmvq_set_q6k_wide(true);
        result(14, w, got);
        ok = compare("tag 14 wide on vs off", ref, got) && ok;
        result(strata::kernels::kNativeQ6KStride224, wp, got);
        ok = compare("tag 114 vs tag 14 wide off", ref, got) && ok;
        result(14, w, ref);
        ok = compare("tag 114 vs tag 14 wide on", ref, got) && ok;
        const char* wide = std::getenv("STRATA_MMVQ_WIDE");
        strata::kernels::native_mmvq_set_q6k_wide(!wide || std::atoi(wide) != 0);
        if (timing && ok) {
            auto time = [&](int tag, const void* weights) {
                const auto warm = std::chrono::steady_clock::now();
                while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - warm).count() < 300) {
                    for (int i = 0; i < 20; ++i)
                        strata::kernels::native_mmvq(tag, weights, xq, y, n_in, n_out, ncols, &s);
                    s.wait_and_throw();
                }
                const auto start = std::chrono::steady_clock::now();
                for (int i = 0; i < 400; ++i)
                    strata::kernels::native_mmvq(tag, weights, xq, y, n_in, n_out, ncols, &s);
                s.wait_and_throw();
                return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 400;
            };
            const double us = time(14, w), us2 = time(strata::kernels::kNativeQ6KStride224, wp);
            std::printf("tag 14: %.1f us, %.1f GB/s; tag 114: %.1f us, %.1f GB/s; speedup %.2fx (210-byte weights)\n",
                        us, wbytes / us / 1e3, us2, wbytes / us2 / 1e3, us / us2);
        }
    }
    if (!timing) ok = check_dequant(s, w, wp, n_in, n_out) && ok;
    sycl::free(w, s);
    sycl::free(wp, s);
    sycl::free(x, s);
    sycl::free(xq, s);
    sycl::free(y, s);
    return ok;
}

}  // namespace

int main(int argc, char** argv) try {
    const bool selftest = argc == 2 && std::strcmp(argv[1], "--selftest") == 0;
    const int n_in = argc > 1 && !selftest ? std::atoi(argv[1]) : 2560;
    const int n_out = argc > 2 ? std::atoi(argv[2]) : 2560;
    const int ncols = argc > 3 ? std::atoi(argv[3]) : 4;
    if (argc > 4 || n_in <= 0 || n_in % 256 || n_out <= 0 || ncols < 1 || ncols > 6) {
        std::fprintf(stderr, "usage: mmvq_bench [n_in (multiple of 256)] [n_out > 0] [ncols 1-6], or --selftest\n");
        return 2;
    }
    bool ok = true;
    for (int blocks : {1, 3, 4, 5, 10}) ok = run_case(blocks * 256, 17, 6, false) && ok;
    if (!selftest && ok) ok = run_case(n_in, n_out, ncols, true);
    return ok ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "mmvq_bench: %s\n", e.what());
    return 1;
}
