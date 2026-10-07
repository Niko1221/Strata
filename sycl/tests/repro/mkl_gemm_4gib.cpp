// Reproducer: oneMKL's mixed-precision GEMM (bf16 in, fp32 out, through dpct::blas::gemm) returns zeros when an input
// matrix starts 4 GiB or more into a device allocation. Where the output sits does not matter. Intel Arc A770 (dg2-g10).
// Build: icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device dg2-g10 -options \"-ze-opt-greater-than-4GB-buffer-required\"" -qmkl=sequential -Isycl/include mkl_gemm_4gib.cpp
// Run with UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1. Expected: every row "mismatches 0"; observed: X or W above 4 GiB gives zeros.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cstring>
static const size_t GiB = 1ull << 30, MiB = 1ull << 20;
static uint16_t bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t(u >> 16); }

int main() {
    auto& q = dpct::get_in_order_queue();
    dpct::blas::descriptor_ptr h = new dpct::blas::descriptor();
    h->set_queue(&q);
    const int T = 2801, N = 10240, K = 320;
    std::vector<uint16_t> hx((size_t)T * K), hw((size_t)N * K);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = bf(0.02f * float(int((i * 7) % 13) - 6));
    for (size_t i = 0; i < hw.size(); ++i) hw[i] = bf(0.02f * float(int((i * 5) % 11) - 5));
    const size_t total = 9 * GiB;
    char* arena = (char*)sycl::malloc_device(total, q);
    if (!arena) { std::printf("alloc failed\n"); return 2; }
    const size_t ysz = (size_t)T * N * 4;
    uint16_t* dx0 = (uint16_t*)sycl::malloc_device(hx.size() * 2, q);
    uint16_t* dw0 = (uint16_t*)sycl::malloc_device(hw.size() * 2, q);
    float* yref = (float*)sycl::malloc_device(ysz, q);
    q.memcpy(dx0, hx.data(), hx.size() * 2); q.memcpy(dw0, hw.data(), hw.size() * 2).wait();
    const float alpha = 1.f, beta0 = 0.f;
    auto run = [&](const uint16_t* X, const uint16_t* W, float* Y) {
        dpct::blas::gemm(h, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, N, T, K, &alpha, W,
                         dpct::library_data_t::real_bfloat16, K, X, dpct::library_data_t::real_bfloat16, K, &beta0, Y,
                         dpct::library_data_t::real_float, N, dpct::compute_type::f32);
        q.wait();
    };
    q.memset(yref, 0, ysz).wait();
    run(dx0, dw0, yref);
    std::vector<float> ref(ysz / 4), got(ysz / 4);
    q.memcpy(ref.data(), yref, ysz).wait();
    size_t refbad = 0; for (float v : ref) refbad += !std::isfinite(v);
    std::printf("reference (separate allocs): %zu non-finite\n", refbad);
    struct P { const char* name; size_t xoff, woff, yoff; } ps[] = {
        {"X,W,Y low (64MiB)", 64*MiB, 128*MiB, 256*MiB},
        {"X,W low; Y @7.4GiB", 64*MiB, 128*MiB, size_t(7.4*GiB)/256*256},
        {"X @4.5GiB; W,Y low", size_t(4.5*GiB), 128*MiB, 256*MiB},
        {"W @4.5GiB; X,Y low", 64*MiB, size_t(4.5*GiB), 256*MiB},
        {"X,W @3.5GiB; Y low", size_t(3.5*GiB), size_t(3.5*GiB)+64*MiB, 256*MiB},
        {"X,W @4.5GiB; Y low", size_t(4.5*GiB), size_t(4.5*GiB)+64*MiB, 256*MiB},
        {"X,W @7.4GiB; Y low", size_t(7.4*GiB)/256*256, size_t(7.4*GiB)/256*256+64*MiB, 256*MiB},
        {"X,W,Y @7.4GiB+", size_t(7.4*GiB)/256*256, size_t(7.4*GiB)/256*256+64*MiB, size_t(7.4*GiB)/256*256+128*MiB}};
    int bad = 0;
    for (auto& c : ps) {
        uint16_t* X = (uint16_t*)(arena + c.xoff); uint16_t* W = (uint16_t*)(arena + c.woff); float* Y = (float*)(arena + c.yoff);
        q.memcpy(X, hx.data(), hx.size() * 2); q.memcpy(W, hw.data(), hw.size() * 2); q.memset(Y, 0, ysz).wait();
        run(X, W, Y);
        q.memcpy(got.data(), Y, ysz).wait();
        size_t nb = 0, mism = 0, zeros = 0; for (size_t i = 0; i < got.size(); ++i) { if (!std::isfinite(got[i])) ++nb; else if (got[i] != ref[i]) ++mism; zeros += got[i] == 0.f; }
        std::printf("%-22s non-finite %zu  mismatches %zu  zeros %zu\n", c.name, nb, mism, zeros);
        bad += (nb || mism);
    }
    std::printf(bad ? "RESULT: %d BAD\n" : "RESULT: all ok\n", bad);
    return bad != 0;
}
