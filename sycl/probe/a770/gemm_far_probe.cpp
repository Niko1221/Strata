// gemm_far_probe: oneMKL fp16 GEMM correctness with operands far into ONE malloc_device (BIG=1) and into a range of
// physical segments behind one reserved address range (the segmented expert-cache arena).  gemm_far_probe <GiB> <segment GiB>
// Build: icpx -fsycl -std=c++20 -O2 gemm_far_probe.cpp -o gemm_far_probe -lmkl_sycl_blas -lmkl_intel_lp64 -lmkl_tbb_thread -lmkl_core -ltbb
// A770 16 GB, xe, compute runtime 26.05, oneAPI 2026.1, 2026-10-14 (BIG=1 UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 ... 8 0.5):
// one 8 GiB malloc_device, T = 7987: X past 4 GiB -> the last partial tile wrong; W and X past 4 GiB -> nearly every
// element wrong (N 10240, K 320 / 2560); T = 313 right.  16 x 0.5 GiB segments: every case right (also 0.25 / 2 GiB
// segments, W / X / Y anywhere, fp16 and bf16, operands across a segment boundary).
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#include <oneapi/mkl.hpp>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
namespace sx = sycl::ext::oneapi::experimental;
using half = sycl::half; using bf16 = oneapi::mkl::bfloat16;
constexpr size_t GiB = 1ull << 30;
template <class T> void run(sycl::queue& q, uint8_t* arena, size_t total, const char* tn) {
    std::mt19937 rng(1); std::uniform_real_distribution<float> ud(-1, 1);
    struct Sh { int N, K; } shs[] = {{10240, 320}, {10240, 2560}, {1280, 2560}};
    for (auto sh : shs) for (int Tt : {313, 7987}) for (double og : {2.0, 5.4, 6.04}) for (int far : {2, 3}) {
        const int N = sh.N, K = sh.K;
        const size_t wb = (size_t) N * K * sizeof(T), xb = (size_t) Tt * K * sizeof(T), yb = (size_t) N * Tt * 4;
        size_t off = (size_t) (og * GiB) / 256 * 256; if (off + wb + xb + yb > total) off = (total - wb - xb - yb) / 256 * 256;
        std::vector<T> hw((size_t) N * K), hx((size_t) Tt * K);
        for (auto& v : hw) v = (T) ud(rng); for (auto& v : hx) v = (T) ud(rng);
        T* W0 = sycl::malloc_device<T>(hw.size(), q); T* X0 = sycl::malloc_device<T>(hx.size(), q); float* Y0 = sycl::malloc_device<float>((size_t) N * Tt, q);
        q.memcpy(W0, hw.data(), wb); q.memcpy(X0, hx.data(), xb).wait();
        T* Wf = (T*) (arena + off); T* Xf = (T*) (arena + off + wb); float* Yf = (float*) (arena + off + wb + xb);
        q.memcpy(Wf, W0, wb); q.memcpy(Xf, X0, xb).wait();
        auto g = [&](T* W, T* X, float* Y) { q.memset(Y, 0xff, yb); oneapi::mkl::blas::column_major::gemm(q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, N, Tt, K, 1.0f, W, K, X, K, 0.0f, Y, N); q.wait(); };
        std::vector<float> yr((size_t) N * Tt), y((size_t) N * Tt);
        g(W0, X0, Y0); q.memcpy(yr.data(), Y0, yb).wait();
        float* Yt = (far & 4) ? Yf : Y0;
        g((far & 1) ? Wf : W0, (far & 2) ? Xf : X0, Yt); q.memcpy(y.data(), Yt, yb).wait();
        size_t bad = 0; for (size_t i = 0; i < y.size(); ++i) if (!(std::fabs(y[i] - yr[i]) <= 1e-3f * (1 + std::fabs(yr[i])))) ++bad;
        if (bad) std::printf("WRONG %s N=%d K=%d T=%d off=%.2f GiB far=%s%s%s: %zu of %zu\n", tn, N, K, Tt, off / (double) GiB, far & 1 ? "W" : "", far & 2 ? "X" : "", far & 4 ? "Y" : "", bad, y.size());
        sycl::free(W0, q); sycl::free(X0, q); sycl::free(Y0, q);
    }
    std::printf("%s sweep done\n", tn);
}
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const double ag = argc > 1 ? std::atof(argv[1]) : 10.5, sg = argc > 2 ? std::atof(argv[2]) : 0.5;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    auto ctx = q.get_context(); auto dev = q.get_device();
    const size_t gran = sx::get_mem_granularity(dev, ctx, sx::granularity_mode::recommended);
    const size_t seg = ((size_t) (sg * GiB) + gran - 1) / gran * gran, total = ((size_t) (ag * GiB)) / seg * seg;
    if (std::getenv("BIG")) { uint8_t* big = sycl::malloc_device<uint8_t>(total, q); std::printf("one malloc_device %s\n", big ? "ok" : "failed"); if (big) { run<half>(q, big, total, "f16 malloc"); sycl::free(big, q); } }
    const uintptr_t va = sx::reserve_virtual_mem(total, ctx);
    std::vector<sx::physical_mem> phys;
    for (size_t at = 0; at < total; at += seg) { phys.emplace_back(dev, ctx, seg); phys.back().map(va + at, seg, sx::address_access_mode::read_write); }
    std::printf("arena %.2f GiB in %zu segments\n", total / (double) GiB, phys.size());
    run<half>(q, (uint8_t*) va, total, "f16"); 
    return 0;
}
