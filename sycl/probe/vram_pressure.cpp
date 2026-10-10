// vram_pressure: is the plain-SYCL fallback expert GEMM slower when the card is full?  It is not.
//
// #1549 found the prompt path's per-expert GEMM cost going from 2.3 ms with the expert cache at
// 23.83 GiB to 54-122 ms at 28.82 GiB, on the same calls and shapes, and the phase timing charged both
// expert GEMM phases with it. This measures the engine's own Gemm::f16 at the prompt path's shapes and
// routed row counts, with the card empty and again after allocating and TOUCHING 28 GiB of it.
//
//   gate/up 1280x2560  empty  1473 us/call   28 GiB  1463 us/call   (0.99x)
//   down    2560x640   empty   382 us/call   28 GiB   381 us/call   (1.00x)
//
// So a full card does not slow this kernel, and the 111 ms per expert pair the slow arm charged to these
// phases was not GEMM work. The probe was what settled that; keep it for the next card that suspects the
// same thing. It needs a Level Zero loader to start on Windows (ZEL_LIBRARY_PATH, as the engine does) -
// the OpenCL backend has no free-VRAM query, which is the finding on the other side of #1549.
//
//   vram_pressure [calls=800] [fill_gib=28]
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/prefill/gemm.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <dxgi.h>
#include <dxgi1_4.h>
// Does DXGI give a live, per-process dedicated-VRAM figure on this backend?  If it does, it is the
// answer to the OpenCL backend's missing free-VRAM query (#1549): the engine's own usage plus the
// card's budget is the live `free` figure that dpct's get_memory_info cannot produce here.
static void dxgi_state(const char* when) {
    IDXGIFactory* f = nullptr;
    if (FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory), (void**) &f))) { std::printf("DXGI (%s): no factory\n", when); return; }
    IDXGIAdapter* a = nullptr;
    if (FAILED(f->EnumAdapters(0, &a))) { std::printf("DXGI (%s): no adapter\n", when); f->Release(); return; }
    IDXGIAdapter3* a3 = nullptr;
    if (SUCCEEDED(a->QueryInterface(__uuidof(IDXGIAdapter3), (void**) &a3))) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        if (SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
            std::printf("DXGI (%s): this process uses %6.2f GiB of a %6.2f GiB budget (%.2f GiB left)\n", when,
                        (double) info.CurrentUsage / 1073741824.0, (double) info.Budget / 1073741824.0,
                        (double) (info.Budget > info.CurrentUsage ? info.Budget - info.CurrentUsage : 0) / 1073741824.0);
        else std::printf("DXGI (%s): QueryVideoMemoryInfo failed\n", when);
        a3->Release();
    } else std::printf("DXGI (%s): no IDXGIAdapter3\n", when);
    a->Release();
    f->Release();
}
#else
static void dxgi_state(const char*) {}
#endif

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // a crash must not take the log with it
    const int calls = argc > 1 ? std::atoi(argv[1]) : 800;
    const double fill_gib = argc > 2 ? std::atof(argv[2]) : 28.0;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    const int64_t N = 2560;

    // One gate/up (1280x2560) and one down (2560x640) product, at a routed-expert row count. The prompt
    // path's mean was 8-11 rows per expert on the runs this is chasing.
    const int64_t T = 10;
    uint16_t *X = sycl::malloc_device<uint16_t>((size_t) T * N, q);
    uint16_t *W = sycl::malloc_device<uint16_t>((size_t) N * 1280, q);
    // Y is the wider of the two outputs ([T, N] for the down product), or a run overflows it.
    float* Y = sycl::malloc_device<float>((size_t) T * N, q);
    if (!X || !W || !Y) { std::printf("alloc failed\n"); return 1; }
    q.memset(X, 0, (size_t) T * N * 2).wait();

    strata::prefill::Gemm gemm;
    std::string err;
    // The scratch is the largest weight dequantized at once: N*1280 elements.
    if (!gemm.init(&q, N * 1280, err)) {
        std::printf("Gemm::init failed: %s\n", err.c_str());
        return 1;
    }

    auto timed = [&](int64_t n_out, int64_t k) {
        // warm, then time `calls` back-to-back
        for (int i = 0; i < 25; ++i) gemm.f16(X, W, Y, T, n_out, k);
        q.wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < calls; ++i) gemm.f16(X, W, Y, T, n_out, k);
        q.wait();
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / calls;
    };

    // One expert-sized weight, in FP16, at the two prompt-path shapes.
    const int64_t total_b = (int64_t) q.get_device().get_info<sycl::info::device::global_mem_size>();
    std::printf("card: %.2f GiB total, %d compute units\n", (double) total_b / 1073741824.0,
                (int) q.get_device().get_info<sycl::info::device::max_compute_units>());

    std::printf("--- card otherwise empty ---\n");
    dxgi_state("before anything");
    const double gu_empty = timed(1280, N);
    const double dn_empty = timed(N, 640);
    std::printf("gate/up 1280x2560  %.1f us/call\ndown 2560x640        %.1f us/call\n", gu_empty, dn_empty);

    const int64_t fill_b = (int64_t) (fill_gib * 1073741824.0);
    if (fill_b >= total_b - (2ll << 30)) { std::printf("fill of %.1f GiB leaves nothing; skipping\n", fill_gib); return 0; }
    // The fill is allocated, then touched, so the pages are really committed - under WDDM an
    // allocation is not resident until it is written.
    uint8_t* fill = sycl::malloc_device<uint8_t>((size_t) fill_b, q);
    if (!fill) { std::printf("alloc of %.2f GiB failed\n", fill_gib); return 1; }
    q.memset(fill, 1, (size_t) fill_b).wait();
    dxgi_state("after 28 GiB fill");
    size_t free_b = 0, tot_b = 0;
    dpct::get_current_device().get_memory_info(free_b, tot_b);
    std::printf("--- after allocating and touching %.2f GiB (dpct free figure now %.2f GiB) ---\n",
                fill_gib, (double) free_b / 1073741824.0);
    const double gu_fill = timed(1280, N);
    const double dn_fill = timed(N, 640);
    std::printf("gate/up 1280x2560  %.1f us/call  (%.2fx)\ndown 2560x640        %.1f us/call  (%.2fx)\n",
                gu_fill, gu_fill / gu_empty, dn_fill, dn_fill / dn_empty);
    sycl::free(fill, q);
    std::printf("done\n");
    return 0;
}
