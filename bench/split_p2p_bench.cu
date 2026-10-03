// bench/split_p2p_bench.cu - the layer-split hand-off, every path it could take, measured on one machine.
//
// WHAT THIS ANSWERS.  A --layer-split hands one window's activation from one card to the next.  Today that
// hand-off crosses through PINNED HOST RAM (docs/MULTI_GPU.md: "No NVLink or peer-to-peer access is used"):
//
//   * verify windows (src/core/verify.cpp): two copy_from_mapped kernels per row-group per token - the first
//     writes device -> mapped pinned host (record_window's hand-off block), the second reads mapped host ->
//     device on the other card (the window's input block).  Both kernels are inside the captured window graph.
//   * prompt chunks (src/prefill/prefill.cpp): cudaMemcpyAsync D2H into a pinned buffer + a stream sync on the
//     writing stage, then cudaMemcpyAsync H2D on the reading stage's own host thread.
//
// On a machine whose two cards sit on NVLink (nvidia-smi topo: NV#), a direct device-to-device copy should cost
// one link traversal instead of two PCIe ones.  This bench times, for the real transfer shapes:
//
//   p2p-memcpy    cudaMemcpyPeerAsync (copy engine; needs cudaDeviceEnablePeerAccess)
//   p2p-kernel    a copy kernel on the source card storing into the peer card's memory (in-graph friendly)
//   d2d-auto      cudaMemcpyAsync DeviceToDevice with peer access DISABLED (the driver stages through host)
//   pin-bounce    cudaMemcpyAsync D2H + H2D through one pinned buffer (the prefill path's shape)
//   map-bounce    kernel write to mapped pinned + kernel read from it (the verify path's shape)
//
// Each row reports the wall-clock cost of one hand-off as the engine takes it (a sync between the two halves of
// a bounce, like the two stages do) and the useful bandwidth in GB/s.  A final section shows whether
// cudaMemcpyPeerAsync survives cudaStreamBeginCapture - the verify hand-off sits inside the captured window
// graph, so the engine's P2P path is only a drop-in if it does.
//
// BUILD (no strata libraries, seconds):
//   nvcc -O3 -arch=sm_70 -o /tmp/split_p2p_bench bench/split_p2p_bench.cu
// RUN (two GPUs; the bench enables peer access itself):
//   CUDA_VISIBLE_DEVICES=0,1 /tmp/split_p2p_bench
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        const cudaError_t e_ = (x);                                                                                    \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));                    \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

namespace {

__global__ void copy_f32(float* __restrict__ dst, const float* __restrict__ src, int64_t n4) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (int64_t) gridDim.x * blockDim.x)
        ((float4*) dst)[i] = ((const float4*) src)[i];
}

// the verify path's copy_from_mapped_kernel, byte for byte its access pattern (volatile load, float4 store)
__global__ void copy_from_mapped_kernel(float4* __restrict__ dst, const volatile float4* src, int64_t n4) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (int64_t) gridDim.x * blockDim.x) {
        const float4 v = const_cast<const float4*>(src)[i];
        dst[i] = v;
    }
}

struct Shape {
    const char* name;
    int64_t bytes;
};

struct Row {
    std::string path;
    double ms;   // one hand-off, wall clock, launch + the sync the engine needs
    double gbs;  // useful GB/s (one payload size, not the 2x a bounce moves over PCIe)
};

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void launch_copy(void* dst, const void* src, size_t bytes, cudaStream_t s, bool mapped_style) {
    const int64_t n4 = (int64_t) (bytes / 16);
    const int blocks = (int) std::min<int64_t>((n4 + 255) / 256, 4096);
    if (mapped_style) copy_from_mapped_kernel<<<blocks, 256, 0, s>>>((float4*) dst, (const volatile float4*) src, n4);
    else copy_f32<<<blocks, 256, 0, s>>>((float*) dst, (const float*) src, n4);
}

// one timed hand-off: `warmup` untimed, then `iters` timed around launch + the syncs the engine's path needs
template <class F>
void time_path(const char* name, const Shape& sh, int warmup, int iters, F&& one_handoff, std::vector<Row>& out) {
    for (int i = 0; i < warmup; ++i) one_handoff();
    CHECK(cudaDeviceSynchronize());
    const double t0 = now_ms();
    for (int i = 0; i < iters; ++i) one_handoff();
    CHECK(cudaDeviceSynchronize());
    const double ms = (now_ms() - t0) / iters;
    out.push_back({name, ms, (double) sh.bytes / (ms * 1e6)});
}

}  // namespace

int main() {
    int n_dev = 0;
    CHECK(cudaGetDeviceCount(&n_dev));
    if (n_dev < 2) {
        std::fprintf(stderr, "this bench needs two visible GPUs (CUDA_VISIBLE_DEVICES=0,1)\n");
        return 1;
    }
    const int D0 = 0, D1 = 1;
    int can01 = 0, can10 = 0;
    CHECK(cudaDeviceCanAccessPeer(&can01, D0, D1));
    CHECK(cudaDeviceCanAccessPeer(&can10, D1, D0));
    cudaDeviceProp p0{}, p1{};
    CHECK(cudaGetDeviceProperties(&p0, D0));
    CHECK(cudaGetDeviceProperties(&p1, D1));
    std::printf("# %s (sm_%d%d) <-> %s (sm_%d%d); peer access: %d/%d\n", p0.name, p0.major, p0.minor, p1.name,
                p1.major, p1.minor, can01, can10);

    // the real hand-off shapes.  Geometry (include/strata/core/layout.hpp): n_embd=2560, hc=4, so one verify
    // token hands hc*n_embd + n_embd + hc = 12804 floats (Verifier::handoff_floats) and one prompt chunk hands
    // chunk * hc*n_embd = chunk * 10240 floats (Prefill's D).  kVerifyMaxT = 8, prompt chunks 512..2560.
    const int64_t F = 4;   // float
    const std::vector<Shape> shapes = {
        {"verify 2 tokens (100 KB)", 2 * 12804 * F},
        {"verify 4 tokens (200 KB)", 4 * 12804 * F},
        {"verify 8 tokens (400 KB)", 8 * 12804 * F},
        {"2560x256 float (2.5 MB)", 2560 * 256 * F},
        {"2560x4096 float (40 MB)", 2560 * 4096 * F},
        {"prompt chunk 512 (20 MB)", 512 * 10240 * F},
        {"prompt chunk 2048 (84 MB)", 2048 * 10240 * F},
        {"prompt chunk 2560 (104 MB)", 2560 * 10240 * F},
    };
    const size_t Smax = (size_t) shapes.back().bytes;

    // buffers: source and staging on card 0, destination on card 1, one pinned + one mapped pinned host pair
    float *d0 = nullptr, *d0b = nullptr, *d1 = nullptr;
    CHECK(cudaSetDevice(D0));
    CHECK(cudaMalloc(&d0, Smax));
    CHECK(cudaMalloc(&d0b, Smax));
    float *h_pin = nullptr, *h_map = nullptr, *h_map_dev = nullptr;
    CHECK(cudaHostAlloc(&h_pin, Smax, cudaHostAllocPortable));
    CHECK(cudaHostAlloc(&h_map, Smax, cudaHostAllocMapped | cudaHostAllocPortable));
    CHECK(cudaHostGetDevicePointer((void**) &h_map_dev, h_map, 0));
    CHECK(cudaSetDevice(D1));
    CHECK(cudaMalloc(&d1, Smax));
    CHECK(cudaSetDevice(D0));
    cudaStream_t s0, s1;
    CHECK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    CHECK(cudaSetDevice(D1));
    CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CHECK(cudaSetDevice(D0));
    std::memset(h_pin, 1, Smax);
    std::memset(h_map, 1, Smax);

    std::vector<Row> rows;
    std::vector<float> pat(Smax / 4), got(Smax / 4);
    for (size_t i = 0; i < pat.size(); ++i) pat[i] = (float) (i % 1000) + 0.25f;
    CHECK(cudaMemcpy(d0, pat.data(), Smax, cudaMemcpyHostToDevice));
    auto same = [&](const char* what) {
        CHECK(cudaMemcpy(got.data(), d1, Smax, cudaMemcpyDeviceToHost));
        if (std::memcmp(got.data(), pat.data(), Smax) != 0) {
            std::fprintf(stderr, "MISMATCH: %s did not copy the pattern\n", what);
            std::exit(1);
        }
    };

    // ---- correctness, peer access still off: the paths that must not need the link ----
    {
        const size_t B = Smax;
        CHECK(cudaMemcpyAsync(d1, d0, B, cudaMemcpyDeviceToDevice, s0));
        CHECK(cudaStreamSynchronize(s0));
        same("d2d-auto");
        CHECK(cudaMemcpyAsync(h_pin, d0, B, cudaMemcpyDeviceToHost, s0));
        CHECK(cudaStreamSynchronize(s0));
        CHECK(cudaMemcpyAsync(d1, h_pin, B, cudaMemcpyHostToDevice, s1));
        CHECK(cudaStreamSynchronize(s1));
        same("pin-bounce");
        launch_copy(h_map_dev, d0, B, s0, true);
        CHECK(cudaStreamSynchronize(s0));
        launch_copy(d1, h_map_dev, B, s1, true);
        CHECK(cudaStreamSynchronize(s1));
        same("map-bounce");
        std::printf("# d2d-auto / pin-bounce / map-bounce copy the pattern byte for byte\n");
    }

    // ---- with peer access DISABLED: the driver's own D2D staging ("cudaMemcpy through host") ----
    for (const Shape& sh : shapes) {
        const size_t B = (size_t) sh.bytes;
        const int iters = B >= (size_t) 16 << 20 ? 10 : 30;
        time_path("d2d-auto", sh, 3, iters, [&] {
            CHECK(cudaMemcpyAsync(d1, d0, B, cudaMemcpyDeviceToDevice, s0));
            CHECK(cudaStreamSynchronize(s0));
        }, rows);
    }

    // ---- enable peer access both ways; everything below may use the link ----
    CHECK(cudaSetDevice(D0));
    const cudaError_t pe01 = cudaDeviceEnablePeerAccess(D1, 0);
    CHECK(cudaSetDevice(D1));
    const cudaError_t pe10 = cudaDeviceEnablePeerAccess(D0, 0);
    CHECK(cudaSetDevice(D0));
    std::printf("# cudaDeviceEnablePeerAccess: 0->1 %s, 1->0 %s\n", cudaGetErrorString(pe01), cudaGetErrorString(pe10));
    if (pe01 != cudaSuccess || pe10 != cudaSuccess) {
        std::fprintf(stderr, "peer access failed; the P2P rows below are not meaningful\n");
        return 1;
    }

    // ---- cudaMemcpyPeerAsync under stream capture: the verify hand-off lives inside a captured graph ----
    {
        cudaStream_t sc;
        CHECK(cudaStreamCreate(&sc));
        bool ok = cudaStreamBeginCapture(sc, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
        if (ok) ok = cudaMemcpyPeerAsync(d1, D1, d0, D0, 4096, sc) == cudaSuccess;
        cudaGraph_t g = nullptr;
        const cudaError_t ce = cudaStreamEndCapture(sc, &g);
        ok = ok && ce == cudaSuccess && g != nullptr;
        if (ok) {   // and it has to survive instantiate + launch, not just capture
            cudaGraphExec_t ge = nullptr;
            ok = cudaGraphInstantiate(&ge, g, 0) == cudaSuccess && cudaGraphLaunch(ge, sc) == cudaSuccess &&
                 cudaStreamSynchronize(sc) == cudaSuccess;
            if (ge) cudaGraphExecDestroy(ge);
        }
        if (g) cudaGraphDestroy(g);
        cudaStreamDestroy(sc);
        cudaGetLastError();   // clear whatever the trial left behind
        std::printf("# cudaMemcpyPeerAsync inside a captured graph (capture+instantiate+launch): %s\n",
                    ok ? "works" : "REJECTED");
    }

    // ---- correctness of the two P2P paths (peer access is on now) ----
    {
        const size_t B = Smax;
        CHECK(cudaMemcpyPeerAsync(d1, D1, d0, D0, B, s0));
        CHECK(cudaStreamSynchronize(s0));
        same("p2p-memcpy");
        launch_copy(d1, d0, B, s0, false);
        CHECK(cudaStreamSynchronize(s0));
        same("p2p-kernel");
        std::printf("# p2p-memcpy / p2p-kernel copy the pattern byte for byte\n");
    }

    for (const Shape& sh : shapes) {
        const size_t B = (size_t) sh.bytes;
        const int iters = B >= (size_t) 16 << 20 ? 10 : 30;

        time_path("p2p-memcpy", sh, 3, iters, [&] {
            CHECK(cudaMemcpyPeerAsync(d1, D1, d0, D0, B, s0));
            CHECK(cudaStreamSynchronize(s0));
        }, rows);

        time_path("p2p-kernel", sh, 3, iters, [&] {   // the source card's kernel stores into the peer's memory
            launch_copy(d1, d0, B, s0, false);
            CHECK(cudaStreamSynchronize(s0));
        }, rows);

        time_path("pin-bounce", sh, 3, iters, [&] {   // prefill today: D2H + sync, then H2D + sync
            CHECK(cudaMemcpyAsync(h_pin, d0, B, cudaMemcpyDeviceToHost, s0));
            CHECK(cudaStreamSynchronize(s0));
            CHECK(cudaMemcpyAsync(d1, h_pin, B, cudaMemcpyHostToDevice, s1));
            CHECK(cudaStreamSynchronize(s1));
        }, rows);

        time_path("map-bounce", sh, 3, iters, [&] {   // verify today: kernel -> mapped host, sync, kernel -> device
            launch_copy(h_map_dev, d0, B, s0, true);
            CHECK(cudaStreamSynchronize(s0));
            launch_copy(d1, h_map_dev, B, s1, true);
            CHECK(cudaStreamSynchronize(s1));
        }, rows);
    }

    // one row per shape, one column per path (rows[] is path-major: every shape's d2d-auto first, then the
    // per-shape groups of p2p-memcpy / p2p-kernel / pin-bounce / map-bounce)
    std::printf("\n%-26s %10s %10s %10s %10s %10s\n", "shape (one hand-off)", "d2d-auto", "p2p-memcpy", "p2p-kernel",
                "pin-bounce", "map-bounce");
    std::printf("%s\n", std::string(80, '-').c_str());
    for (size_t si = 0; si < shapes.size(); ++si) {
        const Row& d2d = rows[si];
        const Row& pm = rows[shapes.size() + 4 * si + 0];
        const Row& pk = rows[shapes.size() + 4 * si + 1];
        const Row& pb = rows[shapes.size() + 4 * si + 2];
        const Row& mb = rows[shapes.size() + 4 * si + 3];
        std::printf("%-26s %10.4f %10.4f %10.4f %10.4f %10.4f\n", shapes[si].name, d2d.ms, pm.ms, pk.ms, pb.ms, mb.ms);
    }
    std::printf("\n%-26s %10s %10s %10s %10s %10s\n", "useful GB/s", "d2d-auto", "p2p-memcpy", "p2p-kernel",
                "pin-bounce", "map-bounce");
    for (size_t si = 0; si < shapes.size(); ++si) {
        const Row& d2d = rows[si];
        const Row& pm = rows[shapes.size() + 4 * si + 0];
        const Row& pk = rows[shapes.size() + 4 * si + 1];
        const Row& pb = rows[shapes.size() + 4 * si + 2];
        const Row& mb = rows[shapes.size() + 4 * si + 3];
        std::printf("%-26s %10.2f %10.2f %10.2f %10.2f %10.2f\n", shapes[si].name, d2d.gbs, pm.gbs, pk.gbs, pb.gbs, mb.gbs);
    }
    return 0;
}
