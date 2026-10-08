// dsv4/gpu_cuda.cu - the REAL CUDA device layer: the same dsv4::gpu API as src/gpu.cpp, compiled INSTEAD
// of it when the build asks for CUDA (build.sh --cuda, or cmake -DDSV4_WITH_CUDA=ON).
//
// What lives here:
//   * device memory, H2D/D2H, and the driver's real free/total VRAM;
//   * one dequantise-and-dot kernel per ggml type. The decode math is NOT duplicated: it is the shared
//     traits in include/dsv4/dq_traits.hpp, which tests/test_dq_traits.cpp checks element by element
//     against dequant_row()/row_dot() from src/dequant.cpp on the host. Same source, two compilations.
//   * the MoE HIT path: gate/up matvecs, a fused clamped-SwiGLU + routing-weight kernel, then the down
//     matvecs summed into d_y.
//
// Only the ggml types this model actually stores get a kernel (see supported()): Q8_0, Q4_K, Q5_K, Q6_K,
// BF16, F32, IQ3_XXS, IQ2_XXS, IQ1_M and MXFP4 - which is every tensor of DeepSeek-V4-Flash per shapes.txt.
// type_supported() reports anything else as unsupported so the loader can leave those tensors, and those MoE
// layers, on the host instead of uploading bytes no kernel could read. Adding a type = one trait in the
// header plus one case in matvec().
//
// Precision, deliberately: the host reference accumulates a dot product in double and sums left to right;
// the kernel accumulates in float and reduces as a tree. Results agree to float rounding, NOT bit for bit.
// The CPU-emulated device (src/gpu.cpp) is what the golden tests in tests/ are compared against.
#include "dsv4/gpu.hpp"

#include "dsv4/dequant.hpp"
#include "dsv4/iq_tables.hpp"

#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>

namespace dsv4 {
namespace gpu {
namespace {

// ---------------------------------------------------------------- error handling
// A CUDA error is sticky for the rest of the context, so the first one is the interesting one: printed once,
// remembered, and every later call reports failure instead of spamming the log or dying mid-token.
bool g_cuda_err = false;

bool check(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    if (!g_cuda_err) std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
    g_cuda_err = true;
    return false;
}

// ---------------------------------------------------------------- device state
size_t g_total = 0, g_used = 0;
bool g_ready = false;
float* g_down_acc = nullptr;   // kMaxHit * kMaxDim floats: the down projection of each HIT expert

// Staging for MISS experts: a device pool the host blocks are copied into, and a pinned host buffer
// they are copied THROUGH. Pinned matters - an unpinned cudaMemcpyAsync degrades to a staged,
// synchronised copy, and the whole point here is that the transfer queues ahead of the kernels
// instead of stalling the host.
uint8_t* g_stage = nullptr;
size_t g_stage_bytes = 0;
uint8_t* g_stage_pin = nullptr;

// Batched-prefill scratch: the token gather list and the routing weights of one expert are HOST
// arrays, so every batched launch copies them up first. One pair of buffers, reused expert by
// expert: the stream runs the launches in order, so the copy of expert k+1 cannot race expert k.
int32_t* g_d_sel = nullptr;
float* g_d_w = nullptr;
size_t g_batch_max = 0;

constexpr int64_t kMaxDim = 16384;
constexpr int kThreads = 256;

// ---------------------------------------------------------------- codebooks on device
// __constant__ because a warp reads the same grid entry with the same index nearly always, which is exactly
// what the constant cache broadcasts. The G_* macros below point the shared traits at these copies.
__constant__ uint64_t c_iq1s_grid[2048];
__constant__ uint64_t c_iq2xxs_grid[256];
__constant__ uint32_t c_iq3xxs_grid[256];
__constant__ uint8_t c_ksigns_iq2xs[128];
__constant__ uint8_t c_kmask_iq2xs[8];

}  // namespace
}  // namespace gpu
}  // namespace dsv4

// The shared decode traits: with these macros they read device constant memory here, and the plain
// dsv4::iq:: tables when the same header is compiled for the host by tests/test_dq_traits.cpp.
// Device-only on purpose: nothing in this file calls at() from the host, and if the traits were also
// __host__ nvcc compiles a host copy that reads __constant__ memory, which it warns about (#20091-D).
#define DSV4_HOST_DEVICE __device__ __forceinline__
#define G_iq1s_grid dsv4::gpu::c_iq1s_grid
#define G_iq2xxs_grid dsv4::gpu::c_iq2xxs_grid
#define G_iq3xxs_grid dsv4::gpu::c_iq3xxs_grid
#define G_ksigns_iq2xs dsv4::gpu::c_ksigns_iq2xs
#define G_kmask_iq2xs dsv4::gpu::c_kmask_iq2xs
#include "dsv4/dq_traits.hpp"

namespace dsv4 {
namespace gpu {
namespace {

// ---------------------------------------------------------------- matvec kernel
// One block per output row, kThreads threads: each thread walks a strided slice of the row, decoding the
// weights on the fly (the weights are the traffic here, not the activations), then a tree reduction writes
// y[r]. rows is gridDim.x, so the 129280-row output projection is a single launch.
template <class Tr>
__global__ void matvec_kernel(const uint8_t* __restrict__ W, int64_t rows, int in, int rb,
                              const float* __restrict__ x, float* __restrict__ y) {
    const int64_t r = blockIdx.x;
    if (r >= rows) return;
    // The byte stride comes from row_bytes() on the host, the single source of truth shared with
    // dequant.cpp. Checking it against the trait's block size means a trait that disagrees with the type it
    // claims to decode cannot read past the row.
    if (in <= 0 || in % Tr::BE != 0 || (int64_t)(in / Tr::BE) * (int64_t) Tr::BB != (int64_t) rb) {
        if (threadIdx.x == 0) y[r] = 0.f;
        return;
    }
    const uint8_t* w = W + (size_t) r * (size_t) rb;
    float acc = 0.f;
    for (int e = threadIdx.x; e < in; e += kThreads) acc += Tr::at(w, e) * x[e];
    __shared__ float red[kThreads];
    red[threadIdx.x] = acc;
    __syncthreads();
    #pragma unroll
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[r] = red[0];
}

// ---------------------------------------------------------------- batched matmul kernel (prefill)
// One block per (weight row, token) pair: same decode-on-the-fly math as matvec_kernel, but the T
// tokens of a chunk are independent, so the launch is rows*T blocks and the GPU stops being latency
// bound on a single 4096-element dot. `sel` is copied into device memory by matmul() (it is a host
// array), and sel == nullptr means the identity gather.
template <class Tr>
__global__ void matmul_kernel(const uint8_t* __restrict__ W, int64_t rows, int in, int rb,
                              const float* __restrict__ X, int T, const int32_t* __restrict__ sel,
                              float* __restrict__ Y, int64_t x_stride, int64_t y_stride) {
    const int64_t r = blockIdx.x;
    const int t = (int) blockIdx.y;
    if (r >= rows || t >= T) return;
    if (in <= 0 || in % Tr::BE != 0 || (int64_t)(in / Tr::BE) * (int64_t) Tr::BB != (int64_t) rb) {
        if (threadIdx.x == 0) Y[(int64_t) t * y_stride + r] = 0.f;
        return;
    }
    const int32_t src = sel ? sel[t] : t;
    const uint8_t* w = W + (size_t) r * (size_t) rb;
    const float* x = X + (int64_t) src * x_stride;
    float acc = 0.f;
    for (int e = threadIdx.x; e < in; e += kThreads) acc += Tr::at(w, e) * x[e];
    __shared__ float red[kThreads];
    red[threadIdx.x] = acc;
    __syncthreads();
    #pragma unroll
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) Y[(int64_t) t * y_stride + r] = acc;
}

// ---------------------------------------------------------------- MoE kernels
// a = w_k * swiglu_clamped(g, u, limit). The routing weight is folded in here so the down projection needs
// no scaling pass of its own. w_k is passed BY VALUE: p.w is a host array and must never be read here.
__global__ void swiglu_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ a,
                              int64_t n, float limit, float w_k) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xa = g[i], ub = u[i];
    if (limit > 0.f) { xa = fminf(xa, limit); ub = fminf(fmaxf(ub, -limit), limit); }
    a[i] = w_k * (xa / (1.f + expf(-xa)) * ub);
}

// Batched MoE: one launch per expert covers ALL of that expert's tokens of the chunk. w is a DEVICE
// array (n floats) copied up by experts_batch(), because p.w is host memory.
__global__ void swiglu_batch_kernel(const float* __restrict__ g, const float* __restrict__ u,
                                    float* __restrict__ a, int64_t n, int64_t ff, float limit,
                                    const float* __restrict__ w) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * ff) return;
    const int64_t t = i / ff;
    const float* gp = g + i;
    const float* up = u + i;
    float xa = *gp, ub = *up;
    if (limit > 0.f) { xa = fminf(xa, limit); ub = fminf(fmaxf(ub, -limit), limit); }
    a[i] = w[t] * (xa / (1.f + expf(-xa)) * ub);
}

// d_Y[sel[t]] += d_dn[t] for every t: the down projections of one expert, scattered back to their
// token rows. sel is the same device array matmul_kernel reads.
//
// No atomics, deliberately. Within ONE expert a token appears at most once (a token routes to an
// expert once), so this launch has no write conflicts at all; the experts are separate launches on
// one stream, which run strictly in order. The sum therefore happens in expert order, exactly like
// the CPU-emulated backend, instead of the arbitrary order atomicAdd would give.
__global__ void scatter_add_kernel(const float* __restrict__ dn, const int32_t* __restrict__ sel,
                                   int n, int64_t dim, float* __restrict__ Y) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) n * dim) return;
    const int64_t t = i / dim;
    const int64_t c = i - t * dim;
    Y[(int64_t) sel[t] * dim + c] += dn[i];
}

// d_y = sum_k tmp[k] in a fixed order: deterministic, and cheap (dim floats).
__global__ void sum_rows_kernel(const float* __restrict__ tmp, int n, int64_t dim, float* __restrict__ y) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    float s = 0.f;
    for (int k = 0; k < n; ++k) s += tmp[(size_t) k * (size_t) dim + i];
    y[i] = s;
}

// ---------------------------------------------------------------- dispatch
template <class Tr>
bool launch_mv(uint32_t type, const uint8_t* W, int64_t rows, int64_t in, const float* d_x, float* d_y) {
    const size_t rb = row_bytes(type, in);
    if (!rb || rows <= 0 || in <= 0 || in > (1 << 30)) return false;
    // Checked HERE, on the host, before anything is launched: row_bytes() is the single source of truth
    // shared with dequant.cpp, and if a trait's block size disagrees with the type it claims to decode,
    // the honest answer is false ("wrote nothing, use the host") - not a row of zeros on the device.
    if (in % Tr::BE != 0 || (int64_t)(in / Tr::BE) * (int64_t) Tr::BB != (int64_t) rb) return false;
    matvec_kernel<Tr><<<(unsigned) rows, kThreads>>>(W, rows, (int) in, (int) rb, d_x, d_y);
    return check(cudaGetLastError(), "matvec launch");
}

template <class Tr>
bool launch_mm(uint32_t type, const uint8_t* W, int64_t rows, int64_t in, const float* d_X, int T,
               const int32_t* d_sel, float* d_Y, int64_t x_stride, int64_t y_stride) {
    const size_t rb = row_bytes(type, in);
    if (!rb || rows <= 0 || in <= 0 || in > (1 << 30) || T <= 0) return false;
    if (x_stride <= 0) x_stride = in;
    if (y_stride <= 0) y_stride = rows;
    if (in % Tr::BE != 0 || (int64_t)(in / Tr::BE) * (int64_t) Tr::BB != (int64_t) rb) return false;
    // gridDim.y is capped at 65535, and a chunk can be bigger than that only for absurd contexts;
    // splitting on the token axis keeps the result identical (each (row, token) pair is one block).
    for (int t0 = 0; t0 < T; t0 += 65535) {
        const int tn = T - t0 < 65535 ? T - t0 : 65535;
        dim3 grid((unsigned) rows, (unsigned) tn);
        matmul_kernel<Tr><<<grid, kThreads>>>(W, rows, (int) in, (int) rb,
                                              d_X + (int64_t) t0 * x_stride, tn,
                                              d_sel ? d_sel + t0 : nullptr,
                                              d_Y + (int64_t) t0 * y_stride, x_stride, y_stride);
        if (!check(cudaGetLastError(), "matmul launch")) return false;
    }
    return true;
}

/// Every ggml type this file has a kernel for. One table, so type_supported(), experts_supported() and
/// matvec() can never disagree about what is safe to put on the device.
bool supported(uint32_t type) {
    switch (type) {
        case T_F32: case T_BF16: case T_Q8_0: case T_Q4_K: case T_Q5_K: case T_Q6_K:
        case T_IQ3_XXS: case T_IQ2_XXS: case T_IQ1_M: case T_MXFP4:
            return true;
        default: return false;
    }
}

}  // namespace

// ================================================================================================
bool is_emulated() { return false; }

bool init(std::string& err) {
    if (g_ready) return true;
    int ndev = 0;
    if (!check(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount")) { err = "no CUDA device"; return false; }
    if (ndev == 0) { err = "no CUDA device visible"; return false; }
    if (!check(cudaSetDevice(0), "cudaSetDevice")) { err = "cannot select CUDA device 0"; return false; }
    size_t free_b = 0, total_b = 0;
    if (!check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo")) { err = "cannot read VRAM size"; return false; }
    g_total = total_b; g_used = 0;

    // Codebooks: the same bytes src/iq_tables.cpp gives the host decoder.
    bool ok = true;
    ok &= check(cudaMemcpyToSymbol(c_iq1s_grid, iq::iq1s_grid, sizeof(iq::iq1s_grid)), "copy iq1s_grid");
    ok &= check(cudaMemcpyToSymbol(c_iq2xxs_grid, iq::iq2xxs_grid, sizeof(iq::iq2xxs_grid)), "copy iq2xxs_grid");
    ok &= check(cudaMemcpyToSymbol(c_iq3xxs_grid, iq::iq3xxs_grid, sizeof(iq::iq3xxs_grid)), "copy iq3xxs_grid");
    ok &= check(cudaMemcpyToSymbol(c_ksigns_iq2xs, iq::ksigns_iq2xs, sizeof(iq::ksigns_iq2xs)), "copy ksigns_iq2xs");
    ok &= check(cudaMemcpyToSymbol(c_kmask_iq2xs, iq::kmask_iq2xs, sizeof(iq::kmask_iq2xs)), "copy kmask_iq2xs");
    if (!ok) { err = "cannot upload the IQ codebooks"; return false; }

    if (!check(cudaMalloc((void**) &g_down_acc, (size_t) kMaxHit * kMaxDim * sizeof(float)), "alloc down scratch")) {
        err = "cannot allocate the MoE down-projection scratch"; g_down_acc = nullptr; return false;
    }
    // Batched-prefill scratch: kMaxSel token indices + kMaxSel routing weights = 48 KiB. Small enough
    // to always have, and without it matmul()/experts_batch() answer false rather than guess.
    if (!check(cudaMalloc((void**) &g_d_sel, (size_t) kMaxSel * sizeof(int32_t)), "alloc batch sel")) {
        err = "cannot allocate the batched-prefill scratch"; g_d_sel = nullptr; g_ready = false; return false;
    }
    if (!check(cudaMalloc((void**) &g_d_w, (size_t) kMaxSel * sizeof(float)), "alloc batch weights")) {
        err = "cannot allocate the batched-prefill scratch"; g_d_w = nullptr; g_ready = false; return false;
    }
    g_batch_max = kMaxSel;
    g_ready = true;
    return true;
}

void* alloc(size_t n) {
    if (!g_ready || g_cuda_err || g_used + n > g_total) return nullptr;
    void* p = nullptr;
    if (!check(cudaMalloc(&p, n), "cudaMalloc")) return nullptr;
    g_used += n;
    return p;
}

void release(void* p) { if (p) cudaFree(p); }  // g_used is never decreased: the pool is freed with the model

void mem_info(size_t* free_bytes, size_t* total_bytes) {
    size_t f = 0, t = 0;
    if (g_ready && check(cudaMemGetInfo(&f, &t), "cudaMemGetInfo")) {
        if (free_bytes) *free_bytes = f;
        if (total_bytes) *total_bytes = t;
    } else {
        if (free_bytes) *free_bytes = g_ready && g_total > g_used ? g_total - g_used : 0;
        if (total_bytes) *total_bytes = g_ready ? g_total : 0;
    }
}

bool type_supported(uint32_t type) { return g_ready && !g_cuda_err && supported(type); }
bool experts_supported(uint32_t g, uint32_t u, uint32_t d) {
    return g_ready && !g_cuda_err && supported(g) && supported(u) && supported(d);
}

bool staging(uint8_t* device_pool, size_t bytes) {
    if (!g_ready || g_cuda_err || !device_pool || bytes == 0) return false;
    void* pin = nullptr;
    if (!check(cudaHostAlloc(&pin, bytes, cudaHostAllocDefault), "pinned staging buffer")) return false;
    g_stage = device_pool;
    g_stage_bytes = bytes;
    g_stage_pin = (uint8_t*) pin;
    return true;
}

bool has_staging() { return g_stage != nullptr && !g_cuda_err; }

void h2d(void* dst, const void* src, size_t bytes) {
    if (!bytes || g_cuda_err) return;
    check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "H2D copy");
}
void d2h(void* dst, const void* src, size_t bytes) {
    if (!bytes || g_cuda_err) return;
    check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "D2H copy");  // also the sync point of a launch
}

bool matvec(uint32_t type, const uint8_t* W, int64_t rows, int64_t in, const float* d_x, float* d_y) {
    if (!g_ready || g_cuda_err || !supported(type)) return false;
    switch (type) {
        case T_F32:     return launch_mv<dqt::F32T>(type, W, rows, in, d_x, d_y);
        case T_BF16:    return launch_mv<dqt::BF16T>(type, W, rows, in, d_x, d_y);
        case T_Q8_0:    return launch_mv<dqt::Q8_0T>(type, W, rows, in, d_x, d_y);
        case T_Q4_K:    return launch_mv<dqt::Q4_KT>(type, W, rows, in, d_x, d_y);
        case T_Q5_K:    return launch_mv<dqt::Q5_KT>(type, W, rows, in, d_x, d_y);
        case T_Q6_K:    return launch_mv<dqt::Q6_KT>(type, W, rows, in, d_x, d_y);
        case T_IQ3_XXS: return launch_mv<dqt::IQ3_XXST>(type, W, rows, in, d_x, d_y);
        case T_IQ2_XXS: return launch_mv<dqt::IQ2_XXST>(type, W, rows, in, d_x, d_y);
        case T_IQ1_M:   return launch_mv<dqt::IQ1_MT>(type, W, rows, in, d_x, d_y);
        case T_MXFP4:   return launch_mv<dqt::MXFP4T>(type, W, rows, in, d_x, d_y);
        default: return false;
    }
}

// The same switch, for the token-batched launch. One table here too, so a type can never be
// matvec-able but not matmul-able.
bool mm_dispatch(uint32_t type, const uint8_t* W, int64_t rows, int64_t in, const float* d_X, int T,
                 const int32_t* d_sel, float* d_Y, int64_t x_stride, int64_t y_stride) {
    switch (type) {
        case T_F32:     return launch_mm<dqt::F32T>(type, W, rows, in, d_X, T, d_sel, d_Y, x_stride, y_stride);
        case T_BF16:    return launch_mm<dqt::BF16T>(type, W, rows, in, d_X, T, d_sel, d_Y, x_stride, y_stride);
        case T_Q8_0:    return launch_mm<dqt::Q8_0T>(type, W, rows, in, d_X, T, d_sel, d_Y, x_stride, y_stride);
        case T_Q4_K:    return launch_mm<dqt::Q4_KT>(type, W, rows, in, d_X, T, d_sel, d_Y, x_stride, y_stride);
        case T_Q5_K:    return launch_mm<dqt::Q5_KT>(type, W, rows, in, d_X, T, d_sel, d_Y, x_stride, y_stride);
        case T_Q6_K:    return launch_mm<dqt::Q6_KT>(type, W, rows, in, d_X, T, d_sel, d_Y, x_stride, y_stride);
        case T_IQ3_XXS: return launch_mm<dqt::IQ3_XXST>(type, W, rows, in, d_X, T, d_sel, d_Y, x_stride, y_stride);
        case T_IQ2_XXS: return launch_mm<dqt::IQ2_XXST>(type, W, rows, in, d_X, T, d_sel, d_Y, x_stride, y_stride);
        case T_IQ1_M:   return launch_mm<dqt::IQ1_MT>(type, W, rows, in, d_X, T, d_sel, d_Y, x_stride, y_stride);
        case T_MXFP4:   return launch_mm<dqt::MXFP4T>(type, W, rows, in, d_X, T, d_sel, d_Y, x_stride, y_stride);
        default: return false;
    }
}

bool matmul(uint32_t type, const uint8_t* W, int64_t rows, int64_t in,
            const float* d_X, int T, const int32_t* sel, float* d_Y, int64_t x_stride, int64_t y_stride) {
    if (!g_ready || g_cuda_err || !supported(type)) return false;
    if (rows <= 0 || in <= 0 || T <= 0 || T > kMaxSel) return false;
    if (x_stride <= 0) x_stride = in;
    if (y_stride <= 0) y_stride = rows;
    const int32_t* d_sel = nullptr;
    if (sel) {
        if (!g_d_sel) return false;
        if (!check(cudaMemcpy(g_d_sel, sel, (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice), "matmul sel H2D"))
            return false;
        d_sel = g_d_sel;
    }
    return mm_dispatch(type, W, rows, in, d_X, T, d_sel, d_Y, x_stride, y_stride);
}

// A host expert has to be copied up before the kernels can read it. The DEVICE side of that is free of
// races: everything here runs on one stream, so the copy of expert k+1 is executed after expert k's
// kernels have consumed their bytes. The HOST side is not: writing the pinned buffer again while the
// DMA of the previous copy is still reading it would corrupt that transfer. Hence a small ring of
// pinned slots, with a stream sync only when the ring wraps.
constexpr int kStageRing = 4;
uint8_t* g_batch_pin = nullptr;
size_t g_batch_pin_slot = 0;   // bytes per slot
int g_batch_pin_next = 0;

bool ensure_batch_pin(size_t slot_bytes) {
    if (g_batch_pin && g_batch_pin_slot >= slot_bytes) return true;
    // The ring is grown, never shrunk: one allocation covers every layer of the model, and a free
    // while a transfer is still in flight would corrupt it.
    if (g_batch_pin) {
        check(cudaStreamSynchronize(0), "batch ring sync before free");
        cudaFreeHost(g_batch_pin);
    }
    g_batch_pin = nullptr;
    g_batch_pin_slot = 0;
    void* p = nullptr;
    if (!check(cudaHostAlloc(&p, slot_bytes * kStageRing, cudaHostAllocDefault), "batch pinned ring")) return false;
    g_batch_pin = (uint8_t*) p;
    g_batch_pin_slot = slot_bytes;
    g_batch_pin_next = 0;
    return true;
}

bool experts_batch(const ExpBatch* e, int n_exp, uint32_t type_g, uint32_t type_u, uint32_t type_d,
                   int64_t ff, int64_t dim, float swiglu_limit,
                   const float* d_X, int T, float* d_g, float* d_u, float* d_a, float* d_dn, float* d_Y) {
    if (n_exp <= 0) return true;
    if (!g_ready || g_cuda_err) return false;
    if (!supported(type_g) || !supported(type_u) || !supported(type_d)) return false;
    if (!g_d_sel || !g_d_w) return false;
    const size_t bg = row_bytes(type_g, dim) * (size_t) ff;
    const size_t bu = row_bytes(type_u, dim) * (size_t) ff;
    const size_t bd = row_bytes(type_d, ff) * (size_t) dim;
    if (!bg || !bu || !bd) return false;

    // d_Y is ACCUMULATED into (scatter_add does +=), so it has to start at zero. The caller only
    // ever reads back the T rows it asked for, so exactly those are cleared.
    if (T <= 0) return true;
    if (!check(cudaMemsetAsync(d_Y, 0, (size_t) T * dim * sizeof(float)), "batch d_Y memset")) return false;

    for (int ei = 0; ei < n_exp; ++ei) {
        const ExpBatch& b = e[ei];
        const int n = b.n;
        if (n <= 0) continue;
        if (n > kMaxSel || !b.tok || !b.w) return false;
        uint8_t* gp = b.gate;
        uint8_t* gu = b.up;
        uint8_t* gd = b.down;
        if (!gp) {   // host expert: stage it into the device pool, like experts_hit() does
            if (!b.hg || !b.hu || !b.hd || !g_stage) return false;
            if (g_stage_bytes < bg + bu + bd) return false;
            if (!ensure_batch_pin(bg + bu + bd)) return false;
            if (g_batch_pin_next == 0 && !check(cudaStreamSynchronize(0), "batch stage ring sync")) return false;
            uint8_t* pin = g_batch_pin + (size_t) g_batch_pin_next * g_batch_pin_slot;
            g_batch_pin_next = (g_batch_pin_next + 1) % kStageRing;
            std::memcpy(pin, b.hg, bg);
            std::memcpy(pin + bg, b.hu, bu);
            std::memcpy(pin + bg + bu, b.hd, bd);
            if (!check(cudaMemcpyAsync(g_stage, pin, bg + bu + bd, cudaMemcpyHostToDevice), "batch stage H2D"))
                return false;
            gp = g_stage; gu = g_stage + bg; gd = g_stage + bg + bu;
        }
        if (!check(cudaMemcpyAsync(g_d_sel, b.tok, (size_t) n * sizeof(int32_t), cudaMemcpyHostToDevice), "batch tok H2D"))
            return false;
        if (!check(cudaMemcpyAsync(g_d_w, b.w, (size_t) n * sizeof(float), cudaMemcpyHostToDevice), "batch w H2D"))
            return false;

        // gate/up read the chunk through the gather; the down projection reads d_a, which this same
        // launch order has already filled for exactly those n rows.
        if (!mm_dispatch(type_g, gp, ff, dim, d_X, n, g_d_sel, d_g, dim, ff)) return false;
        if (!mm_dispatch(type_u, gu, ff, dim, d_X, n, g_d_sel, d_u, dim, ff)) return false;
        swiglu_batch_kernel<<<(unsigned) (((int64_t) n * ff + 255) / 256), 256>>>(d_g, d_u, d_a, n, ff, swiglu_limit, g_d_w);
        if (!check(cudaGetLastError(), "swiglu batch launch")) return false;
        if (!mm_dispatch(type_d, gd, dim, ff, d_a, n, nullptr, d_dn, ff, dim)) return false;
        scatter_add_kernel<<<(unsigned) (((int64_t) n * dim + 255) / 256), 256>>>(d_dn, g_d_sel, n, dim, d_Y);
        if (!check(cudaGetLastError(), "scatter_add launch")) return false;
    }
    return true;
}

bool experts_hit(const ExpPtrs& p, uint32_t type_g, uint32_t type_u, uint32_t type_d, int64_t ff, int64_t dim,
                 float swiglu_limit, const float* d_x, float* d_g, float* d_u, float* d_a, float* d_y) {
    if (p.n <= 0) return true;
    if (!g_ready || g_cuda_err) return false;
    if (!supported(type_g) || !supported(type_u) || !supported(type_d)) return false;
    if (dim > kMaxDim) return false;   // the down scratch was sized for kMaxDim at init

    const int nk = p.n < kMaxHit ? p.n : kMaxHit;

    // Resolve every expert to a DEVICE pointer first. A host expert is staged into the pool: the
    // copies are queued on the same stream as the kernels below, so the transfer of expert k+1
    // overlaps the matvecs of expert k.
    uint8_t* gp[kMaxHit];
    uint8_t* gu[kMaxHit];
    uint8_t* gd[kMaxHit];
    const size_t bg = row_bytes(type_g, dim) * (size_t) ff;
    const size_t bu = row_bytes(type_u, dim) * (size_t) ff;
    const size_t bd = row_bytes(type_d, ff) * (size_t) dim;
    if (!bg || !bu || !bd) return false;
    for (int k = 0; k < nk; ++k) {
        gp[k] = p.gate[k];
        gu[k] = p.up[k];
        gd[k] = p.down[k];
        if (gp[k]) continue;
        if (!p.hg[k] || !p.hu[k] || !p.hd[k]) return false;
        if (p.bpe < bg + bu + bd || (size_t) (k + 1) * p.bpe > g_stage_bytes) return false;
        uint8_t* dst = g_stage + (size_t) k * p.bpe;
        uint8_t* pin = g_stage_pin + (size_t) k * p.bpe;
        std::memcpy(pin, p.hg[k], bg);
        std::memcpy(pin + bg, p.hu[k], bu);
        std::memcpy(pin + bg + bu, p.hd[k], bd);
        if (!check(cudaMemcpyAsync(dst, pin, bg + bu + bd, cudaMemcpyHostToDevice), "stage H2D")) return false;
        gp[k] = dst;
        gu[k] = dst + bg;
        gd[k] = dst + bg + bu;
    }

    for (int k = 0; k < nk; ++k) {
        if (!matvec(type_g, gp[k], ff, dim, d_x, d_g + (size_t) k * ff)) return false;
        if (!matvec(type_u, gu[k], ff, dim, d_x, d_u + (size_t) k * ff)) return false;
        swiglu_kernel<<<(unsigned) ((ff + 255) / 256), 256>>>(d_g + (size_t) k * ff, d_u + (size_t) k * ff,
                                                              d_a + (size_t) k * ff, ff, swiglu_limit, p.w[k]);
        if (!check(cudaGetLastError(), "swiglu launch")) return false;
    }
    for (int k = 0; k < nk; ++k)
        if (!matvec(type_d, gd[k], dim, ff, d_a + (size_t) k * ff, g_down_acc + (size_t) k * dim)) return false;

    sum_rows_kernel<<<(unsigned) ((dim + 255) / 256), 256>>>(g_down_acc, nk, dim, d_y);
    return check(cudaGetLastError(), "sum_rows launch");
}

}  // namespace gpu
}  // namespace dsv4
