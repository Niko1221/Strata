// src/kernels/cuda/lora.cu - see include/strata/kernels/lora.hpp.
//
// Two paths for y += B (A x), both reading the request flag on the device:
//   * small (n <= SMALL_N tokens per launch, the decode and verify windows): h = A x, then y += B h, two kernels
//     through the layer's own h.  The second is a programmatic dependent launch: it starts while the first runs and
//     loads its column of B before it waits for h.
//   * tiled (the prompt path, eager): h = X A^T and y += h B^T as two tiled fp32 GEMMs through a per-device scratch.
#include "strata/kernels/lora.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int SMALL_N = 8;          // tokens per launch pair of the small path
constexpr int DOWN_THREADS = 256;   // A x: one block per row of A
constexpr int DOWN_WARPS = DOWN_THREADS / 32;
constexpr int UP_THREADS = 128;     // B h: one output per thread
constexpr int TILE = 64, TK = 16, TILE_THREADS = 256;
constexpr int kMaxSplits = 32, kSplitBlocks = 512;   // the tiled A x: K split until about this many blocks

Lora g_lora;
bool g_on_host = false;
constexpr int kDevices = 64;
// a: rank x n_in, bt: rank x n_out (B transposed), h: the small path's A x (SMALL_N x rank).  h is per layer: two
// streams of one device never run the same layer at once, but a layer split with both stages on one card reads a
// prompt on two streams at once, on different layers.
struct LayerDev { float* a = nullptr; float* bt = nullptr; float* h = nullptr; };
struct Scratch { float* h = nullptr; int64_t tokens = 0; };   // the tiled path's A x, T x rank_max
struct DevTables {
    std::vector<LayerDev> layers;
    int* on = nullptr;
    std::map<cudaStream_t, Scratch> scratch;   // per stream (see LayerDev); g_scratch_mu
    bool ready() const { return on != nullptr; }
};
DevTables g_dev[kDevices];
struct LayerHost { int64_t rank = 0, n_in = 0, n_out = 0; std::vector<float> a, bt; };
std::vector<LayerHost> g_host;
std::mutex g_scratch_mu;

int cur_device() {
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= kDevices) d = 0;
    return d;
}
void free_tables(DevTables& t) {
    for (LayerDev& l : t.layers) { cudaFree(l.a); cudaFree(l.bt); cudaFree(l.h); }
    cudaFree(t.on);
    for (auto& [stream, sc] : t.scratch) cudaFree(sc.h);
    t = DevTables{};
}
bool upload_here(std::string& err) {
    DevTables& t = g_dev[cur_device()];
    if (t.ready()) return true;
    const int flag = g_on_host ? 1 : 0;
    t.layers.assign(g_host.size(), LayerDev{});
    bool ok = cudaMalloc(&t.on, sizeof(int)) == cudaSuccess &&
              cudaMemcpy(t.on, &flag, sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess;
    for (size_t l = 0; ok && l < g_host.size(); ++l) {
        const LayerHost& h = g_host[l];
        if (h.rank == 0) continue;
        LayerDev& d = t.layers[l];
        ok = cudaMalloc(&d.a, h.a.size() * sizeof(float)) == cudaSuccess &&
             cudaMalloc(&d.bt, h.bt.size() * sizeof(float)) == cudaSuccess &&
             cudaMalloc(&d.h, (size_t) SMALL_N * h.rank * sizeof(float)) == cudaSuccess &&
             cudaMemcpy(d.a, h.a.data(), h.a.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMemcpy(d.bt, h.bt.data(), h.bt.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    }
    if (!ok) {
        err = "lora: device allocation failed";
        free_tables(t);
        return false;
    }
    return true;
}

__device__ __forceinline__ float load_x(const float* x, int64_t i) { return __ldg(x + i); }
__device__ __forceinline__ float load_x(const uint16_t* x, int64_t i) { return f32_from_f16(__ldg(x + i)); }

// The small path (decode, verify, short eager calls), two kernels per SMALL_N tokens, deterministic (fixed
// summation order, no atomics) and capturable (no host sync, the per-layer h allocated at upload):
//   down: h[t, r] = sum_k a[r, k] x[t, k]    one block per row r of A (A read once, x from L2)
//   up:   y[t, o] += sum_r bt[r, o] h[t, r]  one thread per output o (bt coalesced across the warp)
template <typename XT, int NT>
__global__ void __launch_bounds__(DOWN_THREADS) lora_down_kernel(const float* __restrict__ a, const XT* __restrict__ x,
                                                                 int64_t ldx, float* __restrict__ h, int rank, int n_in,
                                                                 const int* __restrict__ on) {
    if (*on == 0) return;
    __shared__ float red[NT][DOWN_WARPS];
    const int r = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float* ar = a + (int64_t) r * n_in;
    float p[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) p[t] = 0.0f;
#pragma unroll 4
    for (int k = threadIdx.x; k < n_in; k += DOWN_THREADS) {
        const float av = __ldg(ar + k);
#pragma unroll
        for (int t = 0; t < NT; ++t) p[t] = fmaf(av, load_x(x, (int64_t) t * ldx + k), p[t]);
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        float v = p[t];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) red[t][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < NT) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < DOWN_WARPS; ++w) v += red[threadIdx.x][w];
        h[threadIdx.x * rank + r] = v;
    }
}

template <int NT>
__global__ void __launch_bounds__(UP_THREADS) lora_up_kernel(const float* __restrict__ bt, const float* __restrict__ h,
                                                             float* __restrict__ y, int64_t ldy, int rank, int n_out,
                                                             const int* __restrict__ on) {
    if (*on == 0) return;
    extern __shared__ float hs[];   // NT x rank
    for (int i = threadIdx.x; i < NT * rank; i += UP_THREADS) hs[i] = h[i];
    __syncthreads();
    const int o = blockIdx.x * UP_THREADS + threadIdx.x;
    if (o >= n_out) return;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
#pragma unroll 8
    for (int r = 0; r < rank; ++r) {
        const float bv = __ldg(bt + (int64_t) r * n_out + o);
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[t] = fmaf(bv, hs[t * rank + r], acc[t]);
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) y[(int64_t) t * ldy + o] += acc[t];
}

// The up kernel when B fits in registers (rank <= PDL_RANK) and the device has programmatic dependent launch: the
// same sums as lora_up_kernel, but the B column is loaded before cudaGridDependencySynchronize, while the down kernel
// still runs.  The flag is read first, so an off request loads nothing (the host only writes it between syncs).
#if defined(__HIPCC__)
constexpr bool kPdl = false;
#else
constexpr bool kPdl = true;
#endif
constexpr int PDL_RANK = 64;
template <int NT>
__global__ void __launch_bounds__(UP_THREADS) lora_up_pdl_kernel(const float* __restrict__ bt, const float* h,
                                                                 float* __restrict__ y, int64_t ldy, int rank, int n_out,
                                                                 const int* __restrict__ on) {
    if (*on == 0) return;
    extern __shared__ float hs[];   // NT x rank
    const int o = blockIdx.x * UP_THREADS + threadIdx.x;
    float bv[PDL_RANK];
#pragma unroll
    for (int r = 0; r < PDL_RANK; ++r) bv[r] = (r < rank && o < n_out) ? __ldg(bt + (int64_t) r * n_out + o) : 0.0f;
#if !defined(__HIPCC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();   // h written and visible; before sm_90 the launch is an ordinary one
#endif
    for (int i = threadIdx.x; i < NT * rank; i += UP_THREADS) hs[i] = __ldcg(h + i);
    __syncthreads();
    if (o >= n_out) return;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
#pragma unroll
    for (int r = 0; r < PDL_RANK; ++r) {
        if (r < rank) {
#pragma unroll
            for (int t = 0; t < NT; ++t) acc[t] = fmaf(bv[r], hs[t * rank + r], acc[t]);
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) y[(int64_t) t * ldy + o] += acc[t];
}

template <typename XT, int NT>
void launch_small(const float* a, const float* bt, float* h, const XT* x, int64_t ldx, float* y, int64_t ldy, int rank,
                  int n_in, int n_out, const int* on, cudaStream_t s) {
    lora_down_kernel<XT, NT><<<(unsigned) rank, DOWN_THREADS, 0, s>>>(a, x, ldx, h, rank, n_in, on);
    const unsigned up_blocks = (unsigned) ((n_out + UP_THREADS - 1) / UP_THREADS);
    const size_t up_smem = (size_t) NT * rank * sizeof(float);
#if !defined(__HIPCC__)
    if (kPdl && rank <= PDL_RANK) {
        cudaLaunchAttribute attr[1] = {};
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(up_blocks);
        cfg.blockDim = dim3(UP_THREADS);
        cfg.dynamicSmemBytes = up_smem;
        cfg.stream = s;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        cudaLaunchKernelEx(&cfg, lora_up_pdl_kernel<NT>, bt, (const float*) h, y, ldy, rank, n_out, on);
        return;
    }
#endif
    lora_up_kernel<NT><<<up_blocks, UP_THREADS, up_smem, s>>>(bt, h, y, ldy, rank, n_out, on);
}

// C[T, N] (=, or += with `add`) X[T, K] . W, W given as [N, K] (w_nk) or [K, N] row-major, all fp32 but X.
// 64 x 64 output tile per block, 4 x 4 per thread, K in slabs of 16.  Split K: block z sums k in [z kc, (z+1) kc)
// into C + z * split_stride (the partial sums, added up by lora_sum_kernel in a fixed order).
template <typename XT, bool W_NK>
__global__ void __launch_bounds__(TILE_THREADS) lora_tiled_kernel(const XT* __restrict__ X, int64_t ldx,
                                                                  const float* __restrict__ W, int64_t ldw,
                                                                  float* __restrict__ C, int64_t ldc, int T, int N, int K,
                                                                  int kc, int64_t split_stride, int add,
                                                                  const int* __restrict__ on) {
    if (*on == 0) return;
    const int kb = blockIdx.z * kc, ke = min(K, kb + kc);
    C += blockIdx.z * split_stride;
    __shared__ float xs[TK][TILE + 1];
    __shared__ float ws[TK][TILE + 1];
    const int t0 = blockIdx.y * TILE, n0 = blockIdx.x * TILE;
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;   // outputs (t0 + ty + 16 i, n0 + tx + 16 j)
    float acc[4][4] = {};
    for (int k0 = kb; k0 < ke; k0 += TK) {
        for (int i = threadIdx.x; i < TILE * TK; i += TILE_THREADS) {
            const int row = i / TK, kk = i % TK, k = k0 + kk;   // row-fastest over k: coalesced X and [N, K] W
            const int t = t0 + row, nn = n0 + row;
            xs[kk][row] = t < T && k < ke ? load_x(X, (int64_t) t * ldx + k) : 0.0f;
            if (W_NK) ws[kk][row] = nn < N && k < ke ? __ldg(W + (int64_t) nn * ldw + k) : 0.0f;
        }
        if (!W_NK)
            for (int i = threadIdx.x; i < TILE * TK; i += TILE_THREADS) {
                const int kk = i / TILE, col = i % TILE, k = k0 + kk, nn = n0 + col;   // coalesced [K, N] W
                ws[kk][col] = nn < N && k < ke ? __ldg(W + (int64_t) k * ldw + nn) : 0.0f;
            }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < TK; ++kk) {
            float xv[4], wv[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) { xv[i] = xs[kk][ty + 16 * i]; wv[i] = ws[kk][tx + 16 * i]; }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(xv[i], wv[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int t = t0 + ty + 16 * i;
        if (t >= T) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int nn = n0 + tx + 16 * j;
            if (nn >= N) continue;
            float* c = C + (int64_t) t * ldc + nn;
            *c = add ? *c + acc[i][j] : acc[i][j];
        }
    }
}

// out[i] = sum_z part[z * count + i], z in order
__global__ void lora_sum_kernel(const float* __restrict__ part, int splits, int64_t count, float* __restrict__ out,
                                const int* __restrict__ on) {
    if (*on == 0) return;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float v = 0.0f;
    for (int z = 0; z < splits; ++z) v += part[z * count + i];
    out[i] = v;
}

bool capturing(cudaStream_t s) {
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    return cudaStreamIsCapturing(s, &st) == cudaSuccess && st != cudaStreamCaptureStatusNone;
}

template <typename XT>
void apply(int64_t layer, const XT* x, int64_t ldx, int64_t n, float* y, int64_t ldy, void* stream) {
    if (!g_lora.covers(layer) || n < 1) return;
    DevTables& t = g_dev[cur_device()];
    if (!t.ready()) throw std::runtime_error("lora_apply: the adapter is not on this device (lora_replicate)");
    const LayerHost& h = g_host[(size_t) layer];
    const LayerDev& d = t.layers[(size_t) layer];
    const cudaStream_t s = (cudaStream_t) stream;
    if (ldx == 0) ldx = h.n_in;
    if (ldy == 0) ldy = h.n_out;
    const bool cap = capturing(s);
    if (cap && n > kLoraCaptureTokens) throw std::runtime_error("lora_apply: too many tokens for a captured call");
    if (cap || n <= 4 * SMALL_N) {
        for (int64_t t0 = 0; t0 < n; t0 += SMALL_N) {
            const int nb = (int) std::min<int64_t>(SMALL_N, n - t0);
            const XT* xt = x + t0 * ldx;
            float* yt = y + t0 * ldy;
            const int R = (int) h.rank, I = (int) h.n_in, O = (int) h.n_out;
            switch (nb) {
            case 1: launch_small<XT, 1>(d.a, d.bt, d.h, xt, ldx, yt, ldy, R, I, O, t.on, s); break;
            case 2: launch_small<XT, 2>(d.a, d.bt, d.h, xt, ldx, yt, ldy, R, I, O, t.on, s); break;
            case 3: launch_small<XT, 3>(d.a, d.bt, d.h, xt, ldx, yt, ldy, R, I, O, t.on, s); break;
            case 4: launch_small<XT, 4>(d.a, d.bt, d.h, xt, ldx, yt, ldy, R, I, O, t.on, s); break;
            case 5: launch_small<XT, 5>(d.a, d.bt, d.h, xt, ldx, yt, ldy, R, I, O, t.on, s); break;
            case 6: launch_small<XT, 6>(d.a, d.bt, d.h, xt, ldx, yt, ldy, R, I, O, t.on, s); break;
            case 7: launch_small<XT, 7>(d.a, d.bt, d.h, xt, ldx, yt, ldy, R, I, O, t.on, s); break;
            default: launch_small<XT, 8>(d.a, d.bt, d.h, xt, ldx, yt, ldy, R, I, O, t.on, s); break;
            }
        }
    } else {
        if (!g_on_host) return;   // eager: the host flag is the device flag
        float* hs = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_scratch_mu);
            Scratch& sc = t.scratch[s];
            if (sc.tokens < n) {   // only this stream uses it: what it queued must finish first
                cudaStreamSynchronize(s);
                cudaFree(sc.h);
                sc = Scratch{};
                if (cudaMalloc(&sc.h, (size_t) (kMaxSplits + 1) * n * g_lora.rank_max * sizeof(float)) != cudaSuccess)
                    throw std::runtime_error("lora_apply: scratch allocation failed");
                sc.tokens = n;
            }
            hs = sc.h;
        }
        // A x: rank fits one or two tiles across, so K is split until the GPU has blocks enough
        const int64_t tiles = ((h.rank + TILE - 1) / TILE) * ((n + TILE - 1) / TILE);
        const int64_t slabs = (h.n_in + TK - 1) / TK;
        const int splits = (int) std::max<int64_t>(1, std::min<int64_t>({(int64_t) kMaxSplits, (kSplitBlocks + tiles - 1) / tiles,
                                                                         slabs / 8}));
        const int kc = (int) (((slabs + splits - 1) / splits) * TK);
        const int64_t count = n * h.rank;
        float* hsum = hs + (int64_t) splits * count;
        const dim3 g1((unsigned) ((h.rank + TILE - 1) / TILE), (unsigned) ((n + TILE - 1) / TILE), (unsigned) splits);
        lora_tiled_kernel<XT, true><<<g1, TILE_THREADS, 0, s>>>(x, ldx, d.a, h.n_in, hs, h.rank, (int) n, (int) h.rank,
                                                               (int) h.n_in, kc, count, 0, t.on);
        lora_sum_kernel<<<(unsigned) ((count + 255) / 256), 256, 0, s>>>(hs, splits, count, hsum, t.on);
        const dim3 g2((unsigned) ((h.n_out + TILE - 1) / TILE), (unsigned) ((n + TILE - 1) / TILE));
        lora_tiled_kernel<float, false><<<g2, TILE_THREADS, 0, s>>>(hsum, h.rank, d.bt, h.n_out, y, ldy, (int) n,
                                                                    (int) h.n_out, (int) h.rank, (int) h.rank, 0, 1, t.on);
    }
    if (cudaPeekAtLastError() != cudaSuccess) throw std::runtime_error("lora_apply: launch failed");
}

}  // namespace

const Lora& lora() { return g_lora; }

bool lora_upload(const std::vector<LoraLayerHost>& layers, std::string& err) {
    if (g_lora.loaded) { err = "lora: an adapter is already loaded"; return false; }
    g_host.assign(layers.size(), LayerHost{});
    Lora l;
    l.adapted.assign(layers.size(), false);
    for (size_t i = 0; i < layers.size(); ++i) {
        const LoraLayerHost& src = layers[i];
        if (src.rank == 0) continue;
        if (src.rank < 0 || src.rank > kLoraMaxRank || src.n_in < 1 || src.n_out < 1 ||
            src.a.size() != (size_t) (src.rank * src.n_in) || src.b.size() != (size_t) (src.n_out * src.rank)) {
            err = "lora: bad tables for layer " + std::to_string(i);
            g_host.clear();
            return false;
        }
        LayerHost& h = g_host[i];
        h.rank = src.rank; h.n_in = src.n_in; h.n_out = src.n_out;
        h.a = src.a;
        h.bt.resize(src.b.size());
        for (int64_t o = 0; o < src.n_out; ++o)
            for (int64_t r = 0; r < src.rank; ++r) h.bt[(size_t) (r * src.n_out + o)] = src.b[(size_t) (o * src.rank + r)];
        l.adapted[i] = true;
        l.rank_max = std::max(l.rank_max, src.rank);
    }
    if (l.rank_max == 0) { err = "lora: the adapter covers no layer"; g_host.clear(); return false; }
    g_on_host = true;
    if (!upload_here(err)) { g_host.clear(); return false; }
    l.loaded = true;
    g_lora = l;
    return true;
}

bool lora_replicate(std::string& err) { return !g_lora.loaded || upload_here(err); }

void lora_set_enabled(bool on) {
    if (!g_lora.loaded || on == g_on_host) return;
    int prev = 0;
    cudaGetDevice(&prev);
    const int v = on ? 1 : 0;
    for (int d = 0; d < kDevices; ++d) {
        if (!g_dev[d].ready()) continue;
        cudaSetDevice(d);
        cudaDeviceSynchronize();   // nothing in flight may still read the flag
        cudaMemcpy(g_dev[d].on, &v, sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaSetDevice(prev);
    g_on_host = on;
}

bool lora_enabled() { return g_lora.loaded && g_on_host; }

uint64_t lora_device_bytes(const std::vector<LoraLayerHost>& layers) {
    uint64_t b = sizeof(int);
    for (const LoraLayerHost& l : layers) b += (uint64_t) (l.a.size() + l.b.size() + SMALL_N * l.rank) * sizeof(float);
    return b;
}

uint64_t lora_device_bytes() {
    uint64_t b = sizeof(int);
    for (const LayerHost& l : g_host) b += (uint64_t) (l.a.size() + l.bt.size() + SMALL_N * l.rank) * sizeof(float);
    return b;
}

void lora_apply(int64_t layer, const float* x, int64_t ldx, int64_t n, float* y, int64_t ldy, void* stream) {
    apply<float>(layer, x, ldx, n, y, ldy, stream);
}

void lora_apply_f16(int64_t layer, const uint16_t* x, int64_t ldx, int64_t n, float* y, int64_t ldy, void* stream) {
    apply<uint16_t>(layer, x, ldx, n, y, ldy, stream);
}

}  // namespace strata::kernels
