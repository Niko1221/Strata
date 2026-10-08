// dsv4/gpu.hpp - device layer, TWO interchangeable backends behind one API:
//   src/gpu.cpp       CPU-emulated device (malloc "VRAM", memcpy H2D/D2H, ggml-identical matvecs).
//                     Always buildable, and what the tests run against.
//   src/gpu_cuda.cu   real CUDA device (built by build.sh --cuda / cmake -DDSV4_WITH_CUDA=ON).
// Both define exactly the same dsv4::gpu symbols, so a binary links ONE of them; the loader, the memory
// plan and the HIT/MISS tier above this header never know which.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace dsv4 {
namespace gpu {

constexpr int kMaxHit = 8;  // max concurrent resident (HIT) experts per MoE layer

// Pointers into the VRAM expert pool: one slot is [gate | up | down], so the three rows are offsets
// of the same base. w[] is the routing weight already applied by the caller's combine step.
//
// An expert may also be given through the HOST pointers hg/hu/hd. experts_hit() then stages it into
// the device pool registered by staging() and evaluates it on the device like a resident one. That is
// what makes a MISS affordable: decoding an IQ1_M expert element by element on the host costs more
// per token than the rest of the forward pass put together, while the same decode on the device is
// ~100x cheaper and the PCIe transfer of the packed block hides behind the other kernels.
struct ExpPtrs {
    int n = 0;
    uint8_t* gate[kMaxHit] = {};
    uint8_t* up[kMaxHit] = {};
    uint8_t* down[kMaxHit] = {};
    const uint8_t* hg[kMaxHit] = {};
    const uint8_t* hu[kMaxHit] = {};
    const uint8_t* hd[kMaxHit] = {};
    float w[kMaxHit] = {};
    size_t bpe = 0;
};

/// Register the device scratch that host experts are staged through. Returns false, and leaves
/// staging unset, if the device cannot take it: experts_hit() then refuses host experts and the
/// caller keeps computing them on the host.
bool staging(uint8_t* device_pool, size_t bytes);

/// True when host experts can be staged. The CPU-emulated backend answers true with no pool at all:
/// its "device" memory is host memory, so staging would be a pointless copy.
bool has_staging();

bool is_emulated();
bool init(std::string& err);
void* alloc(size_t n);
void release(void* p);
void mem_info(size_t* free_bytes, size_t* total_bytes);

/// Can THIS backend compute over weights of ggml type `type`? The emulated device answers yes for every
/// type src/dequant.cpp can decode; the CUDA device answers yes only for the types it has a kernel for.
/// The loader uses this to avoid uploading weights no kernel could read, and to keep those matvecs on the
/// host instead of silently producing zeros.
bool type_supported(uint32_t type);

/// True only when all three matrices of a MoE layer have a kernel. That is the one case where giving the
/// layer VRAM slots is worth anything: a HIT expert is evaluated by experts_hit() and by nothing else.
bool experts_supported(uint32_t type_gate, uint32_t type_up, uint32_t type_down);

// Funzioni di copia memoria e calcolo dei kernel
void h2d(void* dst, const void* src, size_t bytes);
void d2h(void* dst, const void* src, size_t bytes);

/// y = W * x. False means this backend has no kernel for `type` (see type_supported) and wrote NOTHING:
/// the caller must fall back to the host. Never a partial or zero result.
bool matvec(uint32_t type, const uint8_t* W, int64_t rows, int64_t in, const float* d_x, float* d_y);

// ------------------------------------------------------------------ prefill (batched over tokens)
// A prompt of T tokens evaluates the SAME weight row T times. matvec() re-decodes that row once per
// token, which is what makes dsv4_run prefill cost the same as decode: 43 layers x 256 experts x
// top-6 is ~1.8 GB of weights PER TOKEN, and at one token at a time none of it is reused.
// matmul() decodes each row once and dots it against T activation rows.
//
// Layout is TOKEN-MAJOR on both sides, which is what the forward pass wants:
//   d_X[k*x_stride + i]      is token k's input vector, x_stride >= in
//   d_Y[k*y_stride + r]      is token k's output element r, r in [0,rows), y_stride >= rows
// `sel` (nullable, HOST memory) gathers: token k reads row sel[k] of d_X. That is the MoE case, where
// an expert is asked for the handful of tokens of the chunk that routed to it.
// False: no kernel for this type, nothing written (same contract as matvec()).
constexpr int kMaxSel = 8192;   // largest gather a single batched matmul may ask for

bool matmul(uint32_t type, const uint8_t* W, int64_t rows, int64_t in,
            const float* d_X, int T, const int32_t* sel, float* d_Y, int64_t x_stride, int64_t y_stride);

// One expert of a batched MoE layer: the n tokens of the chunk that routed to it, with their routing
// weights. gate/up/down are device pointers, or host pointers that get staged (see ExpPtrs).
struct ExpBatch {
    uint8_t* gate = nullptr;
    uint8_t* up = nullptr;
    uint8_t* down = nullptr;
    const uint8_t* hg = nullptr;
    const uint8_t* hu = nullptr;
    const uint8_t* hd = nullptr;
    const int32_t* tok = nullptr;   // token index in the chunk, per entry (host)
    const float* w = nullptr;       // routing weight, per entry (host)
    int n = 0;
};

/// The whole batched MoE HIT path, in expert order (deterministic, no atomics):
///   for each expert e, for each of its tokens t:
///     g = Wg*xt ; u = Wu*xt ; a = w_t * swiglu_clamped(g,u,limit) ; d_Y[t] += Wd*a
/// T is the chunk size: d_X is [T][dim] and d_Y is [T][dim], which this call ZEROES and then
/// accumulates into. Scratch: d_g/d_u/d_a hold maxN*ff floats each, d_dn holds maxN*dim, where maxN
/// is the largest e.n over the batch.
bool experts_batch(const ExpBatch* e, int n_exp, uint32_t type_g, uint32_t type_u, uint32_t type_d,
                   int64_t ff, int64_t dim, float swiglu_limit,
                   const float* d_X, int T, float* d_g, float* d_u, float* d_a, float* d_dn, float* d_Y);

/// One MoE layer's HIT experts, all resident in VRAM:
///   d_g[k] = Wg[k] * d_x ; d_u[k] = Wu[k] * d_x ; d_a[k] = w_k * swiglu_clamped(d_g[k], d_u[k], limit)
///   d_y    = sum_k Wd[k] * d_a[k]
/// Scratch must hold kMaxHit * ff (d_g, d_u, d_a) and dim (d_y) floats.
/// Scratch must hold kMaxHit * ff (d_g, d_u, d_a) and dim (d_y) floats.
/// False: one of the three types has no kernel and d_y was left untouched (see experts_supported).
/// Launches are asynchronous with respect to the host, so the caller can compute the MISS experts on the
/// CPU in the meantime and only pay for the device when it reads d_y back with d2h().
bool experts_hit(const ExpPtrs& p, uint32_t type_g, uint32_t type_u, uint32_t type_d, int64_t ff, int64_t dim,
                 float swiglu_limit, const float* d_x, float* d_g, float* d_u, float* d_a, float* d_y);

}  // namespace gpu
}  // namespace dsv4
