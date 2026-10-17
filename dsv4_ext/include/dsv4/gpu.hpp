/**
 * @file gpu.hpp
 * @brief Device layer: two interchangeable backends behind one API.
 *
 * This module abstracts the compute device with a unified `dsv4::gpu` namespace that has exactly
 * two implementations — only ONE is ever compiled into a binary:
 *
 * | Backend              | File          | Description                                          |
 * |----------------------|---------------|------------------------------------------------------|
 * | CPU-emulated         | [`src/gpu.cpp`](../../src/gpu.cpp) | malloc "VRAM", memcpy H2D/D2H, ggml-identical matvecs. Always buildable. |
 * | CUDA device          | [`src/gpu_cuda.cu`](../../src/gpu_cuda.cu) | Real kernels on NVIDIA GPU. Built with `--cuda` / cmake `-DDSV4_WITH_CUDA=ON`. |
 *
 * @par Why two backends?
 *   The emulated backend lets the entire HIT/MISS residency pipeline, memory planning, and tests
 *   run without a GPU — "VRAM" is host memory, H2D/D2H are memcpy, kernels use the same ggml-identical
 *   row math. The CUDA backend replaces this with real device kernels for production inference.
 *
 * @par MISS affordability
 *   A decoded IQ1_M expert element-by-element on the host costs more per token than the rest of the
 *   forward pass. On the device, `experts_hit()` evaluates resident experts ~100x cheaper, and the
 *   PCIe transfer of packed blocks hides behind other kernels via async staging.
 *
 * @par Capability gating
 *   `type_supported()` / `experts_supported()` decide at load time which tensors get uploaded and
 *   which MoE layers run on device. A `matvec()` or `experts_hit()` returning false wrote NOTHING —
 *   the caller falls back to host with one warning. Adding a ggml type = one trait in `dq_traits.hpp`
 *   plus one case in `matvec()`.
 *
 * @par Prefill batching
 *   A prompt of T tokens evaluates the SAME weight row T times. `matvec()` re-decodes once per token;
 *   `matmul()` decodes each row once and dots against T activation rows — saving the decode cost which
 *   dominates at ~1.8 GB/token for DeepSeek-V4-Flash's MoE path.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace dsv4 {
namespace gpu {

/**
 * @def kMaxHit
 * @brief Maximum concurrent resident (HIT) experts per MoE layer (8).
 *
 * Matches DeepSeek-V4-Flash's top-k routing: at most 6 experts are used per token, but the buffer
 * allows up to 8 for alignment and future-proofing.
 */
constexpr int kMaxHit = 8;

/**
 * @struct ExpPtrs
 * @brief Pointers into the VRAM expert pool for `experts_hit()`.
 *
 * An expert slot is [gate | up | down], so the three row arrays hold device pointers to each matrix.
 * Each expert also has an associated routing weight `w[]` already applied by the caller's combine step.
 *
 * @par Host fallback
 *   If device pointers (gate/up/down) are null, host pointers (hg/hu/hd) are used instead. The backend
 *   stages them into its internal pool via `staging()` and evaluates on-device like a resident expert.
 *   The CPU-emulated backend accepts host pointers directly since its "device" IS host memory.
 */
struct ExpPtrs {
    int n = 0;                                    ///< Number of experts (1..kMaxHit).
    uint8_t* gate[kMaxHit] = {};                  ///< Device pointers to gate matrices.
    uint8_t* up[kMaxHit] = {};                    ///< Device pointers to up matrices.
    uint8_t* down[kMaxHit] = {};                  ///< Device pointers to down matrices.
    const uint8_t* hg[kMaxHit] = {};              ///< Host fallback: gate source (staged if device has staging).
    const uint8_t* hu[kMaxHit] = {};              ///< Host fallback: up source.
    const uint8_t* hd[kMaxHit] = {};              ///< Host fallback: down source.
    float w[kMaxHit] = {};                        ///< Routing weight per expert (already applied by caller).
    size_t bpe = 0;                               ///< Bytes per expert (same for all three matrices).
};

/**
 * @brief Register device scratch buffer for staging host experts during `experts_hit()`.
 *
 * When the model encounters a MISS expert on the device, it needs temporary space to stage the
 * expert's gate/up/down pointers before evaluating. This function registers that staging area.
 *
 * The CPU-emulated backend ignores this (its "device" IS host memory) and always returns true.
 *
 * @param device_pool  Pointer to pre-allocated device memory for staging.
 * @param bytes        Size of the staging pool in bytes (must hold max expert size).
 * @return true if staging is available; false if the device cannot accept it.
 */
bool staging(uint8_t* device_pool, size_t bytes);

/**
 * @brief Check if host experts can be staged into the device pool.
 *
 * The CPU-emulated backend always returns true with no pool needed — its "device" memory is
 * host memory, so staging would be a pointless copy. The CUDA backend checks if `staging()` was called.
 * @return true if staging is available.
 */
bool has_staging();

/**
 * @brief Check if this is the CPU-emulated backend.
 * @return true for emulated (cpu), false for real device (cuda).
 */
bool is_emulated();

/**
 * @brief Initialize the GPU device layer.
 *
 * For the emulated backend, reads `DSV4_EMULATED_VRAM_MIB` environment variable (default 8192 MiB)
 * and allocates the "VRAM" pool. For CUDA, initializes the device and queries real VRAM.
 *
 * @param err  Output error message on failure.
 * @return true if initialization succeeded; false otherwise.
 */
bool init(std::string& err);

/**
 * @brief Allocate memory from the "VRAM" pool.
 *
 * The emulated backend allocates from a host memory pool of `DSV4_EMULATED_VRAM_MIB` bytes.
 * Allocations are never freed individually — the entire pool is released when the model unloads.
 *
 * @param n  Number of bytes to allocate.
 * @return Pointer to allocated memory, or nullptr if insufficient space or not initialized.
 */
void* alloc(size_t n);

/**
 * @brief Release a previously allocated pointer (frees from pool).
 *
 * In the emulated backend, this frees the underlying malloc but does NOT decrease `g_used` —
 * the pool is freed entirely when the model unloads, not per-expert.
 *
 * @param p  Pointer returned by a previous `alloc()` call.
 */
void release(void* p);

/**
 * @brief Query VRAM memory information.
 *
 * Both backends report free VRAM *already net of what this model has put on the card*.
 * CUDA uses `cudaMemGetInfo`; emulated backend tracks its own pool counter.
 * This ensures `plan_expert_memory()` is called with zero fixed costs to avoid double-counting.
 *
 * @param free_bytes   Output: bytes currently available (nullptr to skip).
 * @param total_bytes  Output: total VRAM size (nullptr to skip).
 */
void mem_info(size_t* free_bytes, size_t* total_bytes);

/**
 * @brief Check if this backend can compute over weights of the given ggml type.
 *
 * The emulated device answers yes for every type that `src/dequant.cpp` can decode.
 * The CUDA device answers yes only for types it has a dedicated kernel for.
 *
 * The loader uses this to avoid uploading weights no kernel could read, and to keep those
 * matvecs on the host instead of silently producing zeros.
 *
 * @param type  GgmlType enum value.
 * @return true if the backend has a kernel/decoder for this type.
 */
bool type_supported(uint32_t type);

/**
 * @brief Check if all three MoE matrices (gate/up/down) have kernels for their types.
 *
 * This is the one case where giving a layer VRAM slots is worth anything: only when all three
 * matrices can be evaluated on-device does `experts_hit()` become useful. If any type is unsupported,
 * the caller keeps computing that expert on the host.
 *
 * @param type_gate  GgmlType of the gate matrix.
 * @param type_up    GgmlType of the up matrix.
 * @param type_down  GgmlType of the down matrix.
 * @return true if all three types are supported by this backend.
 */
bool experts_supported(uint32_t type_gate, uint32_t type_up, uint32_t type_down);

// ================================================================= memory copy operations

/**
 * @brief Host-to-device memory copy.
 *
 * Emulated: plain memcpy (host to host pool). CUDA: cudaMemcpyHtoD.
 *
 * @param dst    Destination device pointer.
 * @param src    Source host pointer.
 * @param bytes  Number of bytes to copy.
 */
void h2d(void* dst, const void* src, size_t bytes);

/**
 * @brief Device-to-host memory copy.
 *
 * Emulated: plain memcpy (host pool to host). CUDA: cudaMemcpyDtoH.
 *
 * @param dst    Destination host pointer.
 * @param src    Source device pointer.
 * @param bytes  Number of bytes to copy.
 */
void d2h(void* dst, const void* src, size_t bytes);

// ================================================================= matrix-vector product (decode path)

/**
 * @brief Matrix-vector product: y = W * x, with on-the-fly dequantization.
 *
 * Each output row `y[r]` is the dot product of one weight row `W[r]` with input vector `x`.
 * Weights are decoded from compressed format on-the-fly (one block at a time) — this is the
 * decode path where each token evaluates experts independently.
 *
 * @param type      GgmlType enum: quantization format of weight matrix.
 * @param W         Pointer to compressed weight data [rows × in].
 * @param rows      Number of output rows (expert FFN width for gate/up, or dim for down).
 * @param in        Input vector dimension.
 * @param d_x       Input activation vector [in].
 * @param d_y       Output buffer [rows], populated with dot products.
 * @return true if computation succeeded; false if type is unsupported (wrote NOTHING).
 */
bool matvec(uint32_t type, const uint8_t* W, int64_t rows, int64_t in, const float* d_x, float* d_y);

// ================================================================= prefill: batched matrix multiplication over tokens

/**
 * @def kMaxSel
 * @brief Maximum gather indices a single batched matmul may request (8192).
 *
 * Used for the `sel` array in `matmul()` that gathers which token rows each expert evaluates.
 */
constexpr int kMaxSel = 8192;

/**
 * @brief Batched matrix multiplication: one weight matrix, T activation rows.
 *
 * The prefill primitive: a prompt of T tokens evaluates the SAME weight row T times. This function
 * decodes each row ONCE and dots it against T activation rows, saving the decode cost which dominates
 * at ~1.8 GB/token for DeepSeek-V4-Flash's MoE path.
 *
 * @par Layout (TOKEN-MAJOR on both sides)
 *   - `d_X[k*x_stride + i]` is token k's input vector, x_stride >= in
 *   - `d_Y[k*y_stride + r]` is token k's output element r, y_stride >= rows
 *
 * @par Gather mode
 *   `sel` (nullable, HOST memory) gathers: token k reads row `sel[k]` of d_X. This is the MoE case
 *   where an expert is asked for only the handful of tokens that routed to it within a chunk.
 *
 * @param type      GgmlType enum: quantization format of weight matrix.
 * @param W         Pointer to compressed weight data [rows × in].
 * @param rows      Number of output rows (expert FFN width for gate/up, or dim for down).
 * @param in        Input vector dimension.
 * @param d_X       Activation input buffer, TOKEN-MAJOR layout.
 * @param T         Number of tokens in the batch.
 * @param sel       Optional gather indices [T] (HOST memory); null = contiguous (sel[k] = k).
 * @param d_Y       Output buffer [T × rows], TOKEN-MAJOR layout.
 * @param x_stride  Stride between consecutive token vectors in d_X (>= in).
 * @param y_stride  Stride between consecutive output elements per row (>= rows).
 * @return true if computation succeeded; false if type is unsupported (wrote NOTHING).
 */
bool matmul(uint32_t type, const uint8_t* W, int64_t rows, int64_t in,
            const float* d_X, int T, const int32_t* sel, float* d_Y, int64_t x_stride, int64_t y_stride);

/**
 * @struct ExpBatch
 * @brief One expert's data for batched MoE evaluation (`experts_batch()`).
 *
 * Represents a single expert evaluated over a subset of tokens within a prefill chunk.
 * Device pointers (gate/up/down) are used if available; host pointers (hg/hu/hd) are staged
 * through the device pool via `staging()`.
 */
struct ExpBatch {
    uint8_t* gate = nullptr;              ///< Device pointer to gate matrix [ff × dim].
    uint8_t* up = nullptr;                ///< Device pointer to up matrix [ff × dim].
    uint8_t* down = nullptr;              ///< Device pointer to down matrix [dim × ff].
    const uint8_t* hg = nullptr;          ///< Host fallback: gate source.
    const uint8_t* hu = nullptr;          ///< Host fallback: up source.
    const uint8_t* hd = nullptr;          ///< Host fallback: down source.
    const int32_t* tok = nullptr;         ///< Token indices in chunk [n] (HOST memory).
    const float* w = nullptr;             ///< Routing weights [n] (HOST memory, already applied).
    int n = 0;                            ///< Number of tokens routed to this expert.
};

/**
 * @brief Batched MoE HIT path: evaluate multiple experts over token subsets in expert order.
 *
 * The full batched forward pass for a prefill chunk's MoE layer, evaluated in expert order
 * (deterministic, no atomics):
 *   For each expert e with n tokens:
 *     g = Wg * x[tok]  (gate matmul)
 *     u = Wu * x[tok]  (up matmul)
 *     a[w] = w_t * swiglu_clamped(g, u, limit)  (fused SwiGLU with routing weight)
 *     d_Y[tok] += Wd * a  (down matmul, accumulated)
 *
 * @par Scratch requirements
 *   - `d_g`, `d_u`, `d_a`: maxN × ff floats each (maxN = largest ExpBatch::n over all experts)
 *   - `d_dn`: maxN × dim floats
 *   - `d_Y`: T × dim floats, zeroed before accumulation
 *
 * @param e            Array of expert batches.
 * @param n_exp        Number of experts in the array.
 * @param type_g       GgmlType of gate matrices.
 * @param type_u       GgmlType of up matrices.
 * @param type_d       GgmlType of down matrices.
 * @param ff           Expert FFN intermediate dimension.
 * @param dim          Model embedding dimension.
 * @param swiglu_limit SwiGLU clamp limit (>0 enables clamping, <=0 disables).
 * @param d_X          Input activations [T × dim], TOKEN-MAJOR.
 * @param T            Chunk size (total tokens in the prefill batch).
 * @param d_g          Scratch: gate outputs [maxN × ff].
 * @param d_u          Scratch: up outputs [maxN × ff].
 * @param d_a          Scratch: SwiGLU outputs [maxN × ff].
 * @param d_dn         Scratch: down matmul outputs [maxN × dim].
 * @param d_Y          Output: accumulated expert results [T × dim], zeroed then accumulated.
 * @return true if computation succeeded; false if any type is unsupported (wrote NOTHING).
 */
bool experts_batch(const ExpBatch* e, int n_exp, uint32_t type_g, uint32_t type_u, uint32_t type_d,
                   int64_t ff, int64_t dim, float swiglu_limit,
                   const float* d_X, int T, float* d_g, float* d_u, float* d_a, float* d_dn, float* d_Y);

/**
 * @brief Evaluate all HIT experts of one MoE layer (resident in VRAM).
 *
 * The fast path for decode: all top-k experts are already resident in VRAM slots. Computes:
 *   g[k] = Wg[k] * x        (gate matvec)
 *   u[k] = Wu[k] * x        (up matvec)
 *   a[k] = w_k * swiglu_clamped(g[k], u[k], limit)  (fused SwiGLU with routing weight)
 *   y  = sum_k Wd[k] * a[k] (down matvec, accumulated)
 *
 * @par Async behavior
 *   Launches are asynchronous with respect to the host (CUDA streams). The caller can compute
 *   MISS experts on CPU in parallel and only pay for `d2h()` when reading back `d_y`.
 *
 * @param p            Expert pointers (`ExpPtrs`): up to kMaxHit resident experts.
 * @param type_g       GgmlType of gate matrices.
 * @param type_u       GgmlType of up matrices.
 * @param type_d       GgmlType of down matrices.
 * @param ff           Expert FFN intermediate dimension.
 * @param dim          Model embedding dimension.
 * @param swiglu_limit SwiGLU clamp limit (>0 enables clamping, <=0 disables).
 * @param d_x          Input activation vector [dim].
 * @param d_g          Scratch: gate outputs [kMaxHit × ff].
 * @param d_u          Scratch: up outputs [kMaxHit × ff].
 * @param d_a          Scratch: SwiGLU outputs [kMaxHit × ff].
 * @param d_y          Output: accumulated expert results [dim], zeroed then accumulated.
 * @return true if computation succeeded; false if any type is unsupported (d_y untouched).
 */
bool experts_hit(const ExpPtrs& p, uint32_t type_g, uint32_t type_u, uint32_t type_d, int64_t ff, int64_t dim,
                 float swiglu_limit, const float* d_x, float* d_g, float* d_u, float* d_a, float* d_y);

}  // namespace gpu
}  // namespace dsv4
