// include/strata/dsv4/attn.hpp - DeepSeek V4-Flash dense MLA attention block (hash layers L0/L1),
// dsv4-mvp slice S3.
//
// GPU port of src/dsv4/dsv4ref.cpp's attention_forward() for the pure-hash layers (compress ratio 0):
//   qr  = rmsnorm(x @ wq_a.T, q_norm)                       [s][1024]
//   q   = qr @ wq_b.T -> [s][64][512]; per-head RMS renorm; plain rope on the last 64 dims
//   kv  = rmsnorm(x @ wkv.T, kv_norm); plain rope on the last 64 dims
//   o   = MQA-512 attention over the sliding-window cache (topk indirection, attn_sink in the
//         softmax DENOMINATOR only, scale = 512^-0.5, no mscale)
//   inverse rope on o, then the grouped O projection o.view(s,8,4096) @ wo_a -> [s][8192] @ wo_b.
//
// The kernels are head_dim-512 MQA kernels (native_flash_attn tops out at head_dim 256, so this is
// new code).  Q8_0 weights are dequantized on the fly (fp16 scale + 32 int8 per 34-byte block),
// matching dsv4ref's dequant_q8_0 bit-for-bit; accumulation is FP32 like the host reference.
//
// Numerical contract: validated against the golden oracle dumps
// (/data/research/runs/202610-dsv4-golden/prefill/L0{0,1}_post_attn_{main,hc}.npy) by
// strata-dsv4-attn-test at cos >= 0.9999 / rel_amax <= 3e-2.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace strata::dsv4 {

// Device pointers of one layer's attention weights.  Q8_0 tensors are raw GGUF row payloads
// (row-major [rows][cols], (cols/32)*34 bytes per row); the float tensors are dequantized F32.
struct Dsv4AttnWeights {
    const uint8_t* wq_a = nullptr;  // Q8_0 [1024][4096]
    const uint8_t* wq_b = nullptr;  // Q8_0 [64*512=32768][1024]
    const uint8_t* wkv = nullptr;   // Q8_0 [512][4096]
    const uint8_t* wo_a = nullptr;  // Q8_0 [8*1024=8192][4096], row r reads x group (r/1024)
    const uint8_t* wo_b = nullptr;  // Q8_0 [4096][8192]
    const float* q_norm = nullptr;  // F32 [1024]
    const float* kv_norm = nullptr; // F32 [512]
    const float* sinks = nullptr;   // F32 [64] (denominator only)
};

// Plain (non-YaRN) rope tables for the hash layers: cos/sin [max_pos][half], half = n_rot/2 = 32,
// built host-side with dsv4ref's rope_build(dim=64, theta=1e4, orig=0) and uploaded once.
struct Dsv4RopeTables {
    const float* cos = nullptr;
    const float* sin = nullptr;
    int half = 32;
};

// Reusable scratch.  max_k must cover the widest topk row (window size for hash layers).
struct Dsv4AttnWorkspace {
    float* qr = nullptr;      // [max_s][1024]
    float* q = nullptr;       // [max_s][64*512]
    float* kv_new = nullptr;  // [max_s][512]
    float* attn = nullptr;    // [max_s][64*512]
    float* o8k = nullptr;     // [max_s][8192]  [S7c] warp-transposed layout (wo_a writes it with
                              // permuted addressing; only wo_b reads it, likewise transposed)
    float* cache_tmp = nullptr;  // [max_s][512] ring-shift staging for prefill s > window
    int32_t* topk = nullptr;  // [max_s][max_k]
    // [S7c] warp-slice-transposed gemm inputs (see dsv4_xpose1024_rows): within every 1024-float
    // slice, float q = 32L+4j+c lives at j*128 + L*4 + c, so a gemm warp's per-row-block float4
    // reads are fully coalesced instead of 128B-strided.  Pure fp32 permutation.
    float* xt = nullptr;      // [max_s][4096] attention block input, transposed
    float* qr_t = nullptr;    // [max_s][1024] qr after q_norm, transposed
    float* attn_t = nullptr;  // [max_s][64*512] attention output after inverse rope, transposed
    int max_s = 0;
    int max_k = 0;
};

bool dsv4_attn_workspace_init(Dsv4AttnWorkspace& ws, int max_s, int max_k);
void dsv4_attn_workspace_free(Dsv4AttnWorkspace& ws);

// Host-side sliding-window topk builder (dsv4ref window_topk, hash layers: window part only).
// Writes s rows of K = min(s, win) (prefill) or win (decode) indices into topk (row-major,
// -1 = masked); returns K.  topk must hold s*win entries.
int dsv4_window_topk(int win, int s, int64_t start_pos, int32_t* topk);
// [chunk] window-only list padded to `win` entries (prefill rows, start_pos 0)
int dsv4_window_topk_pad(int win, int s, int32_t* topk);

// One dense MLA attention block: x [s][4096] (attention input, already hc_pre + rmsnormed) ->
// out [s][4096] (block output, pre-hc_post).  kv_cache is the layer's sliding-window latent cache
// [win][512]; for start_pos == 0 the fresh kv rows are also written into it (ring shift when
// s > win via ws.cache_tmp), for decode the single new row lands at slot start_pos % win and
// attention runs over the cache.  Hash layers only: no compressor/indexer (S4/S5).
// Returns false (with a message on stderr) when the call is out of bounds: s outside 1..ws.max_s,
// negative start_pos, or a prefill wider than the sliding window (R-02/R-04).
bool dsv4_attn_forward(const Dsv4AttnWeights& w, const Dsv4RopeTables& rope,
                       Dsv4AttnWorkspace& ws, const float* x, float* out, float* kv_cache,
                       int s, int64_t start_pos, int sliding_window, cudaStream_t stream);

// ---------------------------------------------------------------------------
// [S4] internal kernel launchers, shared with dsv4_comp.cu (the compress layers' attention).  These are
// the same kernels dsv4_attn_forward sequences; they are exported so the compressor block reuses the
// validated S3 code paths bit-for-bit instead of carrying copies.  Not part of the S3 contract.
// ---------------------------------------------------------------------------

// out[t][r] = dot(x[t] (+ group offset), dequant(W_row r)); Q8_0 rows, FP32 accumulation.
// [S7c] x must point at the warp-slice-transposed copy (dsv4_xpose1024_rows); x_row_stride must be
// a multiple of 1024 and grouped cols a multiple of 1024 (all dsv4 shapes qualify).  out_xpose != 0
// writes out[t][r] at the transposed position of index r instead (wo_a -> wo_b link).
void dsv4_gemm_q8_rows(const uint8_t* w, int rows, int cols, int group_rows, const float* x, int m,
                       int x_row_stride, float* out, cudaStream_t stream, int out_xpose = 0);

// [S7c] permute nfloats (multiple of 1024) of fp32 into the warp-slice-transposed layout.
void dsv4_xpose1024_rows(const float* x, float* xt, int64_t nfloats, cudaStream_t stream);

// [S7c] permute nfloats (multiple of 2048) into the 64-floats-per-lane layout of the NVFP4 GEMV
// (dsv4_moe.cu): within a 2048-float chunk the float at index r = 64L + 16s + 4v + c (lane L of the
// row's 32-lane half, sub-block s, float4 v, component c) is stored at (s*4+v)*128 + L*4 + c, so
// the gemv's per-lane 16 B reads are contiguous across a warp (4 wavefronts instead of 32).
void dsv4_xpose2048_rows(const float* x, float* xt, int64_t nfloats, cudaStream_t stream);

// RMS norm rows in place (double sum-of-squares, eps 1e-6); w == nullptr -> plain renorm.
void dsv4_rmsnorm_rows(float* x, int rows, int cols, const float* w, cudaStream_t stream);

// Rope on the last 2*half dims, interleaved pairs; row r of token (r / rows_per_token) rotates at
// position start_pos + (r / rows_per_token) * pos_mul.  pos_mul is 1 for plain rows and the compress
// ratio for compressed tokens (phase j*ratio).  inverse != 0 applies the output-side inverse rotary.
// [S7e] rmsnorm + rope as one launch (same values as dsv4_rmsnorm_rows then dsv4_rope_rows_mul on
// the whole row, width == cols).
void dsv4_rmsnorm_rope_rows(float* x, int rows, int cols, int half, int rows_per_token, int pos_mul,
                            const Dsv4RopeTables& rope, int64_t start_pos, int inverse,
                            const float* w, cudaStream_t stream);

void dsv4_rope_rows_mul(float* x, int rows, int width, int half, int rows_per_token, int pos_mul,
                        const Dsv4RopeTables& rope, int64_t start_pos, int inverse,
                        cudaStream_t stream);

// Dense MQA-512 attention over a [s][K] topk list (-1 = masked): scale 512^-0.5, attn_sink in the
// softmax denominator only.  smem sizing is internal.  slots is the number of readable rows in kv:
// an index outside [0, slots) is masked exactly like -1 (R8).
void dsv4_mqa512_rows(const float* q, const float* kv, const int32_t* topk, const float* sinks, int s,
                      int K, int slots, float scale, float* out, cudaStream_t stream);

// ---------------------------------------------------------------------------
// [M2] batched prefill attention (STRATA_DSV4_ATTN_BATCH, default on; =0 selects the per-row path).
//
// A chunked-prefill pass runs 128 rows through the same attention block, and the default path launches
// one MQA-512 grid per row (kNHead blocks = 4 % of the SMs) after uploading a host-built topk table and
// copying that row's 2 KB KV slice into the ring.  The batched path keeps the per-row numerics exactly
// - each row still accumulates its K list in the same order, in the same 128-thread structure - and
// only batches the "row" dimension: one launch for all s rows, tables built on the device, and the ring
// written after the launch instead of per row.
// ---------------------------------------------------------------------------

// True when the batched path is selected (default); STRATA_DSV4_ATTN_BATCH=0 turns it off.
bool dsv4_attn_batch_enabled();

// Device twin of dsv4_window_topk / dsv4_window_topk_pad: `stride` is the row pitch in entries, `nent`
// the number of entries written per row (values outside the window are -1).  Same values as the host
// builders by construction.
void dsv4_window_topk_dev(int win, int s, int64_t start_pos, int stride, int nent, int32_t* topk,
                          cudaStream_t stream);

// Device twin of the start_pos > 0 branch of dsv4_compress_topk: writes the `kc` compressed-slot
// columns at [kw, kw + kc) of each `stride`-pitch row (t + offset while t < (start_pos + j + 1)/ratio,
// else -1).  The r=4 indexer block is NOT built here - its order comes from the selection kernel and
// stays a device-to-device copy.
void dsv4_compress_topk_dev(int64_t ratio, int s, int64_t start_pos, int64_t offset, int kw, int kc,
                            int stride, int32_t* topk, cudaStream_t stream);

// Row-batched MQA-512 for start_pos > 0, where the table is a pass-relative ring list: a window entry
// (< win) resolves to this pass's kv_new row when that row is at or before the attending row's own
// position, everything else (older ring slots and the compressed slots at >= win) reads kv_ring.  The
// caller must therefore write the pass's KV rows into the ring AFTER this launch: writing them first
// would expose a future row's KV to an earlier row ([R-01]), which the per-row default path avoids.
void dsv4_mqa512_rows_ring(const float* q, const float* kv_new, const float* kv_ring,
                           const int32_t* topk, const float* sinks, int s, int K, int slots, int win,
                           int64_t start_pos, float scale, float* out, cudaStream_t stream);

}  // namespace strata::dsv4
