// helper_plugin_abi.h — strata-mi50, 2026-10-23
// C ABI between the engine and a helper-GPU plugin DLL: an expert tier (RemoteExperts) on a GPU that only another HIP runtime
// sees.  On Windows the Radeon Pro VII / MI50 (gfx906) is listed only by the HIP 5.x runtime (amdhip64.dll) and the RX 9070 XT
// only by HIP 6.x/7.x, so one process loads both: the engine (HIP 6/7) and strata_helper_gfx906.dll (HIP 5.x), each with its own
// runtime.  Everything that crosses this boundary is host memory: activations in, compact expert rows out, expert blobs read
// through a callback.  PIANO_DUAL_GPU_2026-10-23.md.
#pragma once
#include <stdint.h>

// 2 (2026-10-26): --remote-expert-opt through the plugin (open2, begin_opt, finish_opt); was 1
// 3 (2026-10-26): the helper's adaptive expert swaps (adapt)
// 4 (2026-10-27): the prompt path reads the experts the helper holds from its VRAM (copy_blobs, STRATA_PREFILL_HELPER_SOURCE)
// 5 (2026-10-27): the helper computes a share of its experts in prompt reads (prefill_reserve / begin / finish)
#define STRATA_HELPER_ABI_VERSION 5
/// The oldest plugin ABI the engine still loads (4: no prefill_* functions, the helper then does not compute in prompt reads).
#define STRATA_HELPER_ABI_MIN 4

#if defined(_WIN32)
#define STRATA_HELPER_EXPORT __declspec(dllexport)
#else
#define STRATA_HELPER_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// (layer, expert) -> the expert's blob in host memory, as the engine's ExpertSource::blob gives it; nullptr = not available.
typedef const uint8_t* (*strata_helper_blob_fn)(void* ctx, int64_t layer, int64_t expert);
/// (layer, expert) -> nonzero when the engine's own GPU cache already holds it (the helper then skips it).
typedef int (*strata_helper_primary_fn)(void* ctx, int32_t layer, int32_t expert);

typedef int (*strata_helper_abi_version_t)(void);
/// Loads the pack's expert layout (and Hadamard spec) into the plugin's own globals, picks the plugin runtime's device
/// `device` and reports its free memory.  0 = ok; otherwise `err` holds the reason.
typedef int (*strata_helper_init_t)(const char* pack_dir, int64_t n_layers, int64_t n_expert, int device,
                                    double* free_gib, char* err, int err_len);
/// Opens the tier: fills up to `slots` experts from `ranked` (pairs layer, expert, in rank order), skipping the ones
/// `primary` reports and the ones `claimed[layer * n_expert + expert]` marks; marks what it takes in `claimed`.
/// `auto_size` != 0: as many as fit with 512 MiB free.  Returns a handle, nullptr on failure.
typedef void* (*strata_helper_open_t)(int slots, const int32_t* ranked, int64_t n_ranked, strata_helper_primary_fn primary,
                                      strata_helper_blob_fn blob, void* ctx, uint8_t* claimed, int auto_size,
                                      char* err, int err_len);
typedef void (*strata_helper_close_t)(void* h);
/// RemoteExperts::begin: `x` is n_tok rows of H floats, `ids` n_tok * k expert ids; `kind` / `primary_res` may be null.
typedef int (*strata_helper_begin_t)(void* h, int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                                     const int32_t* kind, const int32_t* primary_res, char* err, int err_len);
/// RemoteExperts::finish: writes the rows it owns into `out` (n_tok * k rows of H floats).
typedef int (*strata_helper_finish_t)(void* h, float* out, char* err, int err_len);
typedef int (*strata_helper_owns_t)(void* h, int64_t index);
typedef int (*strata_helper_holds_t)(void* h, int64_t layer, int32_t expert);
/// Counters, in this order: resident, computed, launched_layers, returned_bytes, full_row_bytes; doubles: gib, ms_begin, ms_wait.
typedef void (*strata_helper_stats_t)(void* h, int64_t* counters5, double* doubles3);

// ---- version 2 (2026-10-26): --remote-expert-opt.  The helper reduces its expert rows to one weighted sum per token on its GPU
// (RemoteExpertOpt in the plugin's runtime), so one row of H floats per token crosses PCIe instead of one per owned (token, expert).
/// strata_helper_open with `reduce` != 0: the tier can reduce (begin_opt / finish_opt).
typedef void* (*strata_helper_open2_t)(int slots, const int32_t* ranked, int64_t n_ranked, strata_helper_primary_fn primary,
                                       strata_helper_blob_fn blob, void* ctx, uint8_t* claimed, int auto_size, int reduce,
                                       char* err, int err_len);
/// strata_helper_begin, reducing with the router weights `weights` (n_tok * k floats, the window's tokens).
typedef int (*strata_helper_begin_opt_t)(void* h, int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                                         const int32_t* kind, const int32_t* primary_res, const float* weights,
                                         char* err, int err_len);
/// After begin_opt: `sum` gets n_tok rows of H floats (this helper's weighted sum per token, zeros where it owns nothing) and
/// `owned` n_tok * k flags (the (token, expert) entries that sum covers).
typedef int (*strata_helper_finish_opt_t)(void* h, float* sum, uint8_t* owned, char* err, int err_len);

// ---- version 3 (2026-10-26): RemoteExpertOpt::adapt for a plugin tier.  The engine passes the routing usage and the primary's
// residency (n = n_layers * n_expert entries each) and the residency of every tier (`resident`, 1 = some GPU holds it), which the
// plugin updates for the swaps it makes.  The plugin swaps up to `max_swaps` of its cold experts for hot ones no GPU holds, reading
// the new blobs through `blob` and waiting for each copy, so no blob pointer is used after return.  `swapped` gets the count.
typedef int (*strata_helper_adapt_t)(void* h, const float* usage, const int32_t* primary, uint8_t* resident, int64_t n,
                                     int max_swaps, strata_helper_blob_fn blob, void* ctx, int* swapped, char* err, int err_len);

// ---- version 4 (2026-10-27): RemoteExperts::copy_cached_blobs for a plugin tier.  The experts the helper holds are left out of
// the engine's RAM copy, so without this the prompt path read them from the model files (2-3 GB/s on the owner's PC) while the
// helper sat idle.  Copies up to n blobs into the host buffers dst[i] and waits; hit[i] = 1 where the helper held (layers[i],
// experts[i]).  0 = ok (the hits are complete); otherwise err says why and the caller reads the rest from its fallback.
typedef int (*strata_helper_copy_blobs_t)(void* h, const int32_t* layers, const int32_t* experts, uint8_t* const* dst, int64_t n,
                                          uint8_t* hit, char* err, int err_len);

// ---- version 5 (2026-10-27): STRATA_PREFILL_HELPER.  The helper computes, in prompt reads, the experts the engine marks in `take`
// (experts it holds in its VRAM cache), in parallel with the engine's own GPU: gate/up MMQ, SwiGLU, down MMQ and the weighted sum per
// token (fp16), all asynchronously on its own stream.  The engine downloads each layer's fp16 activations (T x n_embd), routed ids and
// router weights (T x k) to host buffers, calls prefill_begin (returns once the work is queued), runs its own experts, then
// prefill_finish and adds the returned sums in its combine.  PIANO_MI50_PROMPT_2026-10-27.md.
/// Allocates the work space for chunks of up to `t_max` tokens (the helper's free VRAM decides: 0 = ok, otherwise `err`).  `layer_ok`
/// (n_layers bytes) gets 1 for the layers it can compute (native IQ/Q formats MMQ covers, no Hadamard fold, no NVFP4 tail).  `x16` and
/// `sum16` get its own page-locked host buffers (t_max x n_embd fp16): the engine may use them as the copy targets/sources.
typedef int (*strata_helper_prefill_reserve_t)(void* h, int64_t t_max, uint8_t* layer_ok, int64_t n_layers, uint16_t** x16,
                                               uint16_t** sum16, char* err, int err_len);
/// Queues layer `layer` for `t` tokens: x16 (t x n_embd fp16), ids / w (t x k) router ids and weights, take[expert] != 0 the experts to
/// compute (all must be in its cache).  Returns before the GPU is done; the host buffers are read before return.
typedef int (*strata_helper_prefill_begin_t)(void* h, int64_t layer, int64_t t, int64_t k, const uint16_t* x16, const int32_t* ids,
                                             const float* w, const uint8_t* take, char* err, int err_len);
/// Waits for the queued layer and writes its per-token weighted sums (t x n_embd fp16) to `sum16`; `ms_wait` gets the host time waited.
typedef int (*strata_helper_prefill_finish_t)(void* h, uint16_t* sum16, double* ms_wait, char* err, int err_len);

#ifdef __cplusplus
}
#endif
