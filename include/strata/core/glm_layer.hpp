// include/strata/core/glm_layer.hpp - ONE glm5-next block, composed.
//
// This is a SECOND block scaffold, not a set of flags on the first one, and the reason is structural rather
// than stylistic.  A `qwen4exp` block is
//
//     gr_read(attn) -> mixer -> gr_write(attn) -> gr_read(ffn) -> MoE -> gr_write(ffn)
//
// where `gr_read`/`gr_write` is a low-rank mixer with a learned gate and a mean injection, and PLE rides in
// front of layer 1.  A glm5-next block is
//
//     hc_pre(attn) -> hc_mix -> <attn_norm> -> <KDA | MLA> -> hc_post(attn)
//     hc_pre(ffn)  -> hc_mix -> <ffn_norm>  -> <dense FFN | MoE> -> hc_post(ffn)
//
// **THE RESIDUAL IS THE FOUR-STREAM mHC STACK AND `hc_post` IS THE ONLY THING THAT WRITES IT.**  There is no
// `x = x + sublayer(x)` anywhere in the model: the attention and the MoE both have their residual add turned
// OFF (the reference passes `add_residual = false` / `add_input = false`), because `hc_post` already folds the
// sublayer output back into all four streams through `post` and `comb`.  A port that "helpfully" adds a plain
// residual anywhere produces a finite, fluent, wrong model.
//
// The other traps, each of which runs fine when wrong:
//
//   * `hc_pre` is a **WEIGHTLESS** RMSNorm of the flattened `[n_embd * hc]` residual, then `hc_*_fn` - there is
//     no learned norm before the projection.
//   * `pre` is `sigmoid + eps`; `post` is `2 * sigmoid`.  The factor of 2 has no counterpart anywhere else.
//   * the final collapse before the head is the stream **MEAN**, not their sum.
//   * KDA's three conv streams have one shared state, and the delta state is per head and updated in place.
//   * MLA's cache holds the 512-wide latent and K == V; there is no RoPE on any layer.
//   * the SwiGLU clamp in the GGUF is **inert** in the reference (see `glm_elt.cu`).
#pragma once

#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/weights.hpp"

#include "strata/kernels/glm.hpp"

#include <cstdint>
#include <string>

namespace strata::core {

struct Doorbell;

/// Everything one glm5-next block needs that is NOT per-layer persistent state.  Carved once, like
/// `GdnBuffers`; the sizes are all functions of `ModelGeometry`.
///
/// The per-layer PERSISTENT state (the KDA conv + delta state, the MLA latent cache) is deliberately NOT here:
/// it must survive every token and there is one per layer, which is a different lifetime from scratch.
struct GlmBuffers {
    // ---- mHC, used twice per layer ----
    /// **THE mHC STACK NEEDS TWO BUFFERS AND THAT IS NOT AN OVERSIGHT.**  `hc_post` reads all `hc` streams of
    /// the residual for every output element, so writing it in place would overwrite the streams the next
    /// element still has to read.  `bb.R` is the authoritative stack and this is where one half's write lands
    /// before it is copied back.
    float* res_scratch = nullptr;      ///< hc * n_embd
    uint16_t* normed_bf16 = nullptr;   ///< n_embd * hc, bf16
    float* normed_f32 = nullptr;       ///< n_embd * hc: the same values, for the quantized images below
    float* mixes = nullptr;            ///< hc_mix (24)
    float* pre = nullptr;              ///< hc
    float* post = nullptr;             ///< hc
    float* comb = nullptr;             ///< hc * hc
    uint8_t* normed_q8k = nullptr;     ///< n_embd * hc, block_q8_K
    uint8_t* normed_q8_0 = nullptr;    ///< n_embd * hc, block_q8_0

    // ---- the normed sublayer input (`attn_norm` / `ffn_norm` applied), and its images ----
    float* cur = nullptr;              ///< n_embd
    uint8_t* cur_q8k = nullptr;        ///< n_embd, block_q8_K
    uint8_t* cur_q8_0 = nullptr;       ///< n_embd, block_q8_0
    uint16_t* cur_bf16 = nullptr;      ///< n_embd

    // ---- ONE WIDE INTERMEDIATE, sized for the widest the arch uses (`wide` below) ----
    //
    // Both `n_v` (KDA's out projection) and the FFN width are narrower than `wide`, and the kernels are told
    // the real length, so one pair of images serves every projection whose input is not `cur`.  Sized images
    // per role would be four more buffers and four more chances to size one wrong.
    uint8_t* wide_q8k = nullptr;       ///< wide, block_q8_K
    uint8_t* wide_q8_0 = nullptr;      ///< wide, block_q8_0
    uint16_t* wide_bf16 = nullptr;     ///< wide

    // ---- KDA ----
    /// **`ntok` TOKEN BLOCKS OF `[q; k; v]`, NOT ONE INTERLEAVED `[q;k;v]` PER TOKEN.**  At `ntok == 1` the two
    /// readings are the same bytes; at `ntok > 1` each stream's `ntok` columns are contiguous, which is what a
    /// projection writing `ntok x n_v` produces and what `glm_kda_conv_silu3` wants.  `q`/`kk`/`v` below are
    /// the three stream bases and `qkv_ld` the token stride to hand the conv - use those, not this.
    float* qkv = nullptr;              ///< 3 * n_v per token
    float* q = nullptr;                ///< `qkv`
    float* kk = nullptr;               ///< `qkv + ntok * n_v`
    float* v = nullptr;                ///< `qkv + 2 * ntok * n_v`
    int64_t qkv_ld = 0;                ///< the conv's token stride: `n_v`, in both the split and the flat form
    int64_t ntok = 1;                  ///< how many tokens the pointers above describe
    float* gate = nullptr;             ///< n_v
    float* beta = nullptr;             ///< n_head (raw; `glm_kda_delta` applies the sigmoid itself)
    float* raw = nullptr;              ///< n_v: `ssm_f_b(ssm_f_a(x))`, the decay's pre-activation
    float* z = nullptr;                ///< n_v: the output gate's pre-activation
    /// The recurrence's output.  `rms_norm(o, ssm_norm) * sigmoid(z)` is applied IN PLACE here, because the
    /// weight's own rows are the head rows and there is nothing left to keep afterwards.
    float* o = nullptr;                ///< n_v
    /// `n_embd`: the attention sublayer's output, i.e. `attn_output @ y`.  Its own buffer and not `cur`,
    /// because `cur` still holds the normed INPUT this projection read and the FFN half's `hc_read` is about
    /// to overwrite it - the two are live at the same time.
    float* attn_out = nullptr;         ///< n_embd
    /// `ntok` blocks of `[ssm_g_a(128); ssm_f_a(128)]`.  `tail` is the g half and `tail_f` the f half, each
    /// `ntok` contiguous columns, because the two are separate projections with separate inputs.
    float* tail = nullptr;             ///< 2 * 128 per token
    float* tail_f = nullptr;           ///< `tail + ntok * 128`
    uint8_t* tail_q8_0 = nullptr;      ///< 128, block_q8_0: what `ssm_g_b`/`ssm_f_b` read (128 is not a Q8_K width)

    // ---- MLA ----
    float* qr = nullptr;               ///< q_lora_rank: the q latent, normed in place
    float* qfull = nullptr;            ///< n_head * mla_head_dim: `wq_b @ qr`
    /// **ONE IMAGE PAIR FOR THE WHOLE HEAD STACK, NOT ONE PER HEAD.**  The absorption and de-absorption loops
    /// below index a folded per-head matrix one band at a time, and each band is a separate projection with its
    /// own input - so the first cut quantized each head's slice into a shared pair just before projecting it.
    /// That is `2 * n_head` launches a loop where the slices are CONTIGUOUS in `qfull`/`kqv`: quantizing the
    /// whole stack once produces the identical blocks at the identical offsets (`mla_head_dim` is a whole
    /// number of both a 256-element Q8_K block and a 32-element Q8_0 one), and the per-head code below reads
    /// band `h` at `q8k_bytes(mla_head_dim) * h`.  Measured on one card, 350 tokens: those two loops were
    /// ~5.6 K of the ~8.3 K kernel launches a token, and the quantizes they issued were 83% of ALL the
    /// `quantize_q8_K`/`quantize_q8_0` calls in the process.
    uint8_t* heads_q8k = nullptr;      ///< n_head * max(mla_head_dim, kv_lora_rank), block_q8_K
    uint8_t* heads_q8_0 = nullptr;     ///< the same width, block_q8_0
    float* qabs = nullptr;             ///< n_head * kv_lora_rank: `wk_b @ q` - the ABSORBED query (Qcur)
    float* kv_cmpr = nullptr;          ///< kv_lora_rank: the latent that goes into the cache AND is V
    float* kqv = nullptr;              ///< n_head * kv_lora_rank: the attention output, pre-de-absorption
    float* head_out = nullptr;         ///< n_head * mla_head_dim: `wv_b @ kqv`

    // ---- the dense FFN's two projections, and nothing else ----
    //
    // The FFN's HIDDEN (`silu(gate) * up`, `ffn width`) needs no images of its own: it is the input to
    // `ffn_down`, which is a wide projection, so it uses `wide_*` like the KDA output does.  The sublayer's
    // RESULT has no buffer here at all - a dense layer writes it straight into `bb.block_out`, which is the
    // `n_embd` scratch the block already owns and the pointer `hc_post` is handed.
    float* ffn_gate = nullptr;         ///< ffn width
    float* ffn_up = nullptr;           ///< ffn width
};

/// `ntok` is how many tokens the carve is for: every field above is per token and the whole carve scales with
/// it, so `ntok == 1` is exactly the decode path's scratch and `ntok == T` is what a chunk needs.
uint64_t glm_buffers_bytes(const ModelGeometry& g, int64_t ntok = 1);
uint64_t glm_buffers_init(const ModelGeometry& g, int64_t ntok, void* base, GlmBuffers& b);

/// The per-layer PERSISTENT state, which is two different things depending on the layer's mixer.
struct GlmLayerState {
    /// KDA: the delta rule's per-head state, `n_head * kda_head_dim^2` floats with the ROW fastest, plus the
    /// conv history.  Read and written in place every token, so it is zeroed at the start of a sequence.
    float* kda_state = nullptr;
    float* kda_conv = nullptr;         ///< (kernel - 1) * 3 * n_v, TAP fastest
    /// MLA: the fp16 latent cache, `max_cells * kv_lora_rank`, one row per token.  K AND V - there is no
    /// second cache.  Indexed by ABSOLUTE position, so `max_cells` travels with it: a write one row past the
    /// end is an arena corruption with nothing to see afterwards, and the bounds check costs one compare.
    uint16_t* mla_cache = nullptr;
    /// Rows in `mla_cache` (0 on a KDA layer).  The KDA half has no use for it - its state is fixed-size.
    int64_t max_cells = 0;
};

/// Floats the KDA state needs (the delta state plus the conv history), zero when the arch has no KDA.
uint64_t glm_kda_state_floats(const ModelGeometry& g);
/// Bytes ONE MLA layer's latent cache needs for `max_cells` tokens, zero when the arch has no MLA.
uint64_t glm_mla_cache_bytes(const ModelGeometry& g, int64_t max_cells);
/// True when this layer is one of the KDA (linear) layers rather than an MLA one.
bool glm_is_kda_layer(const ModelGeometry& g, int64_t layer);

/// Bytes ONE layer's persistent state takes, which is the KDA pair on a linear layer and a latent cache on an
/// MLA one.  The session carve and the pointer setup below must agree byte for byte, so they both come from
/// here rather than from a size written out twice.
uint64_t glm_layer_state_bytes(const ModelGeometry& g, int64_t max_cells, int64_t layer);

/// Points `st` at `base` and returns the bytes it consumed (== `glm_layer_state_bytes`).  Both kinds of state
/// start ZEROED for a new sequence; the pointer arithmetic that splits a KDA layer's carve into its delta state
/// and its conv history lives here, next to the function that sized it.
uint64_t glm_state_init(const ModelGeometry& g, int64_t max_cells, int64_t layer, void* base, GlmLayerState& st);

/// THE BLOCK, SPLIT AT THE ROUTER like the first architecture's, and for the same reason: the routed experts do
/// not fit in VRAM, so a host loop has to run a pool between the two calls.
///
///     pre[l]   hc_pre(attn) -> KDA|MLA -> hc_post(attn) -> hc_pre(ffn) -> ffn_norm -> dense FFN | moe_route
///     post[l]  (MoE layers only:) the shared expert and the combination -> hc_post(ffn)
///
/// `pre` leaves EVERYTHING `post` needs inside `b` itself - the FFN's normed input, and the second half's
/// `post`/`comb` - so the host loop that runs the expert pool between the two calls cannot disturb it by
/// touching the shared `BlockBuffers`.  The constraint is the same as the first architecture's, moved: nothing
/// may write this layer's `GlmBuffers` between the halves.
bool glm_block_layer_pre(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t pos,
                         int32_t pos_base, const GlmBuffers& b, const GlmLayerState& st, const MoEBuffers& mb,
                         int64_t k, const BlockBuffers& bb, void* stream, std::string& err, const Doorbell* db);

/// `b` is needed here and not only in `pre`: the FFN half's `hc_post` reads THIS half's `post`/`comb`, which
/// `hc_read` left in the caller's `GlmBuffers`.  Nothing may write that scratch between the two calls.
bool glm_block_layer_post(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t k,
                          const GlmBuffers& b, const MoEBuffers& mb, const BlockBuffers& bb, const float* parts,
                          void* stream, std::string& err);

/// The whole block, for a caller that already has `parts`.
bool glm_block_layer(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t pos,
                     int32_t pos_base, const GlmBuffers& b, const GlmLayerState& st, const MoEBuffers& mb, int64_t k,
                     const BlockBuffers& bb, const float* parts, void* stream, std::string& err,
                     const Doorbell* db);

// ================================ chunked prefill ================================

/// ONE TOKEN AT A TIME, A LAYER'S SCRATCH IS FREE TO BE SHARED; A CHUNK IS NOT.
///
/// `glm_block_layer_pre` documents the constraint that makes the two-call split work: *pre* leaves everything
/// *post* needs inside the caller's buffers, and nothing may write them between the halves.  A chunked pass
/// runs `pre` for every token of the chunk, then the pool, then `post` for every token - so between token 0's
/// `pre` and token 0's `post`, tokens 1..T-1 run their own `pre` over the same scratch.  Only two fields of
/// `GlmBuffers` are still live at that point, and this struct is what holds them per token:
///
///   * `post` and `comb` - the mHC write the FFN half's `hc_read` produced, which `hc_post` consumes.  (The
///     ATTENTION half's pair is written and consumed inside `pre` and needs no row of its own.)
///   * `cur` - the FFN's normed input, which is what the pool and the router both read.
///   * `R` and `block_out` - the residual and the FFN half's output.  `R` is read at the top of `pre` and
///     written by `hc_write`, so it is sequence state that happens to live in the shared block scratch.
///   * `shared`, `weights`, `ids` - the MoE host's own inputs: what the shared expert produced, the router's
///     weights, and the routed ids the pool is handed.  All three are written by `pre` and read by `post` or by
///     the host between the halves.
///
/// **EVERYTHING ELSE IS GENUINELY SCRATCH** - `res_scratch`, the `normed_*`/`wide_*` images, `mixes`, `pre`,
/// the KDA and MLA working buffers, `ffn_gate`/`ffn_up`, `logits` - because it is written and consumed inside
/// one `pre`.  Those stay in the single shared `GlmBuffers`/`MoEBuffers`, which is why a chunk of 128 costs
/// ~240 KiB a token here rather than a second copy of the whole 1 MiB layer scratch.
struct GlmChunkBuffers {
    float* R = nullptr;          ///< T rows of (hc, n_embd)
    float* cur = nullptr;        ///< T rows of n_embd
    float* block_out = nullptr;  ///< T rows of n_embd
    float* shared = nullptr;     ///< T rows of n_embd
    float* parts = nullptr;      ///< T rows of (k, n_embd), the pool's UNWEIGHTED outputs
    float* post = nullptr;       ///< T rows of hc
    float* comb = nullptr;       ///< T rows of (hc, hc)
    float* weights = nullptr;    ///< T rows of k
    int32_t* ids = nullptr;      ///< T rows of k
    int64_t n_embd = 0, hc = 0, k = 0, T = 0;
};

/// Bytes `glm_chunk_init` needs for a `T`-token chunk at routing width `k`.  Zero when `T < 1`.
uint64_t glm_chunk_bytes(const ModelGeometry& g, int64_t k, int64_t T);
/// Carves `base` into `c` and returns the bytes used (== `glm_chunk_bytes`).
uint64_t glm_chunk_init(const ModelGeometry& g, int64_t k, int64_t T, void* base, GlmChunkBuffers& c);

/// The `t`-th token's view of a layer's buffers: `out_b`/`out_mb`/`out_bb` are copies of the shared scratch with
/// the fields above redirected into the chunk's row `t`, which is what lets one unchanged `glm_block_layer_pre`
/// serve a whole chunk.  `out_bb.parts`-style fields are left alone; the caller passes `c.parts + t*k*n_embd` to
/// `glm_block_layer_post` itself.
void glm_chunk_view(const GlmChunkBuffers& c, const GlmBuffers& b, const MoEBuffers& mb, const BlockBuffers& bb,
                    int64_t t, GlmBuffers& out_b, MoEBuffers& out_mb, BlockBuffers& out_bb);

/// The collapse before the head: the MEAN of the `hc` streams, then the caller's `output_norm` + projection.
/// Separate from `lm_head` because the first architecture's head starts from a single-stream residual.
bool glm_head_mix(const WeightTable& tables, const ModelGeometry& g, const BlockBuffers& bb, float* out, void* stream,
                  std::string& err);

}  // namespace strata::core
