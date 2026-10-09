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
//   * the SwiGLU clamp in the GGUF is **NOT** inert: `swiglu_clamp_exp` is 10.0 on every layer and both
//     oracles apply it (`ggml_swiglu_clamp`); see `glm_elt.cu`, which used to say the opposite.
#pragma once

#include "strata/core/glm_experts.hpp"   // GlmPoolFn, for the draft block's expert hook
#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/native_head.hpp"   // the draft block's output projection rides the same head
#include "strata/core/weights.hpp"

#include "strata/kernels/glm.hpp"

#include <cstdint>
#include <string>
#include <vector>

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
    /// **THE TWO BAND LOOPS RUN ONE HEAD PER GROUP, AND THESE ARE WHAT MAKES THAT POSSIBLE.**  A band is the
    /// same weight rows for every token, so the projection wants `ncols = ntok` - but `project_rows` takes its
    /// columns contiguous and `w` apart, and in token-major `qfull`/`kqv` the same head two tokens apart is
    /// `n_head * w` away.  So the group is transposed head-major into `heads_in`, projected one band at a time
    /// into `band_out` (also head-major, the same `[h][t][w]` shape), and transposed back.
    ///
    /// Both are `n_head * max(mla_head_dim, kv_lora_rank)` floats a token - the WIDER of the two bands, because
    /// one buffer serves both loops and the absorption's band is 512 rows wide where the de-absorption's is 256.
    ///
    /// **WHAT IT COSTS AND WHAT IT SAVES**, on a 128-token chunk of the 4-way split (`UD-IQ4_XS`): the band
    /// GEMVs and the `native_quantize_q8_1` each one issued were 180,224 pairs - together 360,448 of the
    /// chunk's 434,144 kernel launches, 83.0%, at 2.18 us and 1.21 us a call (nsys, before).  They are 22,528
    /// calls now, one `ncols = 8` GEMV and one `ncols = 8` quantize each, so 45,056 launches.  The census weighs
    /// both sides of it: one stage's 128-token chunk went from 51,216 calls to 8,208 over the same 65,664
    /// columns, and a band tensor of one MLA layer - 1,088 MiB in 8,192 calls at one column a call before - does
    /// not reach its top fourteen any more.  The group width is the whole of the difference.  Against that, each
    /// transpose moves `n_head * ntok * w` floats in one launch - for a group of eight, 512 KiB in the
    /// absorption and 1 MiB in the de-absorption - and there are 2 of them a layer: 16 groups x 11 MLA layers x
    /// 2 = 352 launches for the same chunk, in place of the 360,448 that are gone.
    float* heads_in = nullptr;         ///< `[h][t][w]`: the group, head-major, before the band loop
    float* band_out = nullptr;         ///< `[h][t][w]`: the band loop's result, before it goes back
    float* qabs = nullptr;             ///< n_head * kv_lora_rank: the ABSORBED query (Qcur).  `--dsa` only:
                                       ///< the dense path hands `band_out` to the kernel and never lands here
    float* kv_cmpr = nullptr;          ///< kv_lora_rank: the latent that goes into the cache AND is V
    float* kqv = nullptr;              ///< n_head * kv_lora_rank: the attention output, pre-de-absorption.
                                       ///< `--dsa` only: the dense path writes `heads_in` from the kernel
    float* head_out = nullptr;         ///< n_head * mla_head_dim: `wv_b @ kqv`

    // ---- the dense FFN's two projections, and nothing else ----
    //
    // The FFN's HIDDEN (`silu(gate) * up`, `ffn width`) needs no images of its own: it is the input to
    // `ffn_down`, which is a wide projection, so it uses `wide_*` like the KDA output does.  The sublayer's
    // RESULT has no buffer here at all - a dense layer writes it straight into `bb.block_out`, which is the
    // `n_embd` scratch the block already owns and the pointer `hc_post` is handed.
    float* ffn_gate = nullptr;         ///< ffn width
    float* ffn_up = nullptr;           ///< ffn width

    // ---- the DSA indexer's scratch ----
    //
    // `key`, `gate`, `iq` and `iw` are the four projections' outputs and are per token like everything above; the
    // first two are then copied a row at a time into the layer's pool-in-progress, which is why they are not the
    // same buffer as `idx_partial_k`/`_g` in the state.
    //
    // **`score` AND `cells` ARE THE ONE EXCEPTION: ONCE PER CARVE, NOT ONCE PER TOKEN.**  They belong to the
    // SELECTION, which runs one query at a time even inside a chunk - a pool's visibility is a function of that
    // query's own position - so one row is live at a time and a chunk overwrites it `ntok` times.  Scaling them
    // would be `ntok * max_cells / kpool` floats and `ntok * 2051` ints: 64 MB and 33 MB at a 16K context and a
    // 4096-token chunk.  `GlmBuffers`' own note above is the rule; this is the exception and it is stated there
    // too.
    float* idx_key = nullptr;          ///< key_dim per token: `indexer.attn_k`, normed
    float* idx_gate = nullptr;         ///< key_dim per token: `indexer_compressor_gate`
    float* idx_iq = nullptr;           ///< key_dim * idx_heads per token: `indexer.attn_q_b @ qr`
    float* idx_iw = nullptr;           ///< idx_heads per token: `indexer.proj @ cur`, prescaled
    float* idx_score = nullptr;        ///< max_cells / kpool, ONE query's row
    int32_t* idx_cells = nullptr;      ///< glm_dsa_n_sel(): 2051 here, ONE query's row
    int32_t* idx_pos = nullptr;        ///< ntok absolute positions, what `glm_dsa_select` masks by
};

/// `ntok` is how many tokens the carve is for: every field above is per token and the whole carve scales with
/// it, so `ntok == 1` is exactly the decode path's scratch and `ntok == T` is what a chunk needs.
/// `max_cells` sizes the DSA selection's three once-per-carve rows and nothing else; see the note on
/// `glm_buffers_bytes` in the .cpp, which is where the exception to "everything here is per token" is argued.
uint64_t glm_buffers_bytes(const ModelGeometry& g, int64_t ntok, int64_t max_cells);
uint64_t glm_buffers_init(const ModelGeometry& g, int64_t ntok, int64_t max_cells, void* base, GlmBuffers& b);

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

    // ---- the DSA indexer (MLA layers only; see `kernels/glm_dsa.hpp`) ----
    //
    // **THREE ARRAYS, AND THE FIRST TWO HOLD ONLY THE POOL IN PROGRESS.**  A cell's indexer key and gate are read
    // exactly once, by the pool that contains them, so the engine keeps `kpool` cells of each and not the
    // `max_cells` the reference's cache holds - a token writes its row into slot `pos % kpool` and the pool runs
    // when that slot is the last one.  The alternative is 16 MB a layer at a 16K context for bytes nothing reads.
    //
    // `pooled` is the part that IS history: every completed pool, `key_dim` wide, indexed by pool number, and the
    // only thing a query's selection reads.  It is `max_cells / kpool` columns.
    //
    // All three are carved even when `glm_dsa_enabled()` is false.  The session's size must not depend on a switch
    // that is set from the command line, or a run that enables it late would write past a carve sized without it.
    float* idx_partial_k = nullptr;   ///< key_dim * kpool: the current pool's keys, member fastest
    float* idx_partial_g = nullptr;   ///< key_dim * kpool: the same for the compressor gate
    float* idx_pooled = nullptr;      ///< key_dim * (max_cells / kpool)
};

/// Floats the KDA state needs (the delta state plus the conv history), zero when the arch has no KDA.
uint64_t glm_kda_state_floats(const ModelGeometry& g);
/// Bytes ONE MLA layer's latent cache needs for `max_cells` tokens, zero when the arch has no MLA.
uint64_t glm_mla_cache_bytes(const ModelGeometry& g, int64_t max_cells);
/// Bytes ONE MLA layer's WHOLE persistent state (`glm_mla_cache_bytes` rounded up plus the DSA indexer's).
///
/// **THE MTP BLOCK IS AN MLA LAYER THAT IS NOT A TRUNK LAYER, AND THE LAYER-NUMBER PREDICATE CANNOT SAY SO.**
/// `glm_is_kda_layer` is `!(layer % qsa_interval == qsa_interval - 1)`, and the draft block sits at
/// `n_layers` = 45, which is 1 mod 4 - so the predicate calls a block that carries `attn_q_a.weight` and no
/// `ssm_*` at all a GDN layer and sizes it at 12 KB instead of 16 MB.  Sizing therefore comes from THIS
/// function, which asks nothing about the layer, and `glm_state_init`'s MLA branch and the MTP both call it.
uint64_t glm_mla_state_bytes(const ModelGeometry& g, int64_t max_cells);
/// Points an MLA layer's state at `base`; returns the bytes it consumed (== `glm_mla_state_bytes`).
uint64_t glm_mla_state_init(const ModelGeometry& g, int64_t max_cells, void* base, GlmLayerState& st);
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

// ================================ where a chunk's `pre` time goes ================================

/// **THE SECTIONS OF `pre`, TIMED ON THE CARD.**
///
/// `STRATA_GLM_PREFILL_TIME=1` makes `glm_block_layer_pre` and `glm_block_layer_post` record a CUDA event at each
/// section boundary, and `glm_pre_sections_flush` reads the gaps once the caller has synchronized.  Events rather
/// than wall-clock because every section only ENQUEUES: a `printf` around the calls would report the launch cost
/// and nothing else.
///
/// It exists to answer one question that the arms outside the engine cannot.  The chunk's own line says how much
/// of a prefill is the card and how much is the CPU pool; it cannot say whether the card's share is the quantized
/// weight reads (the thing `ntok` batching fixes) or the per-token elementwise work around them (which it does
/// not).  The two want different changes, and this is the measurement that tells them apart.
///
/// The accumulator is a file-scope static, so a report covers every layer and token since the last reset - the
/// chunk.  Reset it before the chunk and report after; nothing is thread-safe and nothing needs to be, because a
/// session's layers run one at a time.
void glm_pre_sections_reset();
/// Reads the events recorded since the last call and adds them to the totals.  Call after a stream sync.
void glm_pre_sections_flush();
/// Prints the totals and resets them.  `tokens` and `layers` are what the totals cover.
void glm_pre_sections_report(int64_t tokens, int64_t layers);

/// The whole block, for a caller that already has `parts`.
bool glm_block_layer(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t pos,
                     int32_t pos_base, const GlmBuffers& b, const GlmLayerState& st, const MoEBuffers& mb, int64_t k,
                     const BlockBuffers& bb, const float* parts, void* stream, std::string& err,
                     const Doorbell* db);

// ================================ the MTP (draft) block ================================

/// **THE BLOCK PAST THE TRUNK, WHICH IS `blk.45` ON THE SHIPPED 46-BLOCK ARTIFACT.**  `nextn_predict_layers`
/// is 1 there, so `n_layers` is 45 and this block's own index is 45 - the pack names its tensors
/// `blk.45.*` like any other block's, and the engine reads them through the same `WeightTable`.
///
/// It is an ordinary MLA + MoE block with NO hyper-connections and four extra tensors.  Its arithmetic, from
/// the reference's `build_glm5next_mtp` (`ik_llama.cpp/src/graphs/build_glm5next.cpp:381-500`):
///
///     cur    = eh_proj(cat(enorm(emb(x) * clamp(pos, 0, 1)), hnorm(h)))
///     inpSA  = cur
///     cur    = <the trunk's own MLA attention over this block's own cache>
///     ffn_inp= cur + inpSA
///     cur    = moe(rms(ffn_norm(ffn_inp))) + shexp(rms(ffn_norm(ffn_inp))) + ffn_inp
///     out    = rms(shared_head_norm(cur)) -> the model's own output head
///
/// The three traps are in the first two lines and in what is NOT there: the position mask zeroes row 0's
/// embedding BEFORE `enorm` (there is no next token for the first cell); `enorm(emb)` occupies the FIRST half
/// of the 8192-wide concat, so swapping the halves is a shape-legal wrong answer; and the block has no
/// `hc_*` tensors, so `hc_read`/`hc_write` must not be called on it even though every trunk layer has them.
///
/// `token` is the token whose embedding is folded in, `hidden` the trunk's post-`output_norm` state at the
/// position the draft is made FROM (the reference's `result_mtp_embd`, i.e. exactly what `glm_head_mix`
/// leaves in `bb.mixed`), and `pos` the absolute position the block's own cache row is written at.  The
/// draft logits land in `logits`, which the caller argmaxes - the reference's draft sampler has no RNG.
struct GlmMtpState {
    /// The block's OWN MLA state: one fp16 latent row per cell, plus the indexer's arrays.  A second cache,
    /// not a view of a trunk layer's - the draft block attends over the draft block's own cells.
    GlmLayerState attn;
    /// The block's input assembly, `n_embd` floats each.  **THE BLOCK NEEDS SCRATCH OF ITS OWN AND CANNOT
    /// BORROW `GlmBuffers` FOR ALL OF IT**: `cat` is `2 * n_embd` (8192) and the only buffers that wide are
    /// `wide_*`, which `mla_layer` and `ffn3` are about to overwrite, and `inp` has to survive the attention
    /// (`inpSA` is added after it).  Five `n_embd` vectors is 80 KB, which is cheaper than the reasoning.
    float* emb = nullptr;      ///< n_embd: `enorm(emb(x) * clamp(pos,0,1))`
    float* hstate = nullptr;   ///< n_embd: `hnorm(h)`
    float* cat = nullptr;      ///< 2 * n_embd: the concat `eh_proj` reads
    float* inp = nullptr;      ///< n_embd: `cur` / `inpSA` / `ffn_inp`
    int64_t max_cells = 0;
    /// Rows of `attn.mla_cache` this block has written since the sequence started, and so the row the next step
    /// goes into.  **THE DRAFT BLOCK'S CACHE IS COMPACTED: ONE ROW PER DRAFT STEP, NOT ONE PER POSITION.**  The
    /// block is fed at the positions its caller picks, so its rows are not its positions and cannot be addressed
    /// by one - a first draft at position 5 writes row 0 and attends to row 0 alone.  That is the reference's
    /// semantics too: its draft context is a fresh context, its cells are allocated in write order, and its mask
    /// (`cells[i].pos <= pos`, `has_seq_id` on an unallocated cell false) leaves every unwritten cell masked
    /// out.  It is reset with the rest of the block's state at a sequence boundary - `session_zero`.
    int64_t n_written = 0;
    /// The host staging the expert pool is called with - the same three copies `session_token` makes, kept
    /// here so a draft step allocates nothing.
    std::vector<float> h_x, h_w, h_out;
    std::vector<int32_t> h_ids;
};

/// Bytes the MTP block's state needs: its MLA cache and indexer, plus the input assembly.  Zero when the
/// model declares no block past the trunk (`n_nextn` 0), so an arch without one carves nothing.
uint64_t glm_mtp_state_bytes(const ModelGeometry& g, int64_t max_cells);
/// Points `st` at `base` and returns the bytes it consumed (== `glm_mtp_state_bytes`).  The MLA half comes
/// from `glm_mla_state_init` so the draft block's cache cannot drift from a trunk MLA layer's.
uint64_t glm_mtp_state_init(const ModelGeometry& g, int64_t max_cells, void* base, GlmMtpState& st);

/// ONE POSITION OF THE DRAFT BLOCK.  Runs the block and leaves the draft logits in `logits` (`n_vocab` f32,
/// DEVICE).  `hidden` is DEVICE, `n_embd` floats.
///
/// The expert hook is `session_token`'s `GlmPoolFn`, called with `layer = g.n_layers` - the draft block is a
/// block of this model and its experts are routed, read and combined exactly like a trunk MoE layer's, so a
/// null `pool` on a model whose draft block has experts is refused rather than run.
///
/// `head` is the run's `NativeHead` or null, and it is NOT optional politeness: `--native` makes
/// `output.weight` a non-resident row of the pack (`skip.insert("output.weight")`), served only by this class,
/// so a draft step that went straight to `lm_head_project` would be handed a `WeightRef` with no planes at all
/// and refuse.  The trunk's `run_head` branches on exactly this, and the draft block's last projection IS the
/// trunk's head - the block owns no head of its own, only `nextn.shared_head_norm` in front of it.
bool glm_mtp_step(const WeightTable& tables, const ModelGeometry& g, const GlmBuffers& b, const MoEBuffers& mb,
                  const BlockBuffers& bb, GlmMtpState& st, int64_t token, const float* hidden, int64_t pos,
                  int64_t k, GlmPoolFn pool, void* pool_user, const NativeHead* head, const float* logits,
                  void* stream, std::string& err);

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
    float* logits = nullptr;     ///< T rows of n_expert, the router's raw output before the top-k
    int64_t n_embd = 0, hc = 0, k = 0, n_expert = 0, T = 0;
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

/// **THE WIDEST TOKEN GROUP ONE `glm_block_layer_pre` CALL MAY BE HANDED.**  The bound was the quantized
/// projections': `native_mmvq` takes at most `NATIVE_MMVQ_MAX_NCOLS` columns, so a wider group would split the
/// very weight read the group exists to amortize.  A native tensor past that cap now goes through
/// `prefill::mmq::dense` - one GEMM over the whole group, off the same bytes - so the cap no longer binds a
/// native one; it still binds a canonical tensor, and `glm_group_max()` below is what a session actually carves.
constexpr int64_t GLM_MAX_NTOK = 8;

/// The ceiling on `STRATA_GLM_GROUP_MAX`.  NOT a free choice: `glm_heads_major` launches a grid whose y is
/// `n_head * T` and refuses past 65535, and 64 heads put the wall at 1023 - so 512 is the largest power of two
/// that is safe, and a wider group would return from that kernel having written nothing.
constexpr int64_t kGlmGroupMaxNtok = 512;

/// **THE GROUP WIDTH THIS SESSION CARVES FOR**: `GLM_MAX_NTOK` unless `STRATA_GLM_GROUP_MAX` names a wider one.
///
/// The group width is what a projection's weight read is divided by, and the read is what a chunk's `pre`
/// costs: a 4,096-token chunk at the default eight enters `pre` 512 times and reads each dense weight matrix
/// 512 times, which is why the section reports 722 GB of reads on a stage holding 1.4 GB.  Eight is where the
/// MMVQ path stops, so the default is unchanged and the wider arms are opt-in - and every carve site asks this
/// one function, so the group carve is always the width the loop is about to use.
int64_t glm_group_max();

/// The same as `glm_chunk_view`, for a GROUP of `ntok` tokens starting at `t0`: `b` is the session's group
/// carve (`glm_buffers_init(g, GLM_MAX_NTOK, ...)`), which already holds every projection's scratch at a group
/// width, and only the fields that outlive the call - the residual, the normed FFN input, the mHC maps, the
/// router's two rows - are redirected into the chunk's rows `t0 .. t0 + ntok - 1`.
///
/// **THE CHUNK'S ROWS ARE WHAT MAKES A GROUP SAFE TO REUSE ONE CARVE FOR.**  Two consecutive groups at the
/// same layer write the same scratch, so nothing a later group needs may live there: `c.cur`, `c.post`,
/// `c.comb`, `c.shared`, `c.ids`, `c.weights` and `c.logits` are per token and in the chunk arena, and the
/// projection scratch is only ever read by the call that wrote it.
void glm_chunk_group_view(const GlmChunkBuffers& c, const GlmBuffers& group, const MoEBuffers& mb,
                          const BlockBuffers& bb, int64_t t0, int64_t ntok, GlmBuffers& out_b, MoEBuffers& out_mb,
                          BlockBuffers& out_bb);

/// The collapse before the head: the MEAN of the `hc` streams, then the caller's `output_norm` + projection.
/// Separate from `lm_head` because the first architecture's head starts from a single-stream residual.
bool glm_head_mix(const WeightTable& tables, const ModelGeometry& g, const BlockBuffers& bb, float* out, void* stream,
                  std::string& err);

}  // namespace strata::core
