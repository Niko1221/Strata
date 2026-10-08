// src/core/glm_layer.cpp - ONE glm5-next block, composed.  The second block scaffold.
//
// The header says WHY this is a second scaffold rather than flags on the first one; this file is the order and
// the buffers.  One block, from `bb.R` back to `bb.R`:
//
//     hc_pre(attn) -> hc_mix -> attn_norm -> <KDA | MLA> -> hc_post(attn)
//     hc_pre(ffn)  -> hc_mix -> ffn_norm  -> <dense FFN | MoE route> -> hc_post(ffn)
//
// **`hc_post` IS THE ONLY WRITE TO THE RESIDUAL.**  There is no `x = x + sublayer(x)` anywhere: the reference
// passes `add_residual = false` to the attention and `add_input = false` to the MoE, because `hc_post` already
// folds the sublayer's output back into all four streams through `post` and `comb`.  A port that adds a plain
// residual anywhere still runs, still talks, and is not this model.
//
// WHAT IS HERE, AND WHAT IS NOT.  mHC, KDA, MLA, the dense SwiGLU FFN, the sigmoid router, the plain-add
// shared expert, and the head collapse.  What is NOT is the sparse indexer: without it the MLA layers run
// DENSE over the whole cache, which is the reference's own behaviour when `--dsa` is absent - so this is the
// port's baseline, not a shortcut, and the indexer is an optimisation that has to reproduce it.
//
// The one path that still refuses by name is the host expert pool: `session_token` has no hook for it, so a
// MoE layer says so rather than silently adding only the shared expert - which would be a fluent wrong answer.
//
// A refusal that names what is missing is worth more than a plausible number: every remaining difference
// between this engine and the oracle is supposed to be one of a known, countable set.
#include "strata/core/glm_layer.hpp"

#include "gemv_util.hpp"

#include "strata/core/layer.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/glm.hpp"
#include "strata/kernels/glm_dsa.hpp"
#include "strata/kernels/quantize_act.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
namespace {

using namespace strata::core::gemv;

/// 16-byte aligned, like every other arena in this engine, so a carved region can hold a `float4`.
struct Arena {
    uint8_t* p = nullptr;
    uint64_t used = 0;

    void* take(uint64_t bytes) {
        const uint64_t at = (used + 15) & ~(uint64_t) 15;
        void* r = p + at;
        used = at + bytes;
        return r;
    }
};

inline uint64_t align16(uint64_t v) { return (v + 15) & ~(uint64_t) 15; }

/// `n_v` is the KDA width: 64 heads of 128.  Zero on an arch without KDA, and every KDA buffer is then zero
/// bytes rather than a wrong size.
int64_t kda_n_v(const ModelGeometry& g) { return g.kda_head_dim * g.n_head; }

/// The FFN width the buffers must hold: the dense-lead width when the arch has one, else the expert width.
/// Both are carved because a layer does not know which it is until it is asked, and the two differ by 6x.
int64_t ffn_width(const ModelGeometry& g) { return std::max(g.n_ff_dense, g.n_ff); }

/// The widest single activation that is NOT `cur`.  Three roles read it and they are not the same width:
/// KDA's out projection reads `n_v` (8192), the FFN's down projection reads `ffn_width` (12288), and MLA's
/// output projection reads the DE-ABSORBED heads, `n_head * mla_head_dim` (16384) - the widest of the three
/// and the one a first cut of this file missed, which put `head_out` 4096 floats past the end of the carve.
/// One image pair serves all three, told the real length at each call.
int64_t wide_width(const ModelGeometry& g) {
    return std::max(kda_n_v(g), std::max(ffn_width(g), g.n_head * g.mla_head_dim));
}

/// The width of ONE MLA head's de-absorbed output, and of the query before absorption: 256 at this geometry.
/// Named because the per-head loops index by it and a bare `g.mla_head_dim` next to `g.kv_lora_rank` is
/// exactly where a 256 and a 512 get swapped.
int64_t mla_head_width(const ModelGeometry& g) { return g.mla_head_dim; }

/// The width of the widest thing the per-head image pair has to hold.  The two per-head loops quantize
/// DIFFERENT widths - the absorption one a head's `mla_head_dim` query, the de-absorption one a head's
/// `kv_lora_rank` latent - and `kv_lora_rank` is the larger of the two (512 against 256), so the pair is
/// sized once for the larger rather than twice for each.  Zero on an arch with no MLA, which makes every MLA
/// buffer zero bytes rather than a wrong size.
int64_t mla_band_width(const ModelGeometry& g) { return std::max(g.mla_head_dim, g.kv_lora_rank); }

const WeightRef* req(const LayerView& v, const char* suffix, std::string& err) {
    const WeightRef* w = v.get(suffix);
    if (w == nullptr) err = v.name(suffix) + " is missing";
    return w;
}

/// ONE PROJECTION, for every form a weight can be in.  Three cases, and the reason all three are here rather
/// than at each call site is that a glm5-next layer mixes them within a single block:
///
///   * quantized, or served straight from the GGUF -> `gemv_quantized`, which owns both of those already and
///     picks the activation image from the TENSOR (never from the role).
///   * plain f32 in the arena -> `glm_f32_gemv`.  glm5-next's router and the KDA conv filters are f32 in the
///     file where qwen4exp's equivalents are BF16, so this case exists on this arch and not on the other.
///   * 16-bit (bf16) in the arena -> `project_bf16`, which is what the engine already does for BF16 weights.
///
/// Reading an f32 weight as bf16 (or the reverse) produces output of the right SHAPE and roughly the right
/// magnitude, which is why the byte count is checked rather than the caller's belief.
///
/// `ntok` is the CHUNKED-PREFILL column count: how many tokens this one call carries, 1..8.  Every array here
/// is then `ntok` consecutive columns with the width fastest and no padding - `x_f32` at stride `n_in`, `y` at
/// stride `n_out`, and the two images at their natural per-column strides (see `gemv_quantized`).  The point is
/// not fewer instructions: it is that the 7.5 GiB of native weights is read ONCE per `ntok` tokens instead of
/// once per token, which is the whole of the GPU half's time at 88% of the card's bandwidth.
bool project(const WeightRef& w, const std::string& name, const float* x_f32, const uint8_t* x80,
             const uint8_t* xq8k, const uint16_t* x_bf16, float* y, int64_t n_in, int64_t n_out, int64_t ntok,
             void* stream, std::string& err) {
    if (w.quantized() || w.native_data != nullptr) {
        kernels::SForm f;
        Planes p;
        if (w.native_data == nullptr) {
            if (!sform_of(w, f, name, err)) return false;
            if (!plane_ptrs(w, name, p, err)) return false;
        }
        return gemv_quantized(w, p, f, x80, xq8k, y, n_in, n_out, name, stream, err, x_f32, false, (int) ntok);
    }
    const uint64_t elems = (uint64_t) w.elements;
    if (w.bytes == elems * 4) {
        if (x_f32 == nullptr) { err = name + ": an f32 weight needs the f32 activation"; return false; }
        // `glm_f32_gemv` is one column; a batch is the same call walked along the columns, which is exact for
        // any `ntok` because nothing is shared between them.
        for (int64_t c = 0; c < ntok; ++c) {
            kernels::glm_f32_gemv(x_f32 + c * n_in, (const float*) w.data, y + c * n_out, n_in, n_out, stream);
        }
        return true;
    }
    if (w.bytes == elems * 2) {
        if (x_f32 == nullptr || x_bf16 == nullptr) { err = name + ": a 16-bit weight needs f32 and bf16 input"; return false; }
        for (int64_t c = 0; c < ntok; ++c) {
            project_bf16(x_f32 + c * n_in, x_bf16 + c * n_in, (const uint16_t*) w.data, y + c * n_out, n_in,
                         n_out, true, stream);
        }
        return true;
    }
    char buf[256];
    std::snprintf(buf, sizeof buf, "%s: %llu B for %llu elements is neither 4 nor 2 bytes per element",
                  name.c_str(), (unsigned long long) w.bytes, (unsigned long long) elems);
    err = buf;
    return false;
}

/// Both quantized images of one activation, so `project` can pick without knowing which weight will read it.
/// `n` must be a multiple of 256 for the Q8_K image and of 32 for the Q8_0 one - which every width this file
/// passes is (4096, 8192, 12288), and the one that is not (128, the `ssm_f_a`/`ssm_g_a` tail) goes through
/// `quantize_q8_0` alone.
void quantize_both(const float* x, uint8_t* x_q8k, uint8_t* x_q8_0, int64_t n, void* stream) {
    kernels::quantize_q8_K(x, x_q8k, n, stream);
    kernels::quantize_q8_0(x, x_q8_0, n, stream);
}

/// One scratch pair of images for the 128-wide KDA tail vectors.  They are too narrow for Q8_K (128 is not a
/// multiple of 256), which is not a limitation here: `ssm_f_b`/`ssm_g_b` are Q8_0 in the file.
constexpr int64_t TAIL = 128;

}  // namespace

// ================================ sizing ================================

namespace {

/// The pools a query can select from, which is all of them once the cache is full: `max_cells / kpool`, rounded
/// UP because a cache whose length is not a multiple of the pool size still carries the last, partial pool's
/// index.  The partial pool itself can never be selected - `glm_dsa_select` only ever reads pools whose last
/// member exists - but `pooled` is addressed by pool number, so the column has to exist.
uint64_t glm_dsa_pools(const ModelGeometry& g, int64_t max_cells) {
    if (g.idx_kpool <= 0 || max_cells <= 0) return 0;
    return (uint64_t) ((max_cells + g.idx_kpool - 1) / g.idx_kpool);
}

/// One query's `score` row: `key_dim` floats for each pool it might have to score.
uint64_t glm_dsa_score_bytes(const ModelGeometry& g, int64_t max_cells) {
    return glm_dsa_pools(g, max_cells) * (uint64_t) g.idx_key_dim * 4;
}

/// The ints one `cells` row needs (`glm_dsa_n_sel`), 4 bytes each.  A named helper rather than the call spelled
/// at each site, because `glm_buffers_bytes` and `glm_buffers_init` are two lists that must agree byte for byte
/// and this is the term most easily written twice with two different readings of `select_tail`.
uint64_t glm_dsa_cells_bytes(const ModelGeometry& g) {
    return (uint64_t) kernels::glm_dsa_n_sel(g.idx_top_k, g.idx_kpool, (int) g.idx_select_tail) * 4;
}

/// ONE MLA LAYER'S indexer state: the pool in progress (key and gate) plus every completed pool.  See
/// `GlmLayerState` for why the per-cell key history is not kept.
uint64_t glm_dsa_state_bytes(const ModelGeometry& g, int64_t max_cells) {
    if (g.idx_kpool <= 0 || g.idx_key_dim <= 0) return 0;
    const uint64_t partial = (uint64_t) g.idx_key_dim * (uint64_t) g.idx_kpool * 4;
    return 2 * align16(partial) + glm_dsa_pools(g, max_cells) * (uint64_t) g.idx_key_dim * 4;
}

}  // namespace

/// **EVERY FIELD IN `GlmBuffers` IS PER TOKEN, SO THE WHOLE CARVE SCALES WITH `ntok`.**  There is no field in
/// the list below that is shared across a chunk: the mHC maps, the normed input, the wide image pair, the KDA
/// streams, the MLA latent and head stacks and the FFN pair are each one token's worth of scratch, written by a
/// projection whose input in a chunk is `ntok` columns.  `glm_buffers_bytes(g, 1)` is what the decode path
/// carves and it is byte for byte the shape that was there before the chunk existed.
///
/// **THE SLOT IS `align16(ntok * one token's field)`, SO A FIELD'S TOKENS ARE CONTIGUOUS.**  The earlier
/// spelling was `ntok * align16(field)`, which pads EACH token's slot to 16 bytes; the two agree at `ntok == 1`
/// and diverge for any field whose one-token byte count is not a multiple of 16 - `pre` and `post` at
/// `hc * 4 = 12`, `comb` at 36, `idx_pos` at 4.  That is exactly the layout the kernels cannot read: every
/// kernel that takes an `ntok` here reads column `j` at `base + j * width` and nothing else (`project` says so
/// in its own comment), so a padded slot would have it read token `j`'s neighbour.  Contiguous is what the
/// callers already assume; this makes the carve agree with them.
///
/// The one place the padded rule would have been safe is the first token of every field, which is why the
/// decode path cannot tell the two apart - and why this had to be fixed before anything could batch a chunk.
///
/// **THE DSA SELECTION'S THREE ROWS ARE THE EXCEPTION: THEY ARE ONCE PER CARVE, NOT ONCE PER TOKEN.**  `idx_iq`
/// and `idx_iw` are projections and scale with `ntok` like everything else; `idx_score`, `idx_cells` and `idx_pos`
/// belong to the selection, which runs one query at a time even inside a chunk (a pool's visibility is a function
/// of that query's own position), so one row is live at a time.  Scaling them would be `ntok * max_cells / kpool`
/// floats - 64 MB at a 16K context and a 4096-token chunk - for storage overwritten once per query.
///
/// `max_cells` is a parameter for those three and for nothing else: every other field is a function of the
/// geometry alone, which is why this function did not need it before.
uint64_t glm_buffers_bytes(const ModelGeometry& g, int64_t ntok, int64_t max_cells) {
    if (ntok < 1) return 0;
    const int64_t n = g.n_embd;
    const int64_t hc = g.hc;
    const int64_t nv = kda_n_v(g);
    const int64_t fw = ffn_width(g);
    const int64_t ww = wide_width(g);

    const uint64_t parts[] = {
        (uint64_t) hc * n * 4,                                  // res_scratch
        (uint64_t) hc * n * 2,                                  // normed_bf16
        (uint64_t) hc * n * 4,                                  // normed_f32
        (uint64_t) g.hc_mix * 4,                                // mixes
        (uint64_t) hc * 4,                                      // pre
        (uint64_t) hc * 4,                                      // post
        (uint64_t) hc * hc * 4,                                 // comb
        q8k_bytes(hc * n),                                      // normed_q8k
        (uint64_t) (hc * n / 32) * 34,                          // normed_q8_0
        (uint64_t) n * 4,                                       // cur
        q8k_bytes(n),                                           // cur_q8k
        (uint64_t) (n / 32) * 34,                               // cur_q8_0
        (uint64_t) n * 2,                                       // cur_bf16
        q8k_bytes(ww),                                          // wide_q8k
        (uint64_t) (ww / 32) * 34,                              // wide_q8_0
        (uint64_t) ww * 2,                                      // wide_bf16
        (uint64_t) 3 * nv * 4,                                  // qkv
        (uint64_t) nv * 4,                                      // gate
        (uint64_t) g.n_head * 4,                                // beta
        (uint64_t) nv * 4,                                      // raw
        (uint64_t) nv * 4,                                      // z
        (uint64_t) nv * 4,                                      // o
        (uint64_t) n * 4,                                       // attn_out
        (uint64_t) 2 * TAIL * 4,                                // tail
        (uint64_t) (TAIL / 32) * 34,                            // tail_q8_0
        (uint64_t) g.q_lora_rank * 4,                           // qr
        (uint64_t) g.n_head * g.mla_head_dim * 4,               // qfull
        q8k_bytes(g.n_head * mla_band_width(g)),                // heads_q8k
        (uint64_t) (g.n_head * mla_band_width(g) / 32) * 34,    // heads_q8_0
        (uint64_t) g.n_head * mla_band_width(g) * 4,            // heads_in
        (uint64_t) g.n_head * mla_band_width(g) * 4,            // band_out
        (uint64_t) g.n_head * g.kv_lora_rank * 4,               // qabs
        (uint64_t) g.kv_lora_rank * 4,                          // kv_cmpr
        (uint64_t) g.n_head * g.kv_lora_rank * 4,               // kqv
        (uint64_t) g.n_head * g.mla_head_dim * 4,               // head_out
        (uint64_t) fw * 4,                                      // ffn_gate
        (uint64_t) fw * 4,                                      // ffn_up
        (uint64_t) g.idx_key_dim * 4,                           // idx_key
        (uint64_t) g.idx_key_dim * 4,                           // idx_gate
        (uint64_t) g.idx_key_dim * g.idx_q_heads * 4,           // idx_iq
        (uint64_t) g.idx_q_heads * 4,                           // idx_iw
        (uint64_t) 4,                                           // idx_pos
    };
    uint64_t total = 0;
    for (uint64_t v : parts) total += align16((uint64_t) ntok * v);
    // THE THREE THAT ARE ONCE PER CARVE - see the note above and `GlmBuffers`.
    const uint64_t once[] = {
        glm_dsa_score_bytes(g, max_cells),                        // idx_score
        glm_dsa_cells_bytes(g),                                   // idx_cells
    };
    for (uint64_t v : once) total += align16(v);
    return total;
}

uint64_t glm_buffers_init(const ModelGeometry& g, int64_t ntok, int64_t max_cells, void* base, GlmBuffers& b) {
    const int64_t n = g.n_embd;
    const int64_t hc = g.hc;
    const int64_t nv = kda_n_v(g);
    const int64_t fw = ffn_width(g);
    const int64_t ww = wide_width(g);

    Arena a{(uint8_t*) base, 0};
    // The SAME slot rule `glm_buffers_bytes` uses, applied in one place so the two lists cannot disagree.
    const auto take = [&](uint64_t one_token) -> void* {
        return a.take(align16((uint64_t) ntok * one_token));
    };
    b.res_scratch = (float*) take((uint64_t) hc * n * 4);
    b.normed_bf16 = (uint16_t*) take((uint64_t) hc * n * 2);
    b.normed_f32 = (float*) take((uint64_t) hc * n * 4);
    b.mixes = (float*) take((uint64_t) g.hc_mix * 4);
    b.pre = (float*) take((uint64_t) hc * 4);
    b.post = (float*) take((uint64_t) hc * 4);
    b.comb = (float*) take((uint64_t) hc * hc * 4);
    b.normed_q8k = (uint8_t*) take(q8k_bytes(hc * n));
    b.normed_q8_0 = (uint8_t*) take((uint64_t) (hc * n / 32) * 34);
    b.cur = (float*) take((uint64_t) n * 4);
    b.cur_q8k = (uint8_t*) take(q8k_bytes(n));
    b.cur_q8_0 = (uint8_t*) take((uint64_t) (n / 32) * 34);
    b.cur_bf16 = (uint16_t*) take((uint64_t) n * 2);
    b.wide_q8k = (uint8_t*) take(q8k_bytes(ww));
    b.wide_q8_0 = (uint8_t*) take((uint64_t) (ww / 32) * 34);
    b.wide_bf16 = (uint16_t*) take((uint64_t) ww * 2);
    b.qkv = (float*) take((uint64_t) 3 * nv * 4);
    b.gate = (float*) take((uint64_t) nv * 4);
    b.beta = (float*) take((uint64_t) g.n_head * 4);
    b.raw = (float*) take((uint64_t) nv * 4);
    b.z = (float*) take((uint64_t) nv * 4);
    b.o = (float*) take((uint64_t) nv * 4);
    b.attn_out = (float*) take((uint64_t) n * 4);
    b.tail = (float*) take((uint64_t) 2 * TAIL * 4);
    b.tail_q8_0 = (uint8_t*) take((uint64_t) (TAIL / 32) * 34);
    b.qr = (float*) take((uint64_t) g.q_lora_rank * 4);
    b.qfull = (float*) take((uint64_t) g.n_head * g.mla_head_dim * 4);
    b.heads_q8k = (uint8_t*) take(q8k_bytes(g.n_head * mla_band_width(g)));
    b.heads_q8_0 = (uint8_t*) take((uint64_t) (g.n_head * mla_band_width(g) / 32) * 34);
    b.heads_in = (float*) take((uint64_t) g.n_head * mla_band_width(g) * 4);
    b.band_out = (float*) take((uint64_t) g.n_head * mla_band_width(g) * 4);
    b.qabs = (float*) take((uint64_t) g.n_head * g.kv_lora_rank * 4);
    b.kv_cmpr = (float*) take((uint64_t) g.kv_lora_rank * 4);
    b.kqv = (float*) take((uint64_t) g.n_head * g.kv_lora_rank * 4);
    b.head_out = (float*) take((uint64_t) g.n_head * g.mla_head_dim * 4);
    b.ffn_gate = (float*) take((uint64_t) fw * 4);
    b.ffn_up = (float*) take((uint64_t) fw * 4);
    b.idx_key = (float*) take((uint64_t) g.idx_key_dim * 4);
    b.idx_gate = (float*) take((uint64_t) g.idx_key_dim * 4);
    b.idx_iq = (float*) take((uint64_t) g.idx_key_dim * g.idx_q_heads * 4);
    b.idx_iw = (float*) take((uint64_t) g.idx_q_heads * 4);
    b.idx_pos = (int32_t*) take((uint64_t) 4);
    // ONCE PER CARVE, not per token - the last three takes, matching the last three terms of
    // `glm_buffers_bytes`.  `take_once` is `take` without the `ntok`.
    const auto take_once = [&](uint64_t bytes) -> void* { return a.take(align16(bytes)); };
    b.idx_score = (float*) take_once(glm_dsa_score_bytes(g, max_cells));
    b.idx_cells = (int32_t*) take_once(glm_dsa_cells_bytes(g));

    // ---- the aliases a chunk needs, set here so no call site does this arithmetic ----
    b.ntok = ntok;
    // `qkv` holds `ntok` TOKEN BLOCKS of `[q; k; v]` - not one interleaved `[q;k;v]` per token.  At `ntok == 1`
    // the two readings are the same bytes, which is why the decode path is unaffected; at `ntok > 1` the three
    // stream pointers below are what makes each stream's columns contiguous, and the projection that produced
    // them (three separate `ntok x n_v` blocks) is what put them there.
    b.q = b.qkv;
    b.kk = b.qkv + (size_t) ntok * nv;
    b.v = b.qkv + (size_t) 2 * ntok * nv;
    // The conv's token stride.  `n_v` in BOTH cases: the split form at `ld = n_v` is the interleaved form once
    // there is only one token, and `glm_kda_conv_silu3` is documented to be the same kernel either way.
    b.qkv_ld = nv;
    // `tail` is `ntok` blocks of `[g_a(128); f_a(128)]`, and the two halves are separate projections whose
    // inputs must each be `ntok` contiguous columns - so the f half starts `ntok * 128` in, not 128.
    b.tail_f = b.tail + (size_t) ntok * TAIL;
    return a.used;
}

// ================================ chunked prefill ================================

uint64_t glm_chunk_bytes(const ModelGeometry& g, int64_t k, int64_t T) {
    if (T < 1) return 0;
    const uint64_t n = (uint64_t) g.n_embd, hc = (uint64_t) g.hc, kk = (uint64_t) k, tt = (uint64_t) T;
    const uint64_t parts[] = {
        hc * n * 4,        // R
        n * 4,             // cur
        n * 4,             // block_out
        n * 4,             // shared
        kk * n * 4,        // parts
        hc * 4,            // post
        hc * hc * 4,       // comb
        kk * 4,            // weights
        kk * 4,            // ids
        (uint64_t) g.n_expert * 4,   // logits
    };
    uint64_t per = 0;
    for (uint64_t v : parts) per += align16(v);
    return per * tt;
}

uint64_t glm_chunk_init(const ModelGeometry& g, int64_t k, int64_t T, void* base, GlmChunkBuffers& c) {
    const int64_t n = g.n_embd, hc = g.hc;
    Arena a{(uint8_t*) base, 0};
    // ONE ROW-GROUP AT A TIME, in the order `glm_chunk_bytes` adds them, so the two cannot drift: every field
    // is `T * row`, and a field that got a `T` in one and not the other would be a silent overlap.
    c.R = (float*) a.take((uint64_t) T * hc * n * 4);
    c.cur = (float*) a.take((uint64_t) T * n * 4);
    c.block_out = (float*) a.take((uint64_t) T * n * 4);
    c.shared = (float*) a.take((uint64_t) T * n * 4);
    c.parts = (float*) a.take((uint64_t) T * k * n * 4);
    c.post = (float*) a.take((uint64_t) T * hc * 4);
    c.comb = (float*) a.take((uint64_t) T * hc * hc * 4);
    c.weights = (float*) a.take((uint64_t) T * k * 4);
    c.ids = (int32_t*) a.take((uint64_t) T * k * 4);
    c.logits = (float*) a.take((uint64_t) T * (uint64_t) g.n_expert * 4);
    c.n_embd = n;
    c.hc = hc;
    c.k = k;
    c.n_expert = g.n_expert;
    c.T = T;
    return a.used;
}

void glm_chunk_view(const GlmChunkBuffers& c, const GlmBuffers& b, const MoEBuffers& mb, const BlockBuffers& bb,
                    int64_t t, GlmBuffers& out_b, MoEBuffers& out_mb, BlockBuffers& out_bb) {
    const size_t n = (size_t) c.n_embd, hc = (size_t) c.hc, k = (size_t) c.k;
    static_assert(sizeof(GlmBuffers) < 4096 && sizeof(MoEBuffers) < 512 && sizeof(BlockBuffers) < 512,
                  "a chunk view is copied twice per token per layer, so it is a struct copy and not a heap one");
    out_b = b;
    out_mb = mb;
    out_bb = bb;
    // `cur` FIRST: it is the field the pool and the router both read, and the pool is the reason this exists.
    out_b.cur = c.cur + (size_t) t * n;
    out_b.post = c.post + (size_t) t * hc;
    out_b.comb = c.comb + (size_t) t * hc * hc;
    out_mb.shared = c.shared + (size_t) t * n;
    out_mb.weights = c.weights + (size_t) t * k;
    out_mb.ids = (int*) (c.ids + (size_t) t * k);
    out_mb.logits = c.logits + (size_t) t * (size_t) c.n_expert;
    out_bb.R = c.R + (size_t) t * hc * n;
    out_bb.block_out = c.block_out + (size_t) t * n;
}

void glm_chunk_group_view(const GlmChunkBuffers& c, const GlmBuffers& group, const MoEBuffers& mb,
                          const BlockBuffers& bb, int64_t t0, int64_t ntok, GlmBuffers& out_b, MoEBuffers& out_mb,
                          BlockBuffers& out_bb) {
    const size_t n = (size_t) c.n_embd, hc = (size_t) c.hc, k = (size_t) c.k, ex = (size_t) c.n_expert;
    out_b = group;
    out_mb = mb;
    out_bb = bb;
    out_b.ntok = ntok;
    out_b.cur = c.cur + (size_t) t0 * n;
    out_b.post = c.post + (size_t) t0 * hc;
    out_b.comb = c.comb + (size_t) t0 * hc * hc;
    out_mb.shared = c.shared + (size_t) t0 * n;
    out_mb.weights = c.weights + (size_t) t0 * k;
    out_mb.ids = (int*) (c.ids + (size_t) t0 * k);
    out_mb.logits = c.logits + (size_t) t0 * ex;
    out_bb.R = c.R + (size_t) t0 * hc * n;
    out_bb.block_out = c.block_out + (size_t) t0 * n;
}

// ================================ the per-layer persistent state ================================

uint64_t glm_kda_state_floats(const ModelGeometry& g) {
    if (g.kda_head_dim <= 0 || g.n_head <= 0) return 0;
    const uint64_t delta = (uint64_t) g.n_head * g.kda_head_dim * g.kda_head_dim;
    const uint64_t conv = (uint64_t) (g.kda_conv_kernel > 0 ? g.kda_conv_kernel - 1 : 0) * 3 * (uint64_t) kda_n_v(g);
    return delta + conv;
}

uint64_t glm_mla_cache_bytes(const ModelGeometry& g, int64_t max_cells) {
    if (g.kv_lora_rank <= 0 || max_cells <= 0) return 0;
    return (uint64_t) max_cells * (uint64_t) g.kv_lora_rank * 2;
}

bool glm_is_kda_layer(const ModelGeometry& g, int64_t layer) { return !is_qsa_layer(g, layer); }

uint64_t glm_mla_state_bytes(const ModelGeometry& g, int64_t max_cells) {
    // The latent cache and the indexer state, in the order `glm_mla_state_init` lays them down.  The cache's
    // size is aligned UP before the indexer's arrays start: the cache is fp16 and the indexer's are f32, so on
    // a latent width that is not a multiple of 2 the two would otherwise disagree about where a float may live.
    return align16(glm_mla_cache_bytes(g, max_cells)) + glm_dsa_state_bytes(g, max_cells);
}

uint64_t glm_mla_state_init(const ModelGeometry& g, int64_t max_cells, void* base, GlmLayerState& st) {
    st.kda_state = nullptr;
    st.kda_conv = nullptr;
    st.mla_cache = nullptr;
    st.idx_partial_k = nullptr;
    st.idx_partial_g = nullptr;
    st.idx_pooled = nullptr;
    st.max_cells = max_cells;
    st.mla_cache = (uint16_t*) base;
    const uint64_t cache = align16(glm_mla_cache_bytes(g, max_cells));
    // The indexer, in the order `glm_dsa_state_bytes` adds it: the pool in progress (key, then gate), then
    // every completed pool.  All three are zeroed with the rest of the carve and the partial is only ever READ
    // once all `kpool` of its members have been written, so a stale partial cannot be pooled.
    if (glm_dsa_state_bytes(g, max_cells) > 0) {
        uint8_t* p = (uint8_t*) base + cache;
        const uint64_t partial = (uint64_t) g.idx_key_dim * (uint64_t) g.idx_kpool * 4;
        st.idx_partial_k = (float*) p;
        st.idx_partial_g = (float*) (p + align16(partial));
        st.idx_pooled = (float*) (p + 2 * align16(partial));
    }
    return glm_mla_state_bytes(g, max_cells);
}

uint64_t glm_layer_state_bytes(const ModelGeometry& g, int64_t max_cells, int64_t layer) {
    if (glm_is_kda_layer(g, layer)) return glm_kda_state_floats(g) * 4;
    return glm_mla_state_bytes(g, max_cells);
}

uint64_t glm_state_init(const ModelGeometry& g, int64_t max_cells, int64_t layer, void* base, GlmLayerState& st) {
    if (!glm_is_kda_layer(g, layer)) return glm_mla_state_init(g, max_cells, base, st);
    st.kda_state = nullptr;
    st.kda_conv = nullptr;
    st.mla_cache = nullptr;
    st.idx_partial_k = nullptr;
    st.idx_partial_g = nullptr;
    st.idx_pooled = nullptr;
    st.max_cells = 0;
    // The delta state first, then the conv history - the order `glm_kda_state_floats` adds them in, and the
    // order `kda_layer` indexes them with.  Two pointers into one carve, so a change here is a change to both.
    const uint64_t delta = (uint64_t) g.n_head * g.kda_head_dim * g.kda_head_dim;
    st.kda_state = (float*) base;
    st.kda_conv = st.kda_state + delta;
    return glm_kda_state_floats(g) * 4;
}

// ================================ the pieces ================================

namespace {

/// THE mHC READ: `x` (all `hc` streams) -> `normed` (f32 + bf16 + both quantized images) -> `mixes` -> `pre`,
/// `post`, `comb`, and finally `cur = sum_j pre[j] * x[i0 + j*n_embd]`.
///
/// The norm is WEIGHTLESS and over the FLATTENED `n_embd * hc` row, which is the part a reader expects to be a
/// learned norm and is not.  The `hc_*_fn` projection is the only learned step here.
bool hc_read(const WeightTable& tables, const ModelGeometry& g, int64_t layer, const char* which, const char* base_name,
             const char* scale_name, const float* x, const GlmBuffers& b, void* stream, std::string& err) {
    const LayerView v(tables, layer);
    const std::string fn = v.name(which);
    const WeightRef* w_fn = v.get(which);
    if (w_fn == nullptr) { err = fn + " is missing"; return false; }

    const WeightRef* w_base = req(v, base_name, err);
    const WeightRef* w_scale = req(v, scale_name, err);
    if (w_base == nullptr || w_scale == nullptr) return false;

    const int64_t cols = g.n_embd * g.hc;
    kernels::GlmHcShapes hcs;
    hcs.n_embd = g.n_embd;
    hcs.hc = g.hc;
    hcs.mix = g.hc_mix;
    hcs.sinkhorn_iters = 20;
    // TWO epsilons in one struct, and swapping them is silent: `eps` is `hyper_connection.epsilon` and belongs
    // to `pre` and the Sinkhorn; the weightless norm above the `hc_*_fn` projection is the reference's plain
    // `ggml_rms_norm(flat, f_norm_rms_eps)` and takes the model's norm eps.  On layer 0 the difference is 8%.
    hcs.norm_eps = (float) g.rms_eps;

    // `b.ntok` THROUGHOUT: the mHC half is per token and every kernel it uses already takes a token count, so
    // a chunk is the same five calls with a wider T - one normalize, one quantize pair, one 24-wide projection
    // and one Sinkhorn per CHUNK instead of per token.
    kernels::glm_hc_norm_bf16(x, b.normed_bf16, b.normed_f32, hcs, b.ntok, stream);
    quantize_both(b.normed_f32, b.normed_q8k, b.normed_q8_0, cols * b.ntok, stream);

    if (!project(*w_fn, fn, b.normed_f32, b.normed_q8_0, b.normed_q8k, b.normed_bf16, b.mixes, cols, g.hc_mix,
                 b.ntok, stream, err)) {
        return false;
    }
    // `base` and `scale` are 1-D f32 in the engine arena - 24 and 3 floats - and are read by the kernel
    // directly.  There is no projection to run and nothing to quantize.
    if (w_base->bytes < (uint64_t) g.hc_mix * 4 || w_scale->bytes < 3 * 4) {
        err = fn + ": hc base/scale are not f32 of the shape the mHC kernel indexes";
        return false;
    }
    kernels::glm_hc_pre(b.mixes, (const float*) w_scale->data, (const float*) w_base->data, b.pre, b.post, b.comb, hcs,
                        b.ntok, stream);
    kernels::glm_hc_mix(x, b.pre, b.cur, hcs, b.ntok, stream);
    return true;
}

/// The mHC WRITE.  `sub` is the sublayer's output (`n_embd`), `residual` the streams it was fed from.  The
/// result is `res_scratch`, which `hc_write_commit` copies back - `hc_post` reads every source stream for every
/// destination element, so it cannot be handed the same pointer twice.
void hc_write(const ModelGeometry& g, const float* sub, const GlmBuffers& b, const float* residual, void* stream) {
    kernels::GlmHcShapes hcs;
    hcs.n_embd = g.n_embd;
    hcs.hc = g.hc;
    hcs.mix = g.hc_mix;
    hcs.sinkhorn_iters = 20;
    kernels::glm_hc_post(sub, b.post, residual, b.comb, b.res_scratch, hcs, b.ntok, stream);
}

bool hc_write_commit(float* R, const GlmBuffers& b, const ModelGeometry& g, void* stream, std::string& err) {
    if (cudaMemcpyAsync(R, b.res_scratch, (size_t) g.hc * g.n_embd * 4 * (size_t) b.ntok, cudaMemcpyDeviceToDevice,
                        (cudaStream_t) stream) != cudaSuccess) {
        err = "glm5-next: the mHC residual write-back failed";
        return false;
    }
    return true;
}

/// THE KDA LAYER.  `b.cur` holds the normed sublayer input and `sub` receives the `n_embd` output.
///
/// The order is the reference's and every step of it is load-bearing:
///
///   1. q, k, v = wq/wk/wv @ cur, written as three runs of ONE buffer, because the conv that follows is over
///      their CONCATENATION with a single shared state.
///   2. raw = ssm_f_b(ssm_f_a(cur)); z = ssm_g_b(ssm_g_a(cur)); beta = ssm_beta @ cur (RAW - the delta kernel
///      applies the sigmoid itself).
///   3. gate = floor * sigmoid(-(ssm_a * (raw + dt))), bounded to (floor, 0).
///   4. conv over [q;k;v] with kernel 4, then SiLU - on the WHOLE conv output, before the split.
///   5. L2-normalise q and k with the RMS eps as a FLOOR ON THE NORM.
///   6. the delta recurrence, which updates the per-head state in place.
///   7. y = rms_norm(o, ssm_norm) * SIGMOID(z).
///   8. out = ssm_out @ y.
bool kda_layer(const WeightTable& tables, const ModelGeometry& g, int64_t layer, const GlmBuffers& b,
               GlmLayerState& st, void* stream, std::string& err) {
    const LayerView v(tables, layer);
    const int64_t n = g.n_embd;
    const int64_t nv = kda_n_v(g);
    const int64_t nh = g.n_head;
    const int64_t hd = g.kda_head_dim;

    const WeightRef* wq = req(v, "attn_q.weight", err);
    const WeightRef* wk = req(v, "attn_k.weight", err);
    const WeightRef* wv = req(v, "attn_v.weight", err);
    const WeightRef* w_out = req(v, "attn_output.weight", err);
    const WeightRef* w_beta = req(v, "ssm_beta.weight", err);
    const WeightRef* w_fa = req(v, "ssm_f_a.weight", err);
    const WeightRef* w_fb = req(v, "ssm_f_b.weight", err);
    const WeightRef* w_ga = req(v, "ssm_g_a.weight", err);
    const WeightRef* w_gb = req(v, "ssm_g_b.weight", err);
    const WeightRef* w_a = req(v, "ssm_a", err);
    const WeightRef* w_dt = req(v, "ssm_dt.bias", err);
    const WeightRef* w_norm = req(v, "ssm_norm.weight", err);
    const WeightRef* w_cq = req(v, "ssm_conv1d_q.weight", err);
    const WeightRef* w_ck = req(v, "ssm_conv1d_k.weight", err);
    const WeightRef* w_cv = req(v, "ssm_conv1d_v.weight", err);
    if (!wq || !wk || !wv || !w_out || !w_beta || !w_fa || !w_fb || !w_ga || !w_gb || !w_a || !w_dt || !w_norm ||
        !w_cq || !w_ck || !w_cv) {
        return false;
    }

    const float eps = (float) g.rms_eps;

    // ---- 1. q, k, v into the three runs of `qkv`.  Each reads `cur`, and `cur`'s images are already built.
    //      `b.q`/`b.kk`/`b.v` rather than `b.qkv`/`+nv`/`+2nv`: at `ntok > 1` each stream is its own `ntok`
    //      contiguous run of `n_v`, which is what the three projections below write and what the conv wants.
    if (!project(*wq, v.name("attn_q.weight"), b.cur, b.cur_q8_0, b.cur_q8k, b.cur_bf16, b.q, n, nv, b.ntok, stream,
                 err))
        return false;
    if (!project(*wk, v.name("attn_k.weight"), b.cur, b.cur_q8_0, b.cur_q8k, b.cur_bf16, b.kk, n, nv, b.ntok, stream,
                 err))
        return false;
    if (!project(*wv, v.name("attn_v.weight"), b.cur, b.cur_q8_0, b.cur_q8k, b.cur_bf16, b.v, n, nv, b.ntok, stream,
                 err))
        return false;

    // ---- 2. beta raw, the gated output's pre-activation, and the decay's pre-activation.
    if (!project(*w_beta, v.name("ssm_beta.weight"), b.cur, b.cur_q8_0, b.cur_q8k, b.cur_bf16, b.beta, n, nh, b.ntok,
                 stream, err))
        return false;
    if (!project(*w_ga, v.name("ssm_g_a.weight"), b.cur, b.cur_q8_0, b.cur_q8k, b.cur_bf16, b.tail, n, TAIL, b.ntok,
                 stream, err))
        return false;
    kernels::quantize_q8_0(b.tail, b.tail_q8_0, TAIL * b.ntok, stream);
    if (!project(*w_gb, v.name("ssm_g_b.weight"), b.tail, b.tail_q8_0, nullptr, nullptr, b.z, TAIL, nv, b.ntok, stream,
                 err))
        return false;
    if (!project(*w_fa, v.name("ssm_f_a.weight"), b.cur, b.cur_q8_0, b.cur_q8k, b.cur_bf16, b.tail_f, n, TAIL, b.ntok,
                 stream, err))
        return false;
    kernels::quantize_q8_0(b.tail_f, b.tail_q8_0, TAIL * b.ntok, stream);
    if (!project(*w_fb, v.name("ssm_f_b.weight"), b.tail_f, b.tail_q8_0, nullptr, nullptr, b.raw, TAIL, nv, b.ntok,
                 stream, err))
        return false;

    // ---- 3. the bounded decay gate.  `ssm_a` is already `-exp(A_log)` in the file; the kernel applies the
    //      second negation the reference does with `ggml_scale(..., -1.0f)`.
    if (w_a->bytes < (uint64_t) nh * 4 || w_dt->bytes < (uint64_t) nv * 4) {
        err = v.name("ssm_a") + ": ssm_a/ssm_dt are not f32 of the shape the gate indexes";
        return false;
    }
    kernels::glm_kda_gate(b.raw, (const float*) w_dt->data, (const float*) w_a->data, b.gate, nv, nh, hd,
                          (float) g.kda_gate_floor, b.ntok, stream);

    // ---- 4. the conv over the CONCATENATION, then SiLU, in place on all 3*n_v channels.  THE SPLIT FORM AT
    //      `ld = n_v`: with one token the three stream pointers below ARE the interleaved buffer, so this is
    //      the same call the decode path always made, and with a chunk it is one launch over the whole chunk's
    //      state walk instead of one per token - which matters because the conv is the only strictly sequential
    //      step in a KDA layer and a per-token loop would pay its launch latency `T` times.
    if (!st.kda_conv) { err = "glm5-next: the KDA conv state was never carved"; return false; }
    kernels::glm_kda_conv_silu3(b.q, b.kk, b.v, (const float*) w_cq->data, (const float*) w_ck->data,
                                (const float*) w_cv->data, st.kda_conv, nv, b.qkv_ld, b.ntok,
                                (int64_t) g.kda_conv_kernel, stream);

    // ---- 5. L2 on q and k only.  v is NOT normalised.
    kernels::glm_kda_l2norm(b.q, b.kk, nh, hd, eps, b.ntok, stream);

    // ---- 6. the recurrence.  It reads q/k/v in place and writes `b.o`.  The kernel already walks `T` tokens
    //      inside itself, updating the head state in place as it goes - which is the whole reason a chunk can
    //      be layer-major without a parallel scan.
    if (!st.kda_state) { err = "glm5-next: the KDA delta state was never carved"; return false; }
    kernels::glm_kda_delta(b.q, b.kk, b.v, b.gate, b.beta, st.kda_state, b.o, hd, nh, b.ntok, stream);

    // ---- 7. y = rms_norm(o, ssm_norm) * sigmoid(z).  SIGMOID - not the SiLU the conv above uses, and not the
    //      silu the reference's own `ggml_silu` name suggests at a glance.
    if (w_norm->bytes < (uint64_t) hd * 4) {
        err = v.name("ssm_norm.weight") + ": not f32 of the head width";
        return false;
    }
    kernels::rms_norm_weighted(b.o, (const float*) w_norm->data, nh * b.ntok, hd, eps, stream);
    kernels::glm_sigmoid_mul(b.o, b.z, nv * b.ntok, stream);

    // ---- 8. out = ssm_out @ y.  `y` is `n_v` wide, which is not `cur`, so it takes the wide images.
    quantize_both(b.o, b.wide_q8k, b.wide_q8_0, nv * b.ntok, stream);
    return project(*w_out, v.name("attn_output.weight"), b.o, b.wide_q8_0, b.wide_q8k, b.wide_bf16, b.attn_out, nv, n,
                   b.ntok, stream, err);
}

/// THE MLA LAYER - absorbed, NoPE, and sharing the tensor name `attn_output.weight` with KDA's output
/// projection, which is why the two mixers are chosen by `glm_is_kda_layer` and never by a name.
///
/// `b.cur` holds the normed sublayer input; the result lands in `b.attn_out`.  Every step below differs from
/// the first family's full attention in a way that RUNS when it is wrong, so they are spelled out:
///
///   1. `qr = wq_a @ cur` -> 1536, then RMSNorm ON THE LATENT (`attn_q_a_norm`).  The norm is on the
///      compressed query, before the up-projection - not on the heads, where every other arch puts it.
///   2. `q = wq_b @ qr` -> 64 x 256.
///   3. **ABSORPTION.**  Head `h`'s 256 query values are mapped back into the 512-wide latent space by head
///      `h`'s own 256x512 block of `attn_k_b`, so a score is a 512-dot between two latent vectors and the
///      512x256 up-projection of the key never has to be materialised.  That block is rows
///      `[h*512, (h+1)*512)` of the folded matrix - see `project_rows`.
///   4. `kv_cmpr = RMSNorm(wkv_a_mqa @ cur, attn_kv_a_norm)` - exactly 512 wide.  With `rope.dimension_count`
///      zero there is no nope/rope split, so the usual split-and-recombine is a NO-OP rather than a step to
///      get wrong.  The cache stores this, and **K AND V ARE THE SAME TENSOR**.
///   5. attention with `scale = 1/sqrt(mla_head_dim)` = 1/16: the scale is over the 256-wide HEAD, not over
///      the 512 the cache rows are, and using 1/sqrt(512) is a plausible constant that is simply wrong.
///   6. **DE-ABSORPTION.**  Head `h`'s 512 attention values come back through head `h`'s 512x256 block of
///      `attn_v_b`.
///   7. `attn_output @ head_out` reads 16384 = `n_head * mla_head_dim`, the widest activation in the arch and
///      the one `wide_width` exists for.
///
/// `abs_pos` is the ABSOLUTE position of this token: it is the RoPE position of every query in the chunk, and
/// for a layer that runs EVERY position of a sequence it is also the cache row and the last row a query may
/// attend to ("the cache is indexed by position" - there is no window and no ring here).
///
/// **`row0`/`vis0` SEPARATE THOSE TWO ROLES FOR THE ONE LAYER THAT DOES NOT RUN EVERY POSITION.**  The MTP block
/// is fed at the positions its caller picks, so its cache holds a row per DRAFT STEP and not one per position -
/// the reference's own semantics: its draft context is a fresh context whose cells are allocated in write order,
/// so cell `i` holds the `i`-th row written, whatever absolute position that row carries (and its mask is
/// `cells[i].pos <= pos`, which on a fresh cell - `seq_id` unset - is a mask-out, not a zero).  `row0` is the
/// row token 0 of the chunk goes into and `vis0` the number of rows, counted from row 0, that its query may
/// attend to; token `t` uses `row0 + t` and `vis0 + t`.  Both default to the dense reading, `abs_pos` and
/// `abs_pos + 1`, which is what every trunk caller leaves them at.
bool mla_layer(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t abs_pos,
               const GlmBuffers& b, const GlmLayerState& st, void* stream, std::string& err,
               int64_t row0 = -1, int64_t vis0 = -1) {
    if (row0 < 0) row0 = abs_pos;
    if (vis0 < 0) vis0 = abs_pos + 1;
    const LayerView v(tables, layer);
    const int64_t n = g.n_embd;
    const int64_t nh = g.n_head;
    const int64_t hd = mla_head_width(g);
    const int64_t kvl = g.kv_lora_rank;
    const int64_t qlr = g.q_lora_rank;
    if (hd <= 0 || kvl <= 0 || qlr <= 0) {
        err = "glm5-next: the MLA geometry is not set (q_lora_rank/kv_lora_rank/mla_head_dim are zero)";
        return false;
    }
    // The per-head image pair below is indexed at `q8k_bytes(hd) * h`, and that stride is only the head's own
    // Q8_K image when `hd` is a whole number of 256-element blocks.  A head width that is not would make the
    // stride truncate - every head reading the previous head's blocks - and `quantize_q8_K` would have exited
    // on the per-head form this replaced, so the requirement is not new, only moved to where it can say so.
    if (hd % 256 != 0 || kvl % 256 != 0) {
        err = "glm5-next: mla_head_dim " + std::to_string((long long) hd) + " and kv_lora_rank " +
              std::to_string((long long) kvl) + " must both be multiples of 256, the Q8_K block a head's "
              "image is quantized in";
        return false;
    }
    const WeightRef* wq_a = req(v, "attn_q_a.weight", err);
    const WeightRef* wq_a_norm = req(v, "attn_q_a_norm.weight", err);
    const WeightRef* wq_b = req(v, "attn_q_b.weight", err);
    const WeightRef* wk_b = req(v, "attn_k_b.weight", err);
    const WeightRef* wkv_a = req(v, "attn_kv_a_mqa.weight", err);
    const WeightRef* wkv_a_norm = req(v, "attn_kv_a_norm.weight", err);
    const WeightRef* wv_b = req(v, "attn_v_b.weight", err);
    const WeightRef* w_out = req(v, "attn_output.weight", err);
    if (!wq_a || !wq_a_norm || !wq_b || !wk_b || !wkv_a || !wkv_a_norm || !wv_b || !w_out) return false;
    if (wq_a_norm->bytes < (uint64_t) qlr * 4 || wkv_a_norm->bytes < (uint64_t) kvl * 4) {
        err = v.name("attn_q_a_norm.weight") + "/" + v.name("attn_kv_a_norm.weight") +
              ": the MLA latents' norms are not f32 of the rank they normalise";
        return false;
    }
    const float eps = (float) g.rms_eps;

    // ---- 1/2. the query latent, its norm, and the up-projection.
    if (!project(*wq_a, v.name("attn_q_a.weight"), b.cur, b.cur_q8_0, b.cur_q8k, b.cur_bf16, b.qr, n, qlr, b.ntok,
                 stream, err))
        return false;
    kernels::rms_norm_weighted(b.qr, (const float*) wq_a_norm->data, b.ntok, qlr, eps, stream);
    // 1536 is a multiple of 256, so the latent takes the wide pair; `cur` is still live (step 4 reads it) and
    // could not have been reused even if the widths matched.
    quantize_both(b.qr, b.wide_q8k, b.wide_q8_0, qlr * b.ntok, stream);
    kernels::f32_to_bf16_bulk(b.qr, b.wide_bf16, qlr * b.ntok, stream);
    if (!project(*wq_b, v.name("attn_q_b.weight"), b.qr, b.wide_q8_0, b.wide_q8k, b.wide_bf16, b.qfull, qlr,
                 nh * hd, b.ntok, stream, err))
        return false;

    // ---- 3. absorption, one head at a time.  THE BAND IS THE HEAD - both loops below index a folded
    //      per-head matrix, and the two bands are DIFFERENT WIDTHS (512 rows of a 256-wide key map, 256 rows
    //      of a 512-wide value map), which is where a 256 and a 512 get swapped.
    //
    //      **NEITHER THE QUANTIZE NOR THE PROJECTION IS PER TOKEN ANY MORE.**  The projection has to be per
    //      head - each head has its own input and its own band of the folded matrix - but a band is the SAME
    //      weight rows for every token in the group, so the group's tokens are the `ncols` a single
    //      `project_rows` call carries.  What stands in the way is the LAYOUT: `project_rows` takes its columns
    //      contiguous and `w` apart, and token-major `qfull` puts the head fastest, so the same head two tokens
    //      apart is `n_head * hd` away.  `glm_heads_major` swaps the two outer axes for one launch, and that is
    //      the whole of the fix - the group is then the inner axis and head `h`'s columns are adjacent.
    //
    //      **THE QUANTIZE STILL COVERS THE WHOLE GROUP IN ONE CALL**, because the head-major stack is still
    //      `n_head` slices each a whole number of 256-element Q8_K blocks and 32-element Q8_0 ones, so one
    //      `quantize_both` produces exactly the blocks `n_head` separate ones would at the offsets the loop
    //      below reads.  This note used to explain why the quantize was hoisted out of the loop while the
    //      projection stayed in it; both are now once per group.
    //
    //      MEASURED, and this is the whole reason the loop was rewritten.  `nsys` on one 128-token chunk of
    //      the 4-way split (`UD-IQ4_XS`, `--prefill 128`): the two band loops were 180,224 `ncols == 1`
    //      GEMVs, one `native_quantize_q8_1` each, so 360,448 of the chunk's 434,144 kernel launches were
    //      these - 83.0% - at 2.18 us a call for 136 KiB of weights, and the GPU kernel time was 386 ms for
    //      the GEMVs and 218 ms for their quantizes.  The count is exact: 128 tokens x 11 MLA layers x 64
    //      heads x 2 tensors.  AFTER: 22,528 calls and 45,056 launches.  The census weighs both sides of it -
    //      one stage's 128-token chunk went from 51,216 calls to 8,208 over the same 65,664 columns, and the
    //      band tensors that were 8,192 calls at one column a call do not reach its top fourteen any more.  The
    //      group width is the whole of the difference, and the `heads_in` note in the header carries the
    //      arithmetic.
    kernels::glm_heads_major(b.qfull, b.heads_in, nh, b.ntok, hd, stream);
    quantize_both(b.heads_in, b.heads_q8k, b.heads_q8_0, nh * hd * b.ntok, stream);
    const int64_t hd_q8k = (int64_t) q8k_bytes(hd), hd_q8_0 = (hd / 32) * 34;
    // The group's columns for head `h` are `b.ntok` contiguous `hd`-wide ones, so the image strides are the
    // group, not the stack.  `heads_q8k`/`heads_q8_0` are sized for the WIDER band per token and this layout
    // is `nh * ntok * q8k_bytes(hd)` at most, which fits inside it.
    const int64_t grp_q8k = (int64_t) b.ntok * hd_q8k, grp_q8_0 = (int64_t) b.ntok * hd_q8_0;
    for (int64_t h = 0; h < nh; ++h) {
        if (!project_rows(*wk_b, v.name("attn_k_b.weight"), b.heads_in + h * b.ntok * hd,
                          b.heads_q8_0 + h * grp_q8_0, b.heads_q8k + h * grp_q8k,
                          b.band_out + h * b.ntok * kvl, hd, h * kvl, kvl, stream, err, b.ntok)) {
            return false;
        }
    }
    // **NO WAY BACK, UNLESS THE SELECTION NEEDS ONE.**  `band_out` is `[h][t][kvl]`, which is the layout the
    // attention kernel asks for, so a dense layer hands this buffer straight to it and the transpose to
    // token-major `qabs` happens only where the DSA selection below reads `qabs` per token.  See step 5.

    // ---- 4. the latent, its norm, and the cache write.  K == V, so there is one write and no second tensor.
    if (!project(*wkv_a, v.name("attn_kv_a_mqa.weight"), b.cur, b.cur_q8_0, b.cur_q8k, b.cur_bf16, b.kv_cmpr, n,
                 kvl, b.ntok, stream, err))
        return false;
    kernels::rms_norm_weighted(b.kv_cmpr, (const float*) wkv_a_norm->data, b.ntok, kvl, eps, stream);
    if (st.mla_cache == nullptr) { err = "glm5-next: the MLA latent cache was never carved"; return false; }
    if (row0 < 0 || row0 + b.ntok > st.max_cells) {
        err = "glm5-next: MLA cache rows " + std::to_string((long long) row0) + ".." +
              std::to_string((long long) (row0 + b.ntok - 1)) + " are outside the " +
              std::to_string((long long) st.max_cells) + "-row latent cache";
        return false;
    }
    // **ALL `ntok` ROWS GO IN BEFORE ANY QUERY RUNS, AND THAT IS CAUSAL ANYWAY.**  The attention below masks by
    // `n_kv` - query `t` walks `0 .. vis0 + t - 1` and never looks at the rows above it - so a row written early
    // is a row no query in this chunk asks for.  Ordering the write inside the token loop instead would cost a
    // second launch per token to hide data that is already hidden.
    kernels::glm_mla_cache_store_t(b.kv_cmpr, st.mla_cache, row0, kvl, b.ntok, stream);

    // ---- 5. the attention itself, THE WHOLE GROUP IN ONE CALL.  It used to be `ntok` calls of one query each,
    //      with `T` pinned to 1 - the cache holds the sequence and the kernel walks it, so one query was one
    //      launch and a group was eight of them.
    //
    //      **THE MASK IS STILL A FUNCTION OF ABSOLUTE POSITION; IT JUST DOES NOT NEED A LAUNCH A TOKEN.**  The
    //      kernel masks query `t` at `pos_base + t + 1`, so the `ntok` per-token calls were the same walk with
    //      `pos_base` and `n_kv` both counting up by one.  One call at `pos_base = abs_pos` and
    //      `n_kv = vis0 + ntok - 1` reproduces every one of them: query `t`'s limit is `min(abs_pos + t + 1,
    //      n_kv)`, and with `vis0 >= abs_pos + 1` the second term is never the smaller one.
    //
    //      **AND THE TWO `glm_heads_major` CALLS THAT WRAPPED IT ARE GONE WITH IT.**  The group went
    //      `[h][t][kvl] -> [t][h][kvl] -> attention -> [t][h][kvl] -> [h][t][kvl]`, four buffers and two
    //      transposes for a kernel whose own contract is the head-major one at both ends.  It reads `band_out`
    //      where step 3 left it and writes `heads_in` where step 6 wants it, which is the same place and the
    //      same layout the transpose used to put it.
    //
    //      **OR THE DSA SELECTION, WHICH REPLACES THIS CALL AND NOTHING ELSE.**  With `--dsa` the layer attends the
    //      `idx_top_k` cells the k-pool indexer picks instead of every cell - the reference's own optional path
    //      (`cparams.dsa`, off by default there too).  Everything above and below is untouched: the same
    //      absorbed query goes in and the same pre-de-absorption output comes out.
    const bool dsa = kernels::glm_dsa_enabled();
    // The indexer addresses the cache by ABSOLUTE POSITION (`glm_dsa_select`'s pools, `n_vis = (pos+1)/kpool`),
    // which is the same address the dense reading uses and a different one from a compacted row counter.  Only
    // the MTP block compacts, so only it can hit this - and the reference's own draft graph builds its indexer
    // off the draft context's cells, which nothing in this engine has.  Refuse rather than select the wrong
    // cells.
    if (dsa && row0 != abs_pos) {
        err = "glm5-next: --dsa addresses the MLA cache by absolute position and the draft block's cache is "
              "compacted; the two cannot be combined";
        return false;
    }
    // Two of the indexer's values are needed by the PER-QUERY loop below and not only by the batch above it: the
    // key width, and `ape` (the pooled members' additive embedding, which `glm_dsa_pool` reads for every pool).
    const int64_t kd = g.idx_key_dim, ih = g.idx_q_heads;
    const float* ape = nullptr;
    std::vector<int32_t> pos_host;   // the chunk's absolute positions, uploaded in one copy; empty when `!dsa`
    if (dsa) {
        // The four indexer projections, over the whole chunk.  `cur` still carries the images steps 1 and 4 read,
        // and `qr` the ones step 2 read - so `iq` takes the WIDE pair and the other three the `cur` pair.
        const WeightRef* w_ik = req(v, "indexer.attn_k.weight", err);
        const WeightRef* w_kn = req(v, "indexer.k_norm.weight", err);
        const WeightRef* w_kb = req(v, "indexer.k_norm.bias", err);
        const WeightRef* w_ig = req(v, "indexer_compressor_gate.weight", err);
        const WeightRef* w_iq = req(v, "indexer.attn_q_b.weight", err);
        const WeightRef* w_iw = req(v, "indexer.proj.weight", err);
        if (!w_ik || !w_kn || !w_kb || !w_ig || !w_iq || !w_iw) return false;
        if (kd <= 0 || ih <= 0 || st.idx_partial_k == nullptr || st.idx_pooled == nullptr) {
            err = "glm5-next: --dsa needs the indexer, whose geometry (key_length/head_count) and carved state "
                  "this model does not have";
            return false;
        }
        // The key norm's two vectors are read as f32 of the key width, the same check the MLA latents' norms get.
        if (w_kn->bytes < (uint64_t) kd * 4 || w_kb->bytes < (uint64_t) kd * 4) {
            err = v.name("indexer.k_norm.weight") + "/" + v.name("indexer.k_norm.bias") +
                  ": the indexer's key norm is not f32 of the key width";
            return false;
        }
        const WeightRef* w_ape = req(v, "indexer_compressor_ape.weight", err);
        if (w_ape == nullptr) return false;
        if (w_ape->bytes < (uint64_t) kd * g.idx_kpool * 4) {
            err = v.name("indexer_compressor_ape.weight") + ": not f32 of [key_length, kpool]";
            return false;
        }
        if (g.idx_kpool <= 0 || g.idx_kpool > kd) {
            err = "glm5-next: idx_kpool " + std::to_string((long long) g.idx_kpool) + " is not usable";
            return false;
        }
        ape = (const float*) w_ape->data;
        // `iq` is `attn_q_b @ qr` - THE SAME NORMED LATENT the MLA query came from, read twice for two different
        // up-projections, which is why it is `qr` and not `qfull`.
        if (!project(*w_ik, v.name("indexer.attn_k.weight"), b.cur, b.cur_q8_0, b.cur_q8k, b.cur_bf16, b.idx_key, n,
                     kd, b.ntok, stream, err))
            return false;
        kernels::glm_layer_norm(b.idx_key, (const float*) w_kn->data, (const float*) w_kb->data, b.idx_key, b.ntok,
                               kd, eps, stream);
        if (!project(*w_ig, v.name("indexer_compressor_gate.weight"), b.cur, b.cur_q8_0, b.cur_q8k, b.cur_bf16,
                     b.idx_gate, n, kd, b.ntok, stream, err))
            return false;
        if (!project(*w_iq, v.name("indexer.attn_q_b.weight"), b.qr, b.wide_q8_0, b.wide_q8k, b.wide_bf16, b.idx_iq,
                     qlr, kd * ih, b.ntok, stream, err))
            return false;
        // `iw = proj @ cur * prescale`, the prescale being `1/sqrt(key_dim * idx_heads)` - the reference divides
        // by that where it BUILDS the weights, so it belongs to this projection and not to `glm_dsa_score`.
        if (!project(*w_iw, v.name("indexer.proj.weight"), b.cur, b.cur_q8_0, b.cur_q8k, b.cur_bf16, b.idx_iw, n, ih,
                     b.ntok, stream, err))
            return false;
        const float prescale = (float) (1.0 / std::sqrt((double) kd * (double) ih));
        kernels::scale_inplace(b.idx_iw, ih * b.ntok, prescale, stream);
        // The queries' absolute positions, which `glm_dsa_select` masks by - `n_vis = (pos + 1) / kpool`.  They go
        // over in ONE host-to-device copy for the whole chunk: they are the only thing in this path the host has to
        // touch, and doing it per query would be a pageable copy a layer a token on the decode path.
        pos_host.resize((size_t) b.ntok);
        for (int64_t t = 0; t < b.ntok; ++t) pos_host[(size_t) t] = (int32_t) (abs_pos + t);
        cudaMemcpyAsync(b.idx_pos, pos_host.data(), (size_t) b.ntok * 4, cudaMemcpyHostToDevice,
                        (cudaStream_t) stream);
    }

    const int64_t kpool = g.idx_kpool;
    const int top_pools_max = kernels::glm_dsa_top_pools(g.idx_top_k, kpool);
    const int tail = g.idx_select_tail != 0 ? 1 : 0;
    if (!dsa) {
        kernels::glm_mla_attn(b.band_out, st.mla_cache, b.heads_in, nh, kvl, vis0 + b.ntok - 1, b.ntok, abs_pos,
                              (float) (1.0 / std::sqrt((double) hd)), stream);
    } else {
        // The selection is a property of one query, so `--dsa` keeps the per-token walk and with it the two
        // transposes the dense path just dropped: the loop below reads `qabs` and writes `kqv`, both token-major.
        kernels::glm_heads_major(b.band_out, b.qabs, b.ntok, nh, kvl, stream);
        for (int64_t t = 0; t < b.ntok; ++t) {
            // ---- the indexer, for THIS query.  Its own `ik`/`ig` rows go into the pool in progress, this query's
            //      pool is pooled if it completes the pool, and only then is the selection run - a pool is visible to
            //      the query whose own cell is its LAST member (`glm_dsa_select`'s `n_vis`), so pooling after the
            //      selection would hide the current token from itself.
            const int64_t p = abs_pos + t;
            const int64_t slot = p % kpool;
            const int64_t pool_done = (p + 1) / kpool;
            cudaMemcpyAsync(st.idx_partial_k + slot * kd, b.idx_key + t * kd, (size_t) kd * 4, cudaMemcpyDeviceToDevice,
                            (cudaStream_t) stream);
            cudaMemcpyAsync(st.idx_partial_g + slot * kd, b.idx_gate + t * kd, (size_t) kd * 4, cudaMemcpyDeviceToDevice,
                            (cudaStream_t) stream);
            if (slot == kpool - 1) {
                // `glm_dsa_pool` pools from pool 0, so the completed pool is the ONLY pool in this call: its members
                // are the `kpool` rows the partial holds, and its column is `pool_done - 1`.
                kernels::glm_dsa_pool(st.idx_partial_k, st.idx_partial_g, ape, kd, (int) kpool, 1,
                                      st.idx_pooled + (pool_done - 1) * kd, stream);
            }
            int top_pools = (int) (pool_done < top_pools_max ? pool_done : top_pools_max);
            if (pool_done > 0) {
                kernels::glm_dsa_score(b.idx_iq + t * kd * ih, st.idx_pooled, b.idx_iw + t * ih, (int) kd, (int) ih,
                                       /*n_tokens=*/1, (int) pool_done, b.idx_score, stream);
            }
            if (pool_done > 0 || tail) {
                const int n_sel = (int) (kpool * top_pools + (tail ? kpool - 1 : 0));
                kernels::glm_dsa_select(b.idx_score, (int) pool_done, (int) kpool, top_pools, tail, /*n_tokens=*/1,
                                        n_sel, b.idx_pos + t, b.idx_cells, stream);
                kernels::glm_dsa_attn(b.qabs + t * nh * kvl, st.mla_cache, b.idx_cells, (int) kvl, (int) nh, (int) hd,
                                      /*n_tokens=*/1, n_sel, b.kqv + t * nh * kvl, stream);
            } else {
                // Nothing to attend to: with the tail OFF, the first `kpool - 1` queries have no complete pool and the
                // reference zeroes their attention output rather than leaving a stale one.  Unreachable at this
                // model's own reading (`idx_select_tail` is 1), kept because the other reading is a parameter.
                cudaMemsetAsync(b.kqv + t * nh * kvl, 0, (size_t) nh * kvl * 4, (cudaStream_t) stream);
            }
        }
        kernels::glm_heads_major(b.kqv, b.heads_in, nh, b.ntok, kvl, stream);
    }

    // ---- 6. de-absorption.  The same fold as step 3, and here the slices are `kv_lora_rank` wide - still a
    //      whole number of blocks, which is the property that lets one quantize stand in for `n_head`.
    // Same one-call-per-head-per-group as step 3 - see the note there for the numbers.  The bands are the other
    // way round (`kvl`-wide input, `hd` = 256 rows of a 512-wide value map), which is the swap the note in step
    // 3 warns about.  **NO TRANSPOSE IN FRONT OF THE QUANTIZE**: `heads_in` is `[h][t][kvl]` however it was
    // written, by the attention above or by the DSA branch's own `glm_heads_major`.
    quantize_both(b.heads_in, b.heads_q8k, b.heads_q8_0, nh * kvl * b.ntok, stream);
    const int64_t kvl_q8k = (int64_t) q8k_bytes(kvl), kvl_q8_0 = (kvl / 32) * 34;
    const int64_t grp_kvl_q8k = (int64_t) b.ntok * kvl_q8k, grp_kvl_q8_0 = (int64_t) b.ntok * kvl_q8_0;
    for (int64_t h = 0; h < nh; ++h) {
        if (!project_rows(*wv_b, v.name("attn_v_b.weight"), b.heads_in + h * b.ntok * kvl,
                          b.heads_q8_0 + h * grp_kvl_q8_0, b.heads_q8k + h * grp_kvl_q8k,
                          b.band_out + h * b.ntok * hd, kvl, h * hd, hd, stream, err, b.ntok)) {
            return false;
        }
    }
    kernels::glm_heads_major(b.band_out, b.head_out, b.ntok, nh, hd, stream);

    // ---- 7. back to the model's width.
    quantize_both(b.head_out, b.wide_q8k, b.wide_q8_0, nh * hd * b.ntok, stream);
    kernels::f32_to_bf16_bulk(b.head_out, b.wide_bf16, nh * hd * b.ntok, stream);
    return project(*w_out, v.name("attn_output.weight"), b.head_out, b.wide_q8_0, b.wide_q8k, b.wide_bf16,
                   b.attn_out, nh * hd, n, b.ntok, stream, err);
}

/// ONE SwiGLU MLP over the normed input already in `b.cur`: `down(silu(gate(x)) * up(x))`.
///
/// The three names are a parameter because glm5-next spells the same MLP twice - `ffn_gate/up/down` on the
/// three dense-lead layers and `ffn_gate_shexp/up_shexp/down_shexp` for the shared expert on every MoE layer -
/// and the arithmetic is identical on both.  The two writers disagree about nothing else, which is why one
/// function can serve them and a second copy would only be a place for them to drift apart.
///
/// **THE CLAMP IS APPLIED, AND THIS COMMENT USED TO SAY IT WAS NOT.**  It read "NO CLAMP, THOUGH THE GGUF
/// CARRIES ONE ... the reference never applies it", which is the opposite of what both oracles do - the full
/// evidence is in the header of `glm_elt.cu`.  `limit` is `swiglu_clamp_shexp`: BOTH callers here are the
/// shexp path (the dense-lead layers and the shared expert), while the routed experts read
/// `swiglu_clamp_exp` and are clamped on the CPU in `native_gu_rows*`.  `limit <= 0` means no clamp.
///
/// The two up-projections read `cur`, so they share its images; only the down projection needs the wide pair.
bool ffn3(const WeightTable& tables, int64_t layer, const GlmBuffers& b, const char* gate_name, const char* up_name,
          const char* down_name, int64_t n, int64_t fw, float limit, float* out, void* stream, std::string& err) {
    const LayerView v(tables, layer);
    const WeightRef* w_gate = req(v, gate_name, err);
    const WeightRef* w_up = req(v, up_name, err);
    const WeightRef* w_down = req(v, down_name, err);
    if (!w_gate || !w_up || !w_down) return false;

    if (!project(*w_gate, v.name(gate_name), b.cur, b.cur_q8_0, b.cur_q8k, b.cur_bf16, b.ffn_gate, n, fw, b.ntok,
                 stream, err))
        return false;
    if (!project(*w_up, v.name(up_name), b.cur, b.cur_q8_0, b.cur_q8k, b.cur_bf16, b.ffn_up, n, fw, b.ntok, stream,
                 err))
        return false;
    kernels::glm_swiglu(b.ffn_gate, b.ffn_up, limit, fw * b.ntok, stream);

    // `b.ffn_gate` now holds the hidden, and the same `wide` pair that served the KDA output serves it: the
    // width is a parameter of every kernel that touches it, so one pair of images is enough for both roles.
    quantize_both(b.ffn_gate, b.wide_q8k, b.wide_q8_0, fw * b.ntok, stream);
    return project(*w_down, v.name(down_name), b.ffn_gate, b.wide_q8_0, b.wide_q8k, b.wide_bf16, out, fw, n, b.ntok,
                   stream, err);
}

/// THE ROUTER.  `b.cur` is the normed FFN input; the result is `mb.ids`/`mb.weights`, which the host pool reads
/// and `moe_combine_parts` multiplies by.
///
/// The weight is **F32 in this family's files** where qwen4exp's is BF16, so this is `glm_f32_gemv` and not the
/// shared bf16 projection.  Reading 4096x288 of F32 as bf16 gives 288 finite, plausible logits and a different
/// top-8 - a different model, with no error anywhere to see.
///
/// **ONE TOKEN AT A TIME, EVEN INSIDE A GROUP, AND THAT IS NOT AN OVERSIGHT.**  The router is `n_embd x n_expert`
/// of F32 - 4.7 MB against the layer's ~160 MB of quantized weights - so batching it would amortize 3% of the
/// read for the price of a new kernel for both the GEMV and the top-k.  The group path exists for the quantized
/// projections; the router rides along at its own width, and `mb.logits`/`ids`/`weights` are `ntok`-contiguous
/// rows for exactly this loop.
bool glm_router(const WeightTable& tables, const ModelGeometry& g, int64_t layer, const GlmBuffers& b,
                const MoEBuffers& mb, int64_t k, void* stream, std::string& err) {
    const LayerView v(tables, layer);
    const WeightRef* w = req(v, "ffn_gate_inp.weight", err);
    if (w == nullptr) return false;
    const uint64_t need = (uint64_t) g.n_embd * (uint64_t) g.n_expert * 4;
    if (w->bytes < need) {
        err = v.name("ffn_gate_inp.weight") + ": " + std::to_string(w->bytes) + " B, an f32 " +
              std::to_string(g.n_expert) + "x" + std::to_string(g.n_embd) + " router is " + std::to_string(need) + " B";
        return false;
    }
    // `exp_probs_b` steers the SELECTION and is then dropped: the weights are the un-biased probabilities.  A
    // port that biases the weights too still routes to the right experts and scales them wrongly.
    const WeightRef* bias = v.get("exp_probs_b.bias");
    if (bias != nullptr && bias->bytes < (uint64_t) g.n_expert * 4) {
        err = v.name("exp_probs_b.bias") + ": not f32 of the expert count";
        return false;
    }
    const float* bias_f = bias != nullptr ? (const float*) bias->data : nullptr;
    for (int64_t t = 0; t < b.ntok; ++t) {
        const float* cur = b.cur + (size_t) t * (size_t) g.n_embd;
        float* logits = mb.logits + (size_t) t * (size_t) g.n_expert;
        kernels::glm_f32_gemv(cur, (const float*) w->data, logits, g.n_embd, g.n_expert, stream);
        kernels::glm_router_sigmoid_topk(logits, bias_f, mb.ids + (size_t) t * (size_t) k,
                                         mb.weights + (size_t) t * (size_t) k, g.n_expert, k,
                                         (float) g.expert_weights_scale, stream);
    }
    return true;
}

}  // namespace

// ================================ where a chunk's `pre` time goes ================================

namespace {

// The six sections `pre` is cut into, in the order the boundaries are recorded.  The cutting points are the ones
// a change can act on: the mHC read and write are per-token elementwise work, the mixer and `ffn3` are the
// quantized projections, and the router is the one thing a group entry point would still run per token.
const char* const PRE_SECTION_NAMES[] = {
    "hc_read(attn)",                                  // norm + quantize + the 24-wide projection + Sinkhorn + mix
    "attn: norm, KDA|MLA, hc_write",                  // the attention mixer and the mHC write-back
    "hc_read(ffn)",
    "ffn norm + quantize",
    "ffn3 (dense FFN | shared expert)",
    "router",
};
constexpr int PRE_SECTIONS = (int) (sizeof(PRE_SECTION_NAMES) / sizeof(PRE_SECTION_NAMES[0]));
constexpr int PRE_BOUNDARIES = PRE_SECTIONS + 1;   // one before each section and one after the last

/// The switch, read once.  False on every normal run, which is what keeps `section_mark` down to a `bool` test
/// - the device lookup below is not free and must not happen per `pre` call.
bool sections_wanted() {
    static const bool on = std::getenv("STRATA_GLM_PREFILL_TIME") != nullptr;
    return on;
}

/// One `cudaEvent_t` per boundary per call, so a layer's `T` calls are read together at the layer's sync.  The
/// pool grows to the first chunk that needs it and is then reused; nothing here is thread-safe and a session's
/// layers are strictly sequential, so nothing needs to be.
struct SectionTimer {
    bool on = true;
    std::vector<cudaEvent_t> ev;
    size_t used = 0;            // boundaries recorded since the last flush
    double ms[PRE_SECTIONS]{};
    long long calls = 0;
};

/// **ONE TIMER PER CUDA DEVICE, BECAUSE A LAYER SPLIT IS ONE PROCESS ACROSS SEVERAL CARDS.**  The engine
/// `cudaSetDevice`s per stage, and a `cudaEvent_t` belongs to the context that created it: a single shared pool
/// records the first card's events on the second card's stream, which fails with
/// `cudaErrorInvalidResourceHandle` - and while `section_mark` then turns the timer off rather than report
/// garbage, the failure also LATCHES a CUDA error that the next unrelated `check_launch` prints as if its own
/// kernel had failed.  So the pool is indexed by device; each stage gets its own, and a stage change is
/// invisible to the section accounting either way, since a stage never shares a chunk's boundaries with
/// another.  The events are deliberately never destroyed: a destructor would run after the CUDA context is
/// gone, which is the very error this avoids, and they go with the context anyway.
///
/// The table is `thread_local` on top of that, because the chunk pipeline runs a stage a thread and the lazy
/// `resize` below is not atomic: four stages meeting a chunk for the first time would grow one vector at once
/// and read each other's half-written pointers.  A thread is on one device for its whole life, so a table a
/// thread still gives each stage exactly the one timer it had.
SectionTimer& section_timer() {
    static thread_local std::vector<SectionTimer*> per_device;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0) dev = 0;
    if ((size_t) dev >= per_device.size()) per_device.resize((size_t) dev + 1, nullptr);
    if (per_device[(size_t) dev] == nullptr) per_device[(size_t) dev] = new SectionTimer();
    return *per_device[(size_t) dev];
}

/// Records the next boundary.  Silently does nothing when the switch is off, which is every normal run.
void section_mark(void* stream) {
    if (!sections_wanted()) return;
    SectionTimer& t = section_timer();
    if (!t.on) return;
    if (t.used == t.ev.size()) {
        const size_t grow = t.ev.empty() ? 512 : t.ev.size() * 2;
        t.ev.resize(grow);
        for (size_t i = t.used; i < grow; ++i)
            if (cudaEventCreate(&t.ev[i]) != cudaSuccess) { t.on = false; return; }
    }
    if (cudaEventRecord(t.ev[t.used], (cudaStream_t) stream) != cudaSuccess) { t.on = false; return; }
    ++t.used;
}

}  // namespace

void glm_pre_sections_reset() {
    if (!sections_wanted()) return;
    SectionTimer& t = section_timer();
    t.used = 0;
    t.calls = 0;
    for (double& v : t.ms) v = 0.0;
}

void glm_pre_sections_flush() {
    if (!sections_wanted()) return;
    SectionTimer& t = section_timer();
    if (!t.on || t.used == 0) return;
    const size_t calls = t.used / (size_t) PRE_BOUNDARIES;
    // A partial call - a layer that failed between two marks - is dropped rather than read off the end.
    for (size_t i = 0; i < calls; ++i)
        for (int s = 0; s < PRE_SECTIONS; ++s) {
            float ms = 0.f;
            if (cudaEventElapsedTime(&ms, t.ev[i * PRE_BOUNDARIES + s], t.ev[i * PRE_BOUNDARIES + s + 1]) ==
                cudaSuccess)
                t.ms[s] += ms;
        }
    t.calls += (long long) calls;
    t.used = 0;
}

void glm_pre_sections_report(int64_t tokens, int64_t layers) {
    if (!sections_wanted()) return;
    SectionTimer& t = section_timer();
    if (!t.on) return;
    if (t.calls == 0) {
        std::fprintf(stderr, "strata glm prefill: sections: nothing recorded (%lld layers, %lld tokens)\n",
                     (long long) layers, (long long) tokens);
        return;
    }
    double total = 0.0;
    for (double v : t.ms) total += v;
    // **A CALL IS NOT A TOKEN ONCE `pre` TAKES GROUPS.**  One call covers up to `GLM_MAX_NTOK` tokens, so the
    // per-line figure below is milliseconds a CALL and would read eight times too large if it were labelled a
    // token.  What a caller means by "a token's cost" is this divided by the group width, and the width follows
    // from the totals it handed in: `tokens * layers` is how many layer-tokens those calls covered.  Computed
    // rather than passed, so it cannot disagree with them.
    const double per_call = (t.calls > 0) ? (double) (tokens * layers) / (double) t.calls : 1.0;
    std::fprintf(stderr, "strata glm prefill: sections over %lld pre calls (%.1f tokens each) in %.3f s on the card:\n",
                 t.calls, per_call, total / 1000.0);
    for (int s = 0; s < PRE_SECTIONS; ++s)
        std::fprintf(stderr, "strata glm prefill:   %-38s %7.3f s  %5.1f%%  %7.3f ms a call  %7.3f ms a token\n",
                     PRE_SECTION_NAMES[s], t.ms[s] / 1000.0, total > 0 ? 100.0 * t.ms[s] / total : 0.0,
                     t.ms[s] / (double) t.calls, t.ms[s] / (double) t.calls / per_call);
    // **WHAT THE PROJECTIONS READ, BECAUSE THE SECTION TIMES CANNOT SAY IT.**  The native path takes `ncols`
    // and reads a matrix once; the canonical path loops one column at a time and reads it `ncols` times.  Both
    // land in the same section, and only this line tells them apart - `repeated` is the traffic a batched
    // kernel would have divided by the group width.
    {
        const gemv::ProjStats& p = gemv::proj_stats();
        std::fprintf(stderr,
                     "strata glm prefill:   projections %.0f MiB in %lld calls over %lld columns (%lld native, "
                     "%lld canonical looped over %lld columns), %.0f MiB of it repeated per column\n",
                     (double) p.bytes / 1048576.0, p.calls, p.cols, p.native_calls, p.multi_calls, p.multi_cols,
                     (double) p.multi_bytes / 1048576.0);
        // THE PER-TENSOR TABLE, RANKED BY BYTES, because the aggregate cannot say WHICH projection repeats.
        // A tensor whose `calls/cols` is ~1 was read once per column (a weight re-read per token); one near
        // `1 / group` was batched.  `calls` and `cols` are both printed so the ratio is readable, not derived.
        std::vector<std::pair<std::string, gemv::ProjStats::ByName>> rows(p.by_name.begin(), p.by_name.end());
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.bytes > b.second.bytes; });
        const size_t shown = rows.size() < 14 ? rows.size() : 14;
        for (size_t i = 0; i < shown; ++i) {
            const auto& r = rows[i];
            std::fprintf(stderr,
                         "strata glm prefill:     %-34s %8.1f MiB  %8lld calls  %8lld cols  %5.1f cols a call%s\n",
                         r.first.c_str(), (double) r.second.bytes / 1048576.0, r.second.calls, r.second.cols,
                         (double) r.second.cols / (double) r.second.calls, r.second.native ? "  native" : "");
        }
    }
    std::fflush(stderr);
    glm_pre_sections_reset();
    gemv::proj_stats_reset();
}

// ================================ the block ================================

bool glm_block_layer_pre(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t pos,
                         int32_t pos_base, const GlmBuffers& b, const GlmLayerState& st, const MoEBuffers& mb,
                         int64_t k, const BlockBuffers& bb, void* stream, std::string& err, const Doorbell* db) {
    if (bb.R == nullptr) { err = "glm5-next: the block has no residual"; return false; }
    // **THIS IS NOW THE GROUP ENTRY POINT TOO.**  It used to refuse `ntok > 1`, because the router's
    // `mb.logits`/`ids`/`weights` live in the Qwen path's `MoEBuffers` and were carved at one token.  They are
    // `ntok`-wide in a chunk now - `GlmChunkBuffers` carries the three rows and `glm_chunk_group_view` points a
    // group's `mb` at them - so a caller with a group carve can hand this 1..GLM_MAX_NTOK columns and get the
    // same arithmetic with the weights read once.
    //
    // **WHAT A GROUP CHANGES IS ONLY HOW OFTEN A WEIGHT IS READ.**  Every kernel below takes `b.ntok`; the two
    // norms that used to be written `rows = 1` now take it too.  Nothing here reads another token's column, so
    // a group is the same computation as `ntok` single-token calls, and `--prefill`'s own ids comparison is what
    // checks that rather than this comment.
    if (b.ntok < 1) {
        err = "glm5-next: `glm_block_layer_pre` was handed a zero-token scratch";
        return false;
    }
    (void) k;
    (void) db;

    const LayerView v(tables, layer);
    const int64_t n = g.n_embd;
    const float eps = (float) g.rms_eps;

    // The section boundaries `STRATA_GLM_PREFILL_TIME` reports.  Each is a `cudaEventRecord` and nothing else -
    // no sync, no wait - so the numbers are the card's own, and the switch costs a normal run nothing.
    section_mark(stream);

    // ---- the attention half ------------------------------------------------------------------------------
    if (!hc_read(tables, g, layer, "hc_attn_fn.weight", "hc_attn_base.weight", "hc_attn_scale.weight", bb.R, b, stream,
                 err)) {
        return false;
    }
    section_mark(stream);
    const WeightRef* w_attn_norm = req(v, "attn_norm.weight", err);
    if (w_attn_norm == nullptr) return false;
    if (w_attn_norm->bytes < (uint64_t) n * 4) {
        err = v.name("attn_norm.weight") + ": not f32 of the hidden width";
        return false;
    }
    kernels::rms_norm_weighted(b.cur, (const float*) w_attn_norm->data, b.ntok, n, eps, stream);
    quantize_both(b.cur, b.cur_q8k, b.cur_q8_0, n * b.ntok, stream);
    kernels::f32_to_bf16_bulk(b.cur, b.cur_bf16, n * b.ntok, stream);

    // WHICH MIXER IS A PROPERTY OF THE LAYER, never of a tensor name: `attn_output.weight` is KDA's output
    // projection on 34 layers and MLA's on 11, and reading the wrong one gives a right-shaped wrong answer.
    //
    // `pos + pos_base` is the ABSOLUTE position and it is the only thing MLA needs it for - KDA is a
    // recurrence and needs no position at all, which is why `pos` reaches this function only to be added.
    if (glm_is_kda_layer(g, layer)) {
        GlmLayerState& st_mut = const_cast<GlmLayerState&>(st);   // the conv and delta states are updated in place
        if (!kda_layer(tables, g, layer, b, st_mut, stream, err)) return false;
    } else if (!mla_layer(tables, g, layer, (int64_t) pos_base + pos, b, st, stream, err)) {
        return false;
    }

    hc_write(g, b.attn_out, b, bb.R, stream);
    if (!hc_write_commit(bb.R, b, g, stream, err)) return false;
    section_mark(stream);

    // ---- the FFN half ------------------------------------------------------------------------------------
    if (!hc_read(tables, g, layer, "hc_ffn_fn.weight", "hc_ffn_base.weight", "hc_ffn_scale.weight", bb.R, b, stream,
                 err)) {
        return false;
    }
    section_mark(stream);
    const WeightRef* w_ffn_norm = req(v, "ffn_norm.weight", err);
    if (w_ffn_norm == nullptr) return false;
    if (w_ffn_norm->bytes < (uint64_t) n * 4) {
        err = v.name("ffn_norm.weight") + ": not f32 of the hidden width";
        return false;
    }
    kernels::rms_norm_weighted(b.cur, (const float*) w_ffn_norm->data, b.ntok, n, eps, stream);
    quantize_both(b.cur, b.cur_q8k, b.cur_q8_0, n * b.ntok, stream);
    kernels::f32_to_bf16_bulk(b.cur, b.cur_bf16, n * b.ntok, stream);
    section_mark(stream);

    if (g.is_dense_ffn_layer(layer)) {
        // DENSE: the whole FFN runs here, straight into `bb.block_out`, which is where `hc_post` reads the
        // sublayer's result from.  A buffer of its own would be one more copy of 16 KB per layer for nothing.
        if (!ffn3(tables, layer, b, "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", n, g.n_ff_dense,
                  g.swiglu_limit_shexp_or_off(), bb.block_out, stream, err))
            return false;
        // ...AND THE RESIDUAL WRITE TOO.  A dense layer has no host pool between the halves, so the block is
        // not actually split and `post` has nothing left to do.
        hc_write(g, bb.block_out, b, bb.R, stream);
        const bool ok = hc_write_commit(bb.R, b, g, stream, err);
        section_mark(stream);   // and the last one again: a dense layer has no router, and the pool per call is fixed
        section_mark(stream);
        return ok;
    }

    // MOE: `pre` ends at the SHARED EXPERT AND THE ROUTER, and the residual write happens in `post` - because
    // between the two halves the host runs the routed experts into `parts`, and `hc_post` cannot run until
    // every part of the sublayer's output exists.  The shared expert runs HERE rather than in `post` for the
    // first family's reason: it depends only on `cur`, so it is GPU work that overlaps the CPU pool.
    //
    // **THE SHARED EXPERT IS A PLAIN, UNWEIGHTED ADD.**  glm5-next has no `ffn_gate_inp_shexp` tensor at all -
    // the gate that scales qwen4exp's shared expert does not exist here - and `moe_combine_parts` adds
    // `mb.shared` unconditionally, which is exactly this family's rule.
    if (!ffn3(tables, layer, b, "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight", n, g.n_ff,
              g.swiglu_limit_shexp_or_off(), mb.shared, stream, err))
        return false;
    section_mark(stream);
    const bool ok = glm_router(tables, g, layer, b, mb, k, stream, err);
    section_mark(stream);
    return ok;
}

bool glm_block_layer_post(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t k,
                          const GlmBuffers& b, const MoEBuffers& mb, const BlockBuffers& bb, const float* parts,
                          void* stream, std::string& err) {
    (void) tables;
    if (g.is_dense_ffn_layer(layer)) {
        // The dense FFN ran inside `pre` and left its output in `bb.block_out`; the FFN half's `post`/`comb`
        // are still in `b`, which is why they are in `GlmBuffers` and not in the shared block scratch.
        (void) k;
        (void) mb;
        (void) parts;
        (void) stream;
        return true;
    }
    if (parts == nullptr) { err = "glm5-next: a MoE layer needs the routed experts, and `parts` is null"; return false; }
    if (bb.block_out == nullptr) { err = "glm5-next: the block has no output buffer"; return false; }
    // THE COMBINATION IS THE FIRST FAMILY'S, because the rule is: `sum_j w[j]*parts[j] + shared`, and that is
    // what `moe_combine` computes.  The differences are all upstream - where `w` came from (a sigmoid router
    // with a 2.5 scale) and what `shared` is (an ungated add) - so reusing it here is not a shortcut, it is the
    // same arithmetic reached from a different router.
    if (!moe_combine_parts(g, layer, k, mb, parts, bb.block_out, stream, err)) return false;
    hc_write(g, bb.block_out, b, bb.R, stream);
    return hc_write_commit(bb.R, b, g, stream, err);
}

bool glm_block_layer(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t pos,
                     int32_t pos_base, const GlmBuffers& b, const GlmLayerState& st, const MoEBuffers& mb, int64_t k,
                     const BlockBuffers& bb, const float* parts, void* stream, std::string& err,
                     const Doorbell* db) {
    if (!glm_block_layer_pre(tables, g, layer, pos, pos_base, b, st, mb, k, bb, stream, err, db)) return false;
    return glm_block_layer_post(tables, g, layer, k, b, mb, bb, parts, stream, err);
}

bool glm_head_mix(const WeightTable& tables, const ModelGeometry& g, const BlockBuffers& bb, float* out, void* stream,
                  std::string& err) {
    if (bb.R == nullptr) { err = "glm5-next: the head has no residual to collapse"; return false; }
    kernels::GlmHcShapes hcs;
    hcs.n_embd = g.n_embd;
    hcs.hc = g.hc;
    hcs.mix = g.hc_mix;
    hcs.sinkhorn_iters = 20;
    // The MEAN of the streams, not their sum.  A sum is 4x the model's activation scale and still produces
    // text - a confident, washed-out, wrong text.
    kernels::glm_hc_sum(bb.R, out, hcs, 1, stream);
    // THEN `output_norm`, and it is a SEPARATE TENSOR FROM THE FIRST FAMILY'S `output_hc_norm`.  That one is a
    // learned norm over the hc stack inside `gr_read`; this one is a plain RMSNorm over the collapsed single
    // vector, and there is no `gr_read` at the head here to carry it.  Applying only the mean and projecting
    // straight from it drops a per-channel rescale that the model was trained with - finite text, wrong text.
    const WeightRef* wn = tables.find("output_norm.weight");
    if (wn == nullptr) { err = "output_norm.weight is missing"; return false; }
    if (wn->bytes < (uint64_t) g.n_embd * 4) { err = "output_norm.weight is not f32 of the hidden width"; return false; }
    kernels::rms_norm_weighted(out, (const float*) wn->data, 1, g.n_embd, (float) g.rms_eps, stream);
    return true;
}

// ================================ the MTP (draft) block ================================

namespace {

/// The number of shards a `cudaMemcpyAsync` of `n` floats is - written once because the draft block does four
/// of them and a wrong `cudaMemcpyKind` is a silently-free memcpy on unified memory and a crash anywhere else.
void copy_dev(float* dst, const float* src, int64_t n, void* stream) {
    cudaMemcpyAsync(dst, src, (size_t) n * 4, cudaMemcpyDeviceToDevice, (cudaStream_t) stream);
}

}  // namespace

uint64_t glm_mtp_state_bytes(const ModelGeometry& g, int64_t max_cells) {
    if (g.n_nextn <= 0) return 0;
    // Three hidden-width vectors (`emb`, `hstate`, `inp`) and one double-width one (`cat`), which is what
    // `eh_proj`'s two halves are concatenated into.  Nothing here is per-cell: the draft block runs one
    // position at a time, so the only thing that scales with the context is its own MLA cache.
    const uint64_t one = align16((uint64_t) g.n_embd * 4);
    return glm_mla_state_bytes(g, max_cells) + one * 3 + align16((uint64_t) g.n_embd * 8);
}

uint64_t glm_mtp_state_init(const ModelGeometry& g, int64_t max_cells, void* base, GlmMtpState& st) {
    st.max_cells = 0;
    st.emb = nullptr;
    st.hstate = nullptr;
    st.cat = nullptr;
    st.inp = nullptr;
    if (g.n_nextn <= 0) return 0;
    uint8_t* p = (uint8_t*) base;
    p += glm_mla_state_init(g, max_cells, p, st.attn);
    const uint64_t one = align16((uint64_t) g.n_embd * 4);
    st.emb = (float*) p;
    p += one;
    st.hstate = (float*) p;
    p += one;
    st.inp = (float*) p;
    p += one;
    st.cat = (float*) p;
    p += align16((uint64_t) g.n_embd * 8);
    st.max_cells = max_cells;
    st.n_written = 0;
    return glm_mtp_state_bytes(g, max_cells);
}

bool glm_mtp_step(const WeightTable& tables, const ModelGeometry& g, const GlmBuffers& b, const MoEBuffers& mb,
                  const BlockBuffers& bb, GlmMtpState& st, int64_t token, const float* hidden, int64_t pos,
                  int64_t k, GlmPoolFn pool, void* pool_user, const NativeHead* head, const float* logits,
                  void* stream, std::string& err) {
    if (g.n_nextn <= 0) {
        err = "glm5-next: this model declares no block past the trunk (nextn_predict_layers is 0)";
        return false;
    }
    if (st.attn.mla_cache == nullptr || st.emb == nullptr || st.cat == nullptr) {
        err = "glm5-next: the MTP block's state was never carved";
        return false;
    }
    if (hidden == nullptr) { err = "glm5-next: the MTP block was handed no hidden state"; return false; }
    // **THIS STEP IS ONE TOKEN, AND IT SAYS SO RATHER THAN LOOKING GROUP-READY.**  Unlike the trunk block, the
    // whole of it is single-token by construction - `embed_row` of one `token`, the block's own `st.emb`/`st.cat`
    // scratch, `copy_dev(b.cur, st.inp, n)` - so the `1`s in its norms below are not an oversight to be
    // "fixed" by writing `b.ntok`.  Doing that would normalize the first token and leave the rest of the group
    // reading whatever the previous call left in `st.*`, which is the kind of wrong answer that still generates.
    // The guard is here because a GROUP carve is now a `GlmBuffers` with `ntok == 8` that any caller could hand
    // this by mistake.  Making the block take a group is its own change: every `st.*` field above would need a
    // per-token row and `mla_layer`'s `n_written` a row per token, not a count.
    if (b.ntok != 1) {
        err = "glm5-next: the MTP block is a single-token step and was handed a " + std::to_string(b.ntok) +
              "-token scratch";
        return false;
    }

    // **THE DRAFT BLOCK'S OWN INDEX IS `n_layers`, WHICH IS WHERE THE TRUNK STOPS.**  `n_layers` is
    // `block_count - nextn_predict_layers` (model_arch.cpp), so on the shipped 46-block artifact this is 45 -
    // the block's real block index, which is what the pack names its tensors after and what keeps every
    // `expert_layout` row a BLOCK row.  It is emphatically NOT a trunk layer: nothing in the layer loop, the
    // split search or the state carve may be handed this number.
    const int64_t layer = g.n_layers;
    const LayerView v(tables, layer);
    const int64_t n = g.n_embd;
    const float eps = (float) g.rms_eps;

    const WeightRef* w_enorm = req(v, "nextn.enorm.weight", err);
    const WeightRef* w_hnorm = req(v, "nextn.hnorm.weight", err);
    const WeightRef* w_eh = req(v, "nextn.eh_proj.weight", err);
    const WeightRef* w_shn = req(v, "nextn.shared_head_norm.weight", err);
    const WeightRef* w_attn_norm = req(v, "attn_norm.weight", err);
    const WeightRef* w_ffn_norm = req(v, "ffn_norm.weight", err);
    if (!w_enorm || !w_hnorm || !w_eh || !w_shn || !w_attn_norm || !w_ffn_norm) return false;
    if (w_attn_norm->bytes < (uint64_t) n * 4 || w_ffn_norm->bytes < (uint64_t) n * 4 ||
        w_enorm->bytes < (uint64_t) n * 4 || w_hnorm->bytes < (uint64_t) n * 4 || w_shn->bytes < (uint64_t) n * 4) {
        err = v.name("nextn.shared_head_norm.weight") + " and the block's other norms are not f32 of the hidden width";
        return false;
    }

    // ---- 1. the block's input: `eh_proj(cat(enorm(emb(x) * clamp(pos,0,1)), hnorm(h)))`.
    //
    // **THE POSITION MASK IS APPLIED TO THE EMBEDDING AND BEFORE `enorm`, AND IT ZEROES EXACTLY ROW 0.**  The
    // reference multiplies the gathered embedding by `clamp(pos, 0, 1)`, so the first cell of a sequence
    // contributes an `enorm(0)` constant and no token at all - there is no next token for it to predict from.
    // Applying the mask after `enorm` is the same number of characters and a different vector.
    if (!embed_row(tables, g, token, st.emb, stream, err)) return false;
    if (pos <= 0) kernels::scale_inplace(st.emb, n, 0.0f, stream);
    kernels::rms_norm_weighted(st.emb, (const float*) w_enorm->data, 1, n, eps, stream);
    copy_dev(st.hstate, hidden, n, stream);
    kernels::rms_norm_weighted(st.hstate, (const float*) w_hnorm->data, 1, n, eps, stream);
    // `enorm(emb)` OCCUPIES THE FIRST HALF.  `eh_proj` is [8192, 4096], so both halves are the same width and
    // the swap is shape-legal - it loads, it runs, and it is a different model.
    copy_dev(st.cat, st.emb, n, stream);
    copy_dev(st.cat + n, st.hstate, n, stream);
    quantize_both(st.cat, b.wide_q8k, b.wide_q8_0, 2 * n, stream);
    kernels::f32_to_bf16_bulk(st.cat, b.wide_bf16, 2 * n, stream);
    if (!project(*w_eh, v.name("nextn.eh_proj.weight"), st.cat, b.wide_q8_0, b.wide_q8k, b.wide_bf16, st.inp, 2 * n,
                 n, /*ntok=*/1, stream, err))
        return false;

    // ---- 2. the attention half.  `inpSA = cur` and the MLA helper applies `attn_norm` itself, so the block's
    //         own norm goes on HERE and the value it replaces is `st.inp`, which is what the skip adds back.
    //         The block has NO `hc_*` tensors: there is no mHC read and no mHC write anywhere in it.
    copy_dev(b.cur, st.inp, n, stream);
    kernels::rms_norm_weighted(b.cur, (const float*) w_attn_norm->data, 1, n, eps, stream);
    quantize_both(b.cur, b.cur_q8k, b.cur_q8_0, n, stream);
    kernels::f32_to_bf16_bulk(b.cur, b.cur_bf16, n, stream);
    // The layer index reaches `mla_layer` only through `blk.<layer>.` and the per-block tensor names; every
    // shape it reads comes from `g`, which is the same geometry the trunk's MLA layers use.
    //
    // **`pos` IS THE ROPE POSITION AND `n_written` IS THE CACHE ROW, AND THEY PART COMPANY ON THE FIRST DRAFT.**
    // The block is not run at every position of the sequence - the trunk is - so its cache holds one row per
    // step it was called for, and the row a step goes into is the count of the steps before it.  Attending over
    // `pos + 1` rows instead (the dense reading, which is right for a trunk layer and wrong here) would walk
    // rows this block never wrote: on a 5-token prompt the first draft would attend over rows 0..5 with 0..4
    // holding the zero-filled carving, and a zeroed MLA row is not a masked row - K and V are both 0, so its
    // score is exactly 0 and softmax gives it weight `exp(0) = 1` against the one real row's `exp(s)`.  That is
    // a 5/6 dilution of the only row that should count.
    if (!mla_layer(tables, g, layer, pos, b, st.attn, stream, err, st.n_written, st.n_written + 1)) return false;
    st.n_written++;
    // **`glm_add_inplace` IS `dst += src`, AND THIS CALL HAD ITS ARGUMENTS THE OTHER WAY ROUND.**  It read
    // `st.inp = attn(x) + inpSA`, which is what the comment said and what the line did NOT do: written the
    // other way it made `b.attn_out += st.inp` and left `st.inp` at `inpSA`, so the block's attention result
    // was thrown away one line later (`moe_combine_parts` overwrites `b.attn_out`) and the FFN was fed the
    // WRONG vector.  The block then still emits a confident, plausible token - it is an FFN over `inpSA` -
    // which is why the draft was wrong and not empty.  MEASURED: with the attention's output zeroed by hand
    // the drafted logits were bit-identical, which is what a dropped term looks like.
    kernels::glm_add_inplace(st.inp, b.attn_out, n, stream);   // st.inp = inpSA + attn(x) = ffn_inp

    // ---- 3. the FFN half: `moe(rms(ffn_norm(ffn_inp))) + shexp(rms(ffn_norm(ffn_inp))) + ffn_inp`.
    copy_dev(b.cur, st.inp, n, stream);
    kernels::rms_norm_weighted(b.cur, (const float*) w_ffn_norm->data, 1, n, eps, stream);
    quantize_both(b.cur, b.cur_q8k, b.cur_q8_0, n, stream);
    kernels::f32_to_bf16_bulk(b.cur, b.cur_bf16, n, stream);
    if (!ffn3(tables, layer, b, "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight", n, g.n_ff,
              g.swiglu_limit_shexp_or_off(), mb.shared, stream, err))
        return false;
    if (!glm_router(tables, g, layer, b, mb, k, stream, err)) return false;

    // ---- the routed experts, on the host, exactly as `session_token` runs a trunk MoE layer's.
    if (pool == nullptr) {
        err = "glm5-next: the MTP block's MoE needs the expert pool, and the hook is null";
        return false;
    }
    st.h_x.resize((size_t) n);
    st.h_ids.resize((size_t) k);
    st.h_out.resize((size_t) k * (size_t) n);
    cudaStream_t cs = (cudaStream_t) stream;
    // The same handoff `session_token` makes between a block's halves: two async copies onto THIS stream, then
    // a wait.  Only the activation and the ids go over - the router's WEIGHTS stay on the device, because
    // `moe_combine_parts` is what multiplies by them and the pool's output is deliberately unweighted.
    if (cudaMemcpyAsync(st.h_x.data(), b.cur, (size_t) n * 4, cudaMemcpyDeviceToHost, cs) != cudaSuccess ||
        cudaMemcpyAsync(st.h_ids.data(), mb.ids, (size_t) k * 4, cudaMemcpyDeviceToHost, cs) != cudaSuccess) {
        err = "glm5-next: staging the MTP block's expert handoff: " + std::string(cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    if (cudaStreamSynchronize(cs) != cudaSuccess) {
        err = "glm5-next: waiting for the MTP block's expert handoff";
        return false;
    }
    if (!pool(pool_user, layer, st.h_x.data(), st.h_ids.data(), /*nt=*/1, k, st.h_out.data(), err)) return false;
    if (cudaMemcpyAsync(bb.block_out, st.h_out.data(), (size_t) k * (size_t) n * 4, cudaMemcpyHostToDevice,
                        cs) != cudaSuccess) {
        err = "glm5-next: staging the MTP block's expert results: " + std::string(cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    // `moe_combine_parts` writes `sum_j w[j]*parts[j] + shared` - the shared expert AND the routed experts
    // together, which is why `ffn3` above wrote `mb.shared` and this reads it.  It lands in `b.attn_out`, which
    // the skip below consumed a moment ago: `parts` and `out` must not be the same buffer.
    if (!moe_combine_parts(g, layer, k, mb, bb.block_out, b.attn_out, stream, err)) return false;

    // ---- 4. the second skip, the head norm, and the model's own output head.  `ffn_inp` is `st.inp`.
    kernels::glm_add_inplace(b.attn_out, st.inp, n, stream);
    copy_dev(bb.mixed, b.attn_out, n, stream);
    kernels::rms_norm_weighted(bb.mixed, (const float*) w_shn->data, 1, n, eps, stream);
    // THE HEAD IS THE TRUNK'S OWN, AND SO IS THE BRANCH.  `bb.mixed` is what `glm_head_mix` writes and what
    // `lm_head_project` reads, so with `--native` - where `output.weight` is a non-resident row served only by
    // `NativeHead` - the canonical call has no planes to read.  This mirrors `run_head` exactly.
    if (head != nullptr && head->loaded()) return head->run(bb.mixed, const_cast<float*>(logits), stream, err);
    return lm_head_project(tables, g, bb, const_cast<float*>(logits), stream, err);
}

}  // namespace strata::core
