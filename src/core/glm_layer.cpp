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
#include "strata/kernels/quantize_act.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <string>

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

/// **EVERY FIELD IN `GlmBuffers` IS PER TOKEN, SO THE WHOLE CARVE SCALES WITH `ntok`.**  There is no field in
/// the list below that is shared across a chunk: the mHC maps, the normed input, the wide image pair, the KDA
/// streams, the MLA latent and head stacks and the FFN pair are each one token's worth of scratch, written by a
/// projection whose input in a chunk is `ntok` columns.  `glm_buffers_bytes(g, 1)` is what the decode path
/// carves and it is byte for byte the shape that was there before the chunk existed.
///
/// The slot is `ntok * align16(one token's field)` rather than `align16(ntok * field)` so that this function
/// and `glm_buffers_init` can share one list and cannot drift: `init` takes the same list through the same
/// `slot()`.  Either spelling is a valid layout - nothing here needs the field to be `ntok`-contiguous at a
/// boundary - and the one that is checkable against a single list is the one to keep.
uint64_t glm_buffers_bytes(const ModelGeometry& g, int64_t ntok) {
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
        (uint64_t) g.n_head * g.kv_lora_rank * 4,               // qabs
        (uint64_t) g.kv_lora_rank * 4,                          // kv_cmpr
        (uint64_t) g.n_head * g.kv_lora_rank * 4,               // kqv
        (uint64_t) g.n_head * g.mla_head_dim * 4,               // head_out
        (uint64_t) fw * 4,                                      // ffn_gate
        (uint64_t) fw * 4,                                      // ffn_up
    };
    uint64_t total = 0;
    for (uint64_t v : parts) total += (uint64_t) ntok * align16(v);
    return total;
}

uint64_t glm_buffers_init(const ModelGeometry& g, int64_t ntok, void* base, GlmBuffers& b) {
    const int64_t n = g.n_embd;
    const int64_t hc = g.hc;
    const int64_t nv = kda_n_v(g);
    const int64_t fw = ffn_width(g);
    const int64_t ww = wide_width(g);

    Arena a{(uint8_t*) base, 0};
    // The SAME slot rule `glm_buffers_bytes` uses, applied in one place so the two lists cannot disagree.
    const auto take = [&](uint64_t one_token) -> void* {
        return a.take((uint64_t) ntok * align16(one_token));
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
    b.qabs = (float*) take((uint64_t) g.n_head * g.kv_lora_rank * 4);
    b.kv_cmpr = (float*) take((uint64_t) g.kv_lora_rank * 4);
    b.kqv = (float*) take((uint64_t) g.n_head * g.kv_lora_rank * 4);
    b.head_out = (float*) take((uint64_t) g.n_head * g.mla_head_dim * 4);
    b.ffn_gate = (float*) take((uint64_t) fw * 4);
    b.ffn_up = (float*) take((uint64_t) fw * 4);

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
    c.n_embd = n;
    c.hc = hc;
    c.k = k;
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
    out_bb.R = c.R + (size_t) t * hc * n;
    out_bb.block_out = c.block_out + (size_t) t * n;
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

uint64_t glm_layer_state_bytes(const ModelGeometry& g, int64_t max_cells, int64_t layer) {
    return glm_is_kda_layer(g, layer) ? glm_kda_state_floats(g) * 4 : glm_mla_cache_bytes(g, max_cells);
}

uint64_t glm_state_init(const ModelGeometry& g, int64_t max_cells, int64_t layer, void* base, GlmLayerState& st) {
    st.kda_state = nullptr;
    st.kda_conv = nullptr;
    st.mla_cache = nullptr;
    st.max_cells = 0;
    if (!glm_is_kda_layer(g, layer)) {
        st.mla_cache = (uint16_t*) base;
        st.max_cells = max_cells;
        return glm_mla_cache_bytes(g, max_cells);
    }
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
/// `abs_pos` is the ABSOLUTE position of this token, and it is both the cache row to write and the last cell
/// the query may attend to.  There is no window and no ring here: the cache is indexed by position.
bool mla_layer(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t abs_pos,
               const GlmBuffers& b, const GlmLayerState& st, void* stream, std::string& err) {
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
    //      **THE QUANTIZE IS NOT PER HEAD, THE PROJECTION IS.**  The `n_head` queries are contiguous in
    //      `qfull`, and a head's slice is a whole number of 256-element Q8_K blocks and 32-element Q8_0 ones,
    //      so one quantize of the stack produces exactly the blocks `n_head` separate quantizes did, at
    //      offsets `q8k_bytes(hd) * h`.  The projection still has to be per head - each head has its own
    //      input and its own band of the folded matrix - but the quantize was 2 launches a head of a nearly
    //      empty kernel (8 active threads of 128, doing 32 double divisions each).
    //
    //      A CHUNK MULTIPLIES THE OUTER LOOP AND NOT THE INNER ONE.  Head `h` of token `t` is still its own
    //      `project_rows` call - the band is the same weight rows for every token, but the input is a
    //      different `hd`-wide slice of `qfull`, so there is nothing to share between them short of an
    //      h-major `qfull`, which the projection above cannot produce.  What the chunk does buy here is the
    //      one quantize of the whole `ntok x (nh*hd)` stack, which is `2` launches instead of `2 * ntok`.
    quantize_both(b.qfull, b.heads_q8k, b.heads_q8_0, nh * hd * b.ntok, stream);
    const int64_t hd_q8k = (int64_t) q8k_bytes(hd), hd_q8_0 = (hd / 32) * 34;
    const int64_t full_q8k = (int64_t) q8k_bytes(nh * hd), full_q8_0 = ((nh * hd) / 32) * 34;
    for (int64_t t = 0; t < b.ntok; ++t) {
        const float* tfull = b.qfull + t * nh * hd;
        const uint8_t* tq8_0 = b.heads_q8_0 + t * full_q8_0;
        const uint8_t* tq8k = b.heads_q8k + t * full_q8k;
        float* tqabs = b.qabs + t * nh * kvl;
        for (int64_t h = 0; h < nh; ++h) {
            const float* xh = tfull + h * hd;
            if (!project_rows(*wk_b, v.name("attn_k_b.weight"), xh, tq8_0 + h * hd_q8_0, tq8k + h * hd_q8k,
                              tqabs + h * kvl, hd, h * kvl, kvl, stream, err)) {
                return false;
            }
        }
    }

    // ---- 4. the latent, its norm, and the cache write.  K == V, so there is one write and no second tensor.
    if (!project(*wkv_a, v.name("attn_kv_a_mqa.weight"), b.cur, b.cur_q8_0, b.cur_q8k, b.cur_bf16, b.kv_cmpr, n,
                 kvl, b.ntok, stream, err))
        return false;
    kernels::rms_norm_weighted(b.kv_cmpr, (const float*) wkv_a_norm->data, b.ntok, kvl, eps, stream);
    if (st.mla_cache == nullptr) { err = "glm5-next: the MLA latent cache was never carved"; return false; }
    if (abs_pos < 0 || abs_pos + b.ntok > st.max_cells) {
        err = "glm5-next: MLA positions " + std::to_string((long long) abs_pos) + ".." +
              std::to_string((long long) (abs_pos + b.ntok - 1)) + " are outside the " +
              std::to_string((long long) st.max_cells) + "-row latent cache";
        return false;
    }
    // **ALL `ntok` ROWS GO IN BEFORE ANY QUERY RUNS, AND THAT IS CAUSAL ANYWAY.**  The attention below masks by
    // `n_kv` - query `t` walks `0 .. abs_pos + t` and never looks at the rows above it - so a row written early
    // is a row no query in this chunk asks for.  Ordering the write inside the token loop instead would cost a
    // second launch per token to hide data that is already hidden.
    kernels::glm_mla_cache_store_t(b.kv_cmpr, st.mla_cache, abs_pos, kvl, b.ntok, stream);

    // ---- 5. the attention itself: every cell up to and including this one, one query, one token.  `T` is 1,
    //      not the sequence length - the cache holds the sequence and the kernel walks it.  A chunk calls it
    //      `ntok` times, each with its own `pos_base` and its own `n_kv`, because the mask is a function of the
    //      token's absolute position and the kernel takes it as a launch argument.
    for (int64_t t = 0; t < b.ntok; ++t) {
        kernels::glm_mla_attn(b.qabs + t * nh * kvl, st.mla_cache, b.kqv + t * nh * kvl, nh, kvl, abs_pos + t + 1,
                              /*T=*/1, abs_pos + t, (float) (1.0 / std::sqrt((double) hd)), stream);
    }

    // ---- 6. de-absorption.  The same fold as step 3, and here the slices are `kv_lora_rank` wide - still a
    //      whole number of blocks, which is the property that lets one quantize stand in for `n_head`.
    quantize_both(b.kqv, b.heads_q8k, b.heads_q8_0, nh * kvl * b.ntok, stream);
    const int64_t kvl_q8k = (int64_t) q8k_bytes(kvl), kvl_q8_0 = (kvl / 32) * 34;
    const int64_t kqv_q8k = (int64_t) q8k_bytes(nh * kvl), kqv_q8_0 = ((nh * kvl) / 32) * 34;
    for (int64_t t = 0; t < b.ntok; ++t) {
        const float* tkqv = b.kqv + t * nh * kvl;
        const uint8_t* tq8_0 = b.heads_q8_0 + t * kqv_q8_0;
        const uint8_t* tq8k = b.heads_q8k + t * kqv_q8k;
        float* thout = b.head_out + t * nh * hd;
        for (int64_t h = 0; h < nh; ++h) {
            const float* xh = tkqv + h * kvl;
            if (!project_rows(*wv_b, v.name("attn_v_b.weight"), xh, tq8_0 + h * kvl_q8_0, tq8k + h * kvl_q8k,
                              thout + h * hd, kvl, h * hd, hd, stream, err)) {
                return false;
            }
        }
    }

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
    kernels::glm_f32_gemv(b.cur, (const float*) w->data, mb.logits, g.n_embd, g.n_expert, stream);

    // `exp_probs_b` steers the SELECTION and is then dropped: the weights are the un-biased probabilities.  A
    // port that biases the weights too still routes to the right experts and scales them wrongly.
    const WeightRef* bias = v.get("exp_probs_b.bias");
    if (bias != nullptr && bias->bytes < (uint64_t) g.n_expert * 4) {
        err = v.name("exp_probs_b.bias") + ": not f32 of the expert count";
        return false;
    }
    kernels::glm_router_sigmoid_topk(mb.logits, bias != nullptr ? (const float*) bias->data : nullptr, mb.ids,
                                     mb.weights, g.n_expert, k, (float) g.expert_weights_scale, stream);
    return true;
}

}  // namespace

// ================================ the block ================================

bool glm_block_layer_pre(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t pos,
                         int32_t pos_base, const GlmBuffers& b, const GlmLayerState& st, const MoEBuffers& mb,
                         int64_t k, const BlockBuffers& bb, void* stream, std::string& err, const Doorbell* db) {
    if (bb.R == nullptr) { err = "glm5-next: the block has no residual"; return false; }
    // **THE GROUP ENTRY POINT IS NOT THIS ONE YET.**  Everything this calls is `ntok`-aware, but the two things
    // that are still per token - the router's `mb.logits`/`ids`/`weights` and `moe_combine_parts` - live in
    // `MoEBuffers`, which the Qwen path shares and which is carved at one token.  Rather than let a `ntok > 1`
    // carve reach them and quietly compute the first token's route for all of them, it is refused here until
    // `glm_block_group_pre` gives the group its own MoE scratch.
    if (b.ntok != 1) {
        err = "glm5-next: `glm_block_layer_pre` is the single-token entry point and was handed a "
              "multi-token scratch";
        return false;
    }
    (void) mb;
    (void) k;
    (void) db;

    const LayerView v(tables, layer);
    const int64_t n = g.n_embd;
    const float eps = (float) g.rms_eps;

    // ---- the attention half ------------------------------------------------------------------------------
    if (!hc_read(tables, g, layer, "hc_attn_fn.weight", "hc_attn_base.weight", "hc_attn_scale.weight", bb.R, b, stream,
                 err)) {
        return false;
    }
    const WeightRef* w_attn_norm = req(v, "attn_norm.weight", err);
    if (w_attn_norm == nullptr) return false;
    if (w_attn_norm->bytes < (uint64_t) n * 4) {
        err = v.name("attn_norm.weight") + ": not f32 of the hidden width";
        return false;
    }
    kernels::rms_norm_weighted(b.cur, (const float*) w_attn_norm->data, 1, n, eps, stream);
    quantize_both(b.cur, b.cur_q8k, b.cur_q8_0, n, stream);
    kernels::f32_to_bf16_bulk(b.cur, b.cur_bf16, n, stream);

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

    // ---- the FFN half ------------------------------------------------------------------------------------
    if (!hc_read(tables, g, layer, "hc_ffn_fn.weight", "hc_ffn_base.weight", "hc_ffn_scale.weight", bb.R, b, stream,
                 err)) {
        return false;
    }
    const WeightRef* w_ffn_norm = req(v, "ffn_norm.weight", err);
    if (w_ffn_norm == nullptr) return false;
    if (w_ffn_norm->bytes < (uint64_t) n * 4) {
        err = v.name("ffn_norm.weight") + ": not f32 of the hidden width";
        return false;
    }
    kernels::rms_norm_weighted(b.cur, (const float*) w_ffn_norm->data, 1, n, eps, stream);
    quantize_both(b.cur, b.cur_q8k, b.cur_q8_0, n, stream);
    kernels::f32_to_bf16_bulk(b.cur, b.cur_bf16, n, stream);

    if (g.is_dense_ffn_layer(layer)) {
        // DENSE: the whole FFN runs here, straight into `bb.block_out`, which is where `hc_post` reads the
        // sublayer's result from.  A buffer of its own would be one more copy of 16 KB per layer for nothing.
        if (!ffn3(tables, layer, b, "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", n, g.n_ff_dense,
                  g.swiglu_limit_shexp_or_off(), bb.block_out, stream, err))
            return false;
        // ...AND THE RESIDUAL WRITE TOO.  A dense layer has no host pool between the halves, so the block is
        // not actually split and `post` has nothing left to do.
        hc_write(g, bb.block_out, b, bb.R, stream);
        return hc_write_commit(bb.R, b, g, stream, err);
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
    return glm_router(tables, g, layer, b, mb, k, stream, err);
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

}  // namespace strata::core
