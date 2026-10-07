// include/strata/glm/kernels.hpp - the CPU kernels of the GLM-5.3 forward pass.
//
// THE WEIGHT FORMS (docs/GLM53.md, measured):
//   Q4   int4, groups of 64 along the input: byte j of a row holds element 2j in the low nibble and 2j+1 in the
//        high nibble, stored as q + 8; one f32 scale per group.  value = (code - 8) * scale.
//   Q8R  int8 rows (embed_tokens, lm_head): one SIGNED byte per element (two's complement in a U8 tensor), one f32
//        scale per row.  value = (int8_t) code * scale - colibri's fmt 1, which is what wrote this container.
//
// THE ACTIVATION TRICK.  A group's low nibbles are the EVEN elements and its high nibbles the ODD ones, so instead
// of interleaving nibbles per weight (a shuffle per 16 weights, on every row), the activation is permuted ONCE per
// call into, per group of 64, [x0 x2 .. x62 | x1 x3 .. x63] (`Act::xp`).  The bias is folded out the same way:
// sum (q - 8) x = sum q x - 8 * sum x, with the group sum of x computed once per call (`Act::gsum`).  Both are
// exact reorderings of the same products; only the float summation order differs from a row-at-a-time loop.
#pragma once

#include "strata/glm/pool.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace strata::glm {

constexpr int kGroup = 64;

struct Q4 {
    int O = 0, I = 0;                  ///< rows (outputs) x cols (inputs); I % 64 == 0
    const uint8_t* codes = nullptr;    ///< O * I / 2 bytes
    const float* scales = nullptr;     ///< O * I / 64 floats
    const uint8_t* row_codes(int r) const { return codes + (size_t) r * (I / 2); }
    const float* row_scales(int r) const { return scales + (size_t) r * (I / kGroup); }
};

struct Q8R {
    int O = 0, I = 0;
    const uint8_t* codes = nullptr;    ///< O * I bytes, each an int8_t
    const float* scales = nullptr;     ///< O floats
};

/// S activation rows of width I, prepared for the Q4 kernels (see the file comment).
struct Act {
    int S = 0, I = 0;
    std::vector<float> xp;     ///< S * I, permuted per group
    std::vector<float> gsum;   ///< S * I / 64
    void prepare(const float* x, int S_, int I_);
};

/// y[s * ldy + (r - ybase)] = sum_i W[r, i] x[s, i] for r in [r0, r1), every s.  Single-threaded over its row
/// range.  `ybase` lets a caller write a row sub-range into its own buffer (MLA's value half of kv_b).
void q4_rows(const Q4& W, const Act& a, int r0, int r1, float* y, int ldy, int ybase = 0);
/// Dot products the attention core needs.
float dot_f32(const float* a, const float* b, int n);
/// y += a * x
void axpy_f32(float* y, float a, const float* x, int n);
/// The whole matrix, rows spread over the pool.  y is S x O (ldy = O).
void q4_gemm(Pool& pool, const Q4& W, const Act& a, float* y);
/// Convenience: prepare and multiply.  x is S x I, y is S x O.
void q4_gemm(Pool& pool, const Q4& W, const float* x, int S, float* y);

/// out[i] += sum_r coef[r] * W[r0 + r, i] for r in [0, n): the transposed product MLA absorption needs
/// (q_nope through the key half of kv_b).  Single-threaded.
void q4_rows_t(const Q4& W, int r0, int n, const float* coef, float* out);

/// out[k * W.I + i] = W[r0 + k, i] for k in [0, n): rows as floats, in natural order.
void q4_dequant_rows(const Q4& W, int r0, int n, float* out);

/// Absorbed MLA attention of ONE head over a prompt's S queries (the prompt path; one token keeps the per-query
/// path in GlmModel::attention).  `Wk` is the head's key half of kv_b as floats (nope x kvl), `Wv` its value half
/// (vh x kvl).  Query s is `q + s * qstride` (nope, then the roped nr), at position pos0 + s, and sees cache rows
/// 0 .. pos0 + s of `kv` (each kvl latent + nr roped key).  Writes vh floats to `ctx + s * cstride`.  Queries go in
/// blocks of 4 so each cache row is loaded once per block; the math is the per-query path's, in another order.
void mla_head_prompt(const float* Wk, const float* Wv, const float* q, int qstride, const float* kv, int kvl, int nr,
                     int nope, int vh, int S, int pos0, float scale, float* ctx, int cstride);

/// y[r] = sum_i W[r, i] x[i] for one activation, rows spread over the pool (lm_head).
void q8r_gemv(Pool& pool, const Q8R& W, const float* x, float* y);
/// x[i] = W[r, i] (the embedding lookup).
void q8r_row(const Q8R& W, int r, float* x);

// ---- the scalar references: the definition of the formats, used by the tests -------------------------------
void q4_rows_ref(const Q4& W, const float* x, int S, int r0, int r1, float* y, int ldy);
float q4_weight(const Q4& W, int r, int i);

// ---- elementwise ---------------------------------------------------------------------------------------------
void rmsnorm(float* out, const float* x, const float* w, int n, float eps);
void softmax_inplace(float* x, int n);
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

/// GLM's interleaved partial RoPE on one `n`-element slice (n = qk_rope = 64), colibri's `rope_interleave`:
/// the pairs are (2j, 2j+1), and the rotated pair is written to (j, j + n/2).  The output order is NEOX, which
/// is fine because q and k are always rotated by the same function before their dot product.
struct Rope {
    int n = 0;
    std::vector<float> inv;   ///< n/2 inverse frequencies, float (as colibri and HF compute them)
    void init(int n_, double theta);
    void apply(float* v, int pos) const;
};

bool cpu_has_avx2();

}  // namespace strata::glm
