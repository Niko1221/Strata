// Select the projection backend by activation width and token count.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::hy4_flash {

/// Columns at or below which a projection stays on the quantised matvec. `MAX_NCOLS` in native_mmvq.cu;
/// `prefill_chunk_max()` probes the real value, and this is the value the backend switch compares against.
/// They are the same number for a reason - see the header comment.
constexpr int kGemmT = 8;

/// STRATA_HY4_FLASH_GEMM_T overrides the token-count threshold for GEMM.
int gemm_t();

/// The GEMM backend for one pass: a `prefill::Gemm` and the FP16 activation images a chunk's projections
/// read. Opaque because a PUBLIC hy4_flash header must not include `strata/prefill/*` - that library is a
/// PRIVATE dependency of `strata_hy4_flash`, and a public include would make it everyone's.
struct MatCtx;

/// Keep one GEMM context per thread across chunks.
MatCtx* mat_ctx(int chunk_max, std::int64_t widest_row, std::int64_t weight_elems, void* stream,
                std::string& err);

/// Free everything the context holds. Called when the engine tears down; a pass does not need it.
void mat_ctx_release();

// Automatic memory plans must establish their backend before mutating KV, not silently
// switch numerical paths when a deferred router allocation fails.
bool mat_ctx_require_prefill(MatCtx* mc, void* stream, std::string& err);

/// Owned allocation bytes and a conservative capacity plan, for prefill admission before computation.
/// The plan includes the deferred FP16 fallback; opaque library allocations need a separate margin.
std::size_t mat_ctx_owned_bytes();
std::size_t mat_ctx_plan_bytes(int tokens, std::int64_t widest_row, std::int64_t weight_elems);

/// Staged views of one activation for quantized matvec and FP16 matrix multiplication.
struct MatIn {
    MatCtx* mc = nullptr;
    const void* q8 = nullptr;            ///< Q8_1 blocks, what the matvec reads
    const std::uint16_t* f16 = nullptr;  ///< FP16 rows, what the GEMM reads; null = matvec whatever T is
    /// The activation AS IT LIES, F32. The MMQ path quantises from this rather than from `f16`: going
    /// F32 -> FP16 -> F32 -> q8_1 would round twice for nothing, and the FP16 image may not even exist
    /// when only MMQ is wanted. Null off the GEMM path.
    const float* f32 = nullptr;
    /// Staging generations distinguish different activations that reuse the same address.
    std::uint64_t stamp = 0;
    std::int64_t n = 0;                  ///< the row length both images describe (a check, not an index)
};

/// Reset the FP16 arena. Once at the top of a chunk, before any staging.
void mat_pass_begin(MatCtx* mc);

/// Select projection sites individually with STRATA_HY4_FLASH_GEMM_SITES.
enum MatSite : unsigned {
    kSiteQA      = 1u << 0,   ///< mla_project: cur -> q residual
    kSiteQB      = 1u << 1,   ///< mla_project: the normed q residual -> q
    kSiteKVA     = 1u << 2,   ///< mla_project: cur -> the latent and rope key
    kSiteAttnGate= 1u << 3,   ///< mla_output: cur -> the attention gate
    kSiteWO      = 1u << 4,   ///< mla_output: the gated head-wide value -> the block output
    kSiteMixes   = 1u << 5,   ///< hc_pre: the normed flat state -> the hyper-connection coefficients
    kSiteEngramKV= 1u << 6,   ///< engram_layer: the normed lookup -> its key and value
    kSiteFFNGU   = 1u << 7,   ///< dense_ffn: the hidden -> gate and up
    kSiteFFNDown = 1u << 8,   ///< dense_ffn: the SwiGLU -> the block output
    kSiteRoutedDn= 1u << 9,   ///< latent_moe: the hidden -> the experts' subspace
    kSiteRoutedUp= 1u << 10,  ///< latent_moe: the combined latent -> full width
    kSiteShexpGU = 1u << 11,  ///< latent_moe: the hidden -> the shared expert's gate and up
    kSiteShexpDn = 1u << 12,  ///< latent_moe: the shared expert's SwiGLU -> full width
    kSiteAll     = ~0u,
};

/// Whether a projection of `T` columns at `site` through `mc` takes the GEMM. For a caller that has to
/// produce something different for each - the SwiGLU's output, which one path wants in Q8_1 and the other
/// in FP16.
bool mat_gemm_path(const MatCtx* mc, int T, unsigned site = kSiteAll);

/// `x` quantised to Q8_1 in `q8`, plus its FP16 image on the GEMM path: the pair a projection reads.
/// Replaces a bare `native_quantize_q8_1` at a site whose output feeds a matrix.
MatIn mat_stage(MatCtx* mc, const float* x, void* q8, std::int64_t n, int T, void* stream,
                unsigned site = kSiteAll);

/// The same for a site whose Q8_1 image came from a FUSED kernel (`native_rms_norm_quantize_q8_1`,
/// `native_swiglu_quantize_q8_1`): the blocks exist already, so only the FP16 image is made here.
MatIn mat_staged(MatCtx* mc, const float* x, const void* q8, std::int64_t n, int T, void* stream,
                 unsigned site = kSiteAll);

/// A pair that is Q8_1 only - what a caller outside any prompt chunk hands over, and what every decode step
/// hands over. `mat_apply` on it is `native_mmvq`, with no branch that can do anything else.
inline MatIn mat_q8_only(const void* q8, std::int64_t n) {
    MatIn in;
    in.q8 = q8;
    in.n = n;
    return in;
}

/// Apply the selected matrix backend to the staged activation.
void mat_apply(const MatIn& x, int ggml_type, const void* w, float* y, std::int64_t n_in,
               std::int64_t n_out, int T, void* stream, unsigned site = kSiteAll);
void mat_f32(const float* w, const float* x, float* y, int n_in, int n_out, int T, void* stream);
bool mat_decode_equivalent();
void mat_set_decode_equivalent(bool enabled);

/// Project the router inputs using the FP32 weight matrix.
bool mat_router_f32(MatCtx* mc, const float* w, const float* x, float* y, std::int64_t n_in,
                    std::int64_t n_out, int T, void* stream);

}  // namespace strata::hy4_flash
