// include/strata/prefill/rp_moe.hpp - gfx906: the prompt path's IQ4_NL expert products through mx-llama.cpp's
// repacked-weight GEMM (src/prefill/rp/, MIT).  The gathered group buffer holds each expert in the repacked layout
// (a de-aliased nibble plane, then an f16 scale plane: repack_gcn) instead of GGUF blocks, written by `repack_iq4nl`
// in place of the plain copy, so the resident experts (and the decode kernels that read them) are untouched.
// Opt-in, STRATA_PF_RP=1.
#pragma once

#include <cstddef>
#include <cstdint>

#include "strata/prefill/moe_mmq.hpp"

namespace strata::prefill::rp {

/// This build has the repacked GEMM (gfx906 only).
bool built();
/// STRATA_PF_RP=1 and built(): the prompt path's IQ4_NL products take the repacked GEMM.
bool enabled();
/// STRATA_DENSE_RP=1 and built(): under STRATA_DENSE_MMQ=1, a dense projection of a dense_supported type takes the repacked GEMM (its
/// weight repacked into a scratch per call, Gemm::native_mmq) instead of llama.cpp's MMQ.  IQ4_NL / Q8_0: the same
/// products, bitwise; the K-quants sum in another order (the same error against an FP64 reference).
bool dense_enabled();
/// STRATA_PF_RP=1 or 2: the group buffers are sized for repacked slots (2: sized so, products still MMQ - the A/B
/// that separates the GEMM from the memory plan's change).
bool sized();
/// The expert path's repacked GEMM covers this ggml type and shape: IQ4_NL (cols a multiple of 128), and the codebook
/// gate/up types IQ3_XXS, IQ3_S, IQ2_S (cols a multiple of 256; their gather decodes them to int8 rows and FP32 scales,
/// rp/repack-iq3.cuh; IQ2_S only with STRATA_PF_RP_IQ3=2, none with 0); rows a multiple of 64.
bool supported(int ggml_type, int64_t w_rows, int64_t w_cols);
/// The repacked GEMM covers this type as a dense projection (Gemm::native_mmq, STRATA_DENSE_RP=1): IQ4_NL and Q8_0
/// (cols a multiple of 128), Q6_K, Q5_K and IQ4_XS (cols a multiple of 256; IQ4_XS is Strata's own layout,
/// rp/repack-iq4xs.cuh; Q4_K stays on MMQ, faster there); rows a multiple of 64.  The expert path
/// takes `supported`.
bool dense_supported(int ggml_type, int64_t w_rows, int64_t w_cols);
/// `n` GGUF matrices of [rows, cols] of a dense_supported type (`src_stride` bytes apart) into the repacked layout
/// `dst_stride` apart: mx-llama.cpp's device repack kernels (src/prefill/rp/repack-convert.cuh).
void repack(int ggml_type, const void* src, size_t src_stride, void* dst, size_t dst_stride, int64_t rows, int64_t cols,
            int n, void* stream);
/// Bytes of one [rows, cols] matrix in the repacked layout (mx's repack_gcn_nbytes; a little over the GGUF size:
/// rows whose nibble bytes are a multiple of 128 get 16 bytes of de-alias padding).
size_t matrix_bytes(int ggml_type, int64_t rows, int64_t cols);
/// `n` GGUF IQ4_NL matrices of [rows, cols] (`src_stride` bytes apart; one matrix may also be given as two pieces of
/// rows, see repack_iq4nl_rows) into the repacked layout `dst_stride` bytes apart.
void repack_iq4nl(const void* src, size_t src_stride, void* dst, size_t dst_stride, int64_t rows, int64_t cols, int n,
                  void* stream);
/// Rows [row0, row0 + nrows) of one repacked [rows, cols] matrix at `dst`, from `nrows` GGUF rows at `src` (the gather
/// writes gate and up, two pieces of one gate/up matrix, from their own blob offsets).
void repack_iq4nl_rows(const void* src, void* dst, int64_t rows, int64_t cols, int64_t row0, int64_t nrows,
                       void* stream);

/// The gather for a group whose gate/up and down are both IQ4_NL: experts [g.first, g.n) from their blobs (gate at
/// +0, up at +up_off, down at +down_off, GGUF rows) into repacked [2*n_ff, n_embd] gate/up matrices `gu_stride` bytes
/// apart at gu_dst (gate rows, then up rows) and repacked [n_embd, n_ff] down matrices `d_stride` apart at d_dst.  One
/// launch; slot q of the group buffers is expert q.  gu_dst or d_dst null: that part is skipped (a layer where only
/// one of the two is IQ4_NL gathers the other as GGUF blocks).
void gather_group_iq4nl(const mmq::GatherGroup& g, size_t up_off, size_t down_off, int64_t n_embd, int64_t n_ff,
                        void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, void* stream);

/// The gate/up part of a group gather alone, for a `supported` gate/up type: experts [g.first, g.n) from their blobs
/// (gate at +0, up at +up_off) into repacked [2*n_ff, n_embd] matrices `gu_stride` bytes apart at gu_dst.
void gather_group_gu(const mmq::GatherGroup& g, int gu_type, size_t up_off, int64_t n_embd, int64_t n_ff, void* gu_dst,
                     size_t gu_stride, void* stream);

/// The group gather for the repacked part(s): gather_group_gu for a codebook gate/up (`gu_type`), then
/// gather_group_iq4nl for the rest (an IQ4_NL gate/up, the down); a null destination is skipped, as there.
void gather_group(const mmq::GatherGroup& g, int gu_type, size_t up_off, size_t down_off, int64_t n_embd, int64_t n_ff,
                  void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, void* stream);

/// The repacked GEMM over a group, with mmq::Product's meaning (w / expert_bytes: the repacked matrices; xq, bounds,
/// ids, total_rows, dst, ld_dst as for mmq::Context::run).  Keeps its small routing scratch (the identity source
/// map, the tile prefix sums and tile table) between calls.
class Context {
public:
    Context() = default;
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    void run(const mmq::Product& p, void* stream);

private:
    // two buffers, each only grown: the identity source map (valid for its first iota_rows_ entries) and the tile
    // tables a call rewrites (one shared buffer let a call's tables land on the map's later entries)
    int32_t* iota_ = nullptr;
    int64_t iota_rows_ = 0;
    void* tables_ = nullptr;
    size_t tables_bytes_ = 0;
};

}  // namespace strata::prefill::rp
