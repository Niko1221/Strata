// include/strata/kernels/iq_kernels.hpp - the GGUF expert and embedding formats on the GPU: the i-quants (IQ1_M,
// IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL) and Q2_0 of the IQ2_XS / IQ3_XXS model files, and Q4_K, Q5_K,
// Q5_1 and Q8_0 of the UD-Q4_K_XL file.
//
// The block layouts, codebook grids and dot products are llama.cpp's (ggml-common.h, ggml-cuda/vecdotq.cuh,
// ggml-cuda/dequantize.cuh; MIT, see third_party/ggml/LICENSE and VERSION.txt), so a weight means exactly what it
// means in llama.cpp.  Activations are q8_1 (32 values, fp16 scale and fp16 sum), the llama.cpp CUDA contract.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// ggml type ids handled here.
bool iq_supported(int ggml_type) noexcept;
/// Bytes of one row of `n` values of `ggml_type` (n a multiple of the type's block).
size_t iq_row_bytes(int ggml_type, int64_t n) noexcept;

/// q8_1 blocks for `n_rows` rows of `n_cols` floats (n_cols a multiple of 32): y is n_rows * n_cols/32 blocks.
void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream);

/// y[c][r] = W[r] . x[c] for `ncols` columns of q8_1 activations (x stride n_in/32 blocks per column).
void iq_mmvq(int ggml_type, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);
/// iq_mmvq for 2..8 columns from their interleaved copy (native_quantize_q8_1_il), bitwise: IQ3_S.
bool iq_mmvq_il_supported(int ggml_type) noexcept;
void iq_mmvq_il(int ggml_type, const void* w, const void* x_il, float* y, int n_in, int n_out, int ncols, void* stream);

/// Dequantize `n` contiguous values (n a multiple of 256) to fp16 / fp32.
void iq_dequant_f16(int ggml_type, const void* src, int64_t n, uint16_t* dst, void* stream);
void iq_dequant_f32(int ggml_type, const void* src, int64_t n, float* dst, void* stream);
/// Rows `tokens[0..n_tok)` (device ids) of a GGUF embedding table (`row_bytes` per row; the table may be mapped
/// host memory) dequantized to fp32, `n_embd` per row (a multiple of 256).
void iq_embed_rows(int ggml_type, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok,
                   int64_t n_embd, float* out, void* stream);
/// One expert's gate and up matrices (n_ff rows of n_embd each) into the interleaved fp16 layout the prompt path
/// uses: row 2r = gate row r, row 2r+1 = up row r.
void iq_dequant_gu_f16(int ggml_type, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst,
                       void* stream);
/// iq_dequant_gu_f16 and iq_dequant_f16 (the down matrix) of n experts in one launch: expert i's blob at blobs[i]
/// (a DEVICE array), its gate/up into gu + i * 2 * n_ff * n_embd and its down into dn + i * n_embd * n_ff.
void iq_dequant_experts_f16(int gu_type, int d_type, const uint8_t* const* blobs, int n, size_t up_off, size_t down_off,
                            int64_t n_ff, int64_t n_embd, uint16_t* gu, uint16_t* dn, void* stream);

/// The layout of one native expert blob: [gate rows | up rows | down rows], raw GGUF blocks.
struct NativeExpertLayout {
    int gu_type = -1, d_type = -1;
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       // bytes per row
    size_t up_off = 0, down_off = 0;    // byte offsets inside the blob
    size_t bytes = 0;                   // the whole blob
};
NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff);
/// Whether the GPU has the dot products (native_expert_grouped) and the dequantizers (the prompt path) for a layer's
/// gate/up and down formats at this geometry.
bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept;

/// Bytes of scratch `native_expert_grouped` needs for `cap_entries` entries.
size_t native_expert_scratch_bytes(int64_t cap_entries, int64_t n_ff);

/// Grouped experts in the native format: group g's blob at device address grp_ptr[g]; its entries
/// [grp_start[g], grp_start[g+1]) read token ent_tok[e]'s q8_1 activation (n_embd/32 blocks per token in x_q8_1)
/// and write row ent_dst[e] of `out` (n_embd floats).  Counts are read on the device.
void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream);
/// Copies the first grp_start[*n_groups] rows of `rows` (n_embd floats each, n_embd % 4 == 0) to `out`, at most
/// `max_rows`, in 16-byte stores: rows bound for mapped host memory cross PCIe in full transactions (the grouped
/// kernels' one-float stores there cost ~0.2 ms a layer).  Counts are read on the device.
void native_expert_rows_out(const float* rows, const int32_t* n_groups, const int32_t* grp_start, int64_t n_embd,
                            int64_t max_rows, float* out, void* stream);

}  // namespace strata::kernels
