// include/strata/kernels/cpu/native_expert.hpp - plan v0.3 P6: one routed expert in its GGUF form on the CPU.
//
// The IQ2_XS / IQ3_XXS model files keep their experts in i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
// IQ3_S gate/up; Q2_0 or IQ4_NL down) whose values cannot be re-expressed in the Q2_0 pack form.  A native expert
// blob is the three GGUF slices back to back, [gate rows | up rows | down rows], and the arithmetic is ggml-cpu's
// own (`ggml_get_type_traits_cpu`): the activation is quantized with the weight type's `vec_dot_type` and each row
// is one `vec_dot`, exactly what llama.cpp's CPU backend computes for the same tensor.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::kernels::cpu {

/// Bytes of the largest quantized activation any native layer uses.  It is sized for the WIDEST family, not the
/// first one: qwen4exp's 2560 values as Q8_K are 10 x 292 = 2920 B, and glm5-next's 4096 are 16 x 292 = 4672 B,
/// which the old 4096 did not hold.  This is a scratch bound (`pool.hpp` embeds it per worker), so it is a cost,
/// not a contract - `native_fmt` sets the real `act_bytes` per layer and every kernel takes the format.
inline constexpr size_t kNativeActBytes = 8192;
/// Bytes of the largest quantized down activation: 640 values as Q8_0 are 20 x 34 = 680 B, glm5-next's 2048 are
/// 64 x 34 = 2176 B (or 8 x 292 as Q8_K, 2336 B) - the old 1024 held neither.
inline constexpr size_t kNativeHBytes = 4096;
/// Rows of the largest expert intermediate the pool can hold `MAXT` of.  **THIS ONE IS NOT A SCRATCH BOUND.**
/// `ExpertPool::SplitBufMulti` keeps `float ff[MAXT][kMaxExpertFF]` per split entry, and the multi-token native
/// phase writes `n_ff` floats into it; the row count that phase splits each expert by has to be the same number.
/// Both were qwen4exp's 640 - which glm5-next's 2048 rows would have overrun by 3x into the next entry's
/// activations, and split at the wrong multiple on the way.  `native_fmt` refuses a wider expert rather than
/// letting one in.
inline constexpr int kMaxExpertFF = 4096;

/// One layer's native expert geometry.
struct NativeFmt {
    int gu_type = -1, d_type = -1;      ///< ggml types of gate/up and of down
    int gu_act = -1, d_act = -1;        ///< their vec_dot_type (the activation formats)
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       ///< bytes per weight row
    size_t up_off = 0, down_off = 0;    ///< inside the blob
    size_t bytes = 0;                   ///< the whole blob
    size_t act_bytes = 0, h_bytes = 0;  ///< quantized activation sizes (n_embd of gu_act, n_ff of d_act)
};

/// Whether this build has the ggml-cpu path.
bool native_experts_available() noexcept;
/// Fills `f` for a layer; false (with a reason) when ggml-cpu has no dot product for a type.
bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err);

/// x (n_embd floats) -> the gate/up activation (act_bytes).
void native_quant_act(const NativeFmt& f, const float* x, void* dst);
/// h (n_ff floats) -> the down activation (h_bytes).
void native_quant_h(const NativeFmt& f, const float* h, void* dst);

/// From how many tokens native_gu_rows gives this gate/up type to a multi-token kernel (#152; ggml-cpu's per-token dot
/// below that).  1: a token's rows are the same alone and in any group.
int native_gu_mt_min(int gu_type);
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]) for rows r in [r0, r1), `nt` tokens.
void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1);
/// out[t][r] = down_r . hq[t] for rows r in [r0, r1).
void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1);

/// The same two, reading ONE expert's three roles as the three separate pointers they already are.
///
/// A native pack whose experts live in the GGUF keeps gate, up and down as three tensors - and within each, one
/// expert's rows are one contiguous run (`FileExpertSource::slices`).  The blob forms above exist because the
/// kernels were written for an assembled `[gate | up | down]` buffer, and the blob's two halves are exactly
/// `blob` and `blob + up_off`.  Nothing about the arithmetic wants them adjacent, so a caller that has the
/// slices can skip the assembly; on the glm5-next pack that assembly was a 11.67 MB memcpy per expert, 8 experts
/// a layer over 42 layers, per token - 3.9 GB copied single-threaded to feed 279 MB of dot products.
///
/// These use ggml-cpu's per-token `vec_dot` for every group size.  The blob forms hand a group of two or more
/// tokens to the AVX-512/AVX-2 multi-token kernels, which take the assembled offsets and so cannot be called
/// with slices; at one token they are the same code and the same bits (see `native_gu_mt_min`).
///
/// `native_rows_sliceable` is therefore the question a caller must ask FIRST: for this gate/up type, this down
/// type and a group of `nt` tokens, would the blob form pick a multi-token kernel?  True means the slices
/// compute the same bits and are free to use; false means the caller has to assemble a blob.
bool native_rows_sliceable(int gu_type, int d_type, int nt);
void native_gu_rows_ptrs(const NativeFmt& f, const uint8_t* gate, const uint8_t* up, const void* const* act,
                         int nt, float* const* ff, int r0, int r1);
void native_down_rows_ptr(const NativeFmt& f, const uint8_t* down, const void* const* hq, int nt, float* const* out,
                          int r0, int r1);

/// The two row functions above, but allowed the MULTI-TOKEN kernels at a group size: the blob was never the
/// point, only a fixed distance between gate and up, and two slices out of one mapping have that.  They are what
/// makes a chunk pay - the per-token dot re-decodes every row for every token, which is the pool's real limit.
/// `slice_mt_on()` decides (on unless `STRATA_NO_SLICE_MT`), and off they are `native_gu_rows_ptrs` /
/// `native_down_rows_ptr` exactly.  At `nt == 1` both are those functions whatever the switch says.
void native_gu_rows_slice(const NativeFmt& f, const uint8_t* gate, const uint8_t* up, const void* const* act,
                          int nt, float* const* ff, int r0, int r1);
void native_down_rows_slice(const NativeFmt& f, const uint8_t* down, const void* const* hq, int nt, float* const* out,
                            int r0, int r1);
/// Whether the gate/up half is on (on unless `STRATA_NO_SLICE_MT`).
bool slice_mt_on();
/// Whether the down rows' half is on (opt-in, `STRATA_SLICE_MT_DOWN`).
bool slice_mt_down_on();

}  // namespace strata::kernels::cpu
