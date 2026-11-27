// include/strata/hy4flash_gguf/cpu/native_expert.hpp - plan v0.3 P6: one routed expert in its GGUF form on the CPU.
//
// The IQ2_XS / IQ3_XXS model files keep their experts in i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
// IQ3_S gate/up; Q2_0 or IQ4_NL down) whose values cannot be re-expressed in the Q2_0 pack form.  A native expert
// blob is the three GGUF slices back to back, [gate rows | up rows | down rows]. Activations use the weight type's
// ggml `vec_dot_type`; dispatch selects a format-specific kernel or ggml's per-token dot. The arithmetic mode
// controls whether grouping tokens may change the selected row arithmetic.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::hy4_flash::cpu {

/// Bytes of the largest quantized activation any native layer uses (2560 values as Q8_K: 10 x 292 = 2920;
/// a 3072-wide expert input takes 12 x 292 = 3504).
inline constexpr size_t kNativeActBytes = 4096;
/// Maximum quantized down-activation size for the compiled expert geometry.
inline constexpr size_t kNativeHBytes = 1024;

// Bounds used to allocate scratch for variable native expert geometries.

/// The widest expert intermediate the native path takes.  `FF` is the compiled geometry's and is
/// static_asserted against it, so it cannot grow; this is the separate bound.
inline constexpr int kNativeFfMax = 1536;
/// Bytes of the widest quantized down activation that follows from it: 1536 values are 1632 (Q8_0: 48 x 34) or
/// 1752 (Q8_K: 6 x 292).  `native_fmt` refuses a layer whose down activation exceeds this.
inline constexpr size_t kNativeHBytesMax = 2048;

/// Arithmetic fixed for one pool batch. WindowInvariant preserves the legacy singleton result at every width.
enum class NativeArithmetic { Default, WindowInvariant };

/// One layer's native expert geometry.
struct NativeFmt {
    int gu_type = -1, d_type = -1;      ///< ggml types of gate/up and of down
    int gu_act = -1, d_act = -1;        ///< their vec_dot_type (the activation formats)
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       ///< bytes per weight row
    size_t up_off = 0, down_off = 0;    ///< inside the blob
    size_t bytes = 0;                   ///< the whole blob
    size_t act_bytes = 0, h_bytes = 0;  ///< quantized activation sizes (n_embd of gu_act, n_ff of d_act)
    /// NVFP4 (ggml type 40) only: the blob ends with {s_gate, s_up, s_down, 0} at this offset, the per-expert
    /// weight_scale_2 the block format has no room for (tools/iq_pack.py). 0 = no tail. Each scale goes on its own
    /// projection's FP32 output: gate * s_gate, up * s_up, down * s_down. Do not fold s_down into up: the hidden
    /// drops to ~1e-5 and its q8 block scale becomes an FP16 subnormal (expert error 2-12% instead of 1.1%).
    size_t tail_off = 0;
    /// prism.hadamard (hadamard.hpp): the experts of this layer are stored in a rotated basis, and
    /// native_quant_act / native_quant_h apply y = H_block (signs * x) before quantizing.  had_block 0 = none.
    int had_block = 0;
    bool had_gu = false, had_d = false;
    int layer = -1;                     ///< the layer this format belongs to (set by expert_layout_load; STRATA_ACT_DUMP only)
    /// Expert SwiGLU clamp. Zero preserves the unclamped activation.
    float swiglu_clamp = 0.f;
    const float* gu_signs = nullptr;    ///< n_embd signs for x (nullptr: identity)
    const float* d_signs = nullptr;     ///< n_ff signs for h (nullptr: identity)
    /// Optional non-owning planes for a scattered GGUF expert. Null keeps the original blob/offset path.
    /// Explicit gate/up planes use the scalar-per-row native dispatch, including the requested clamp.
    /// Their lifetime must cover the complete synchronous pool batch; no pointer subtraction is required.
    const uint8_t* gate_data = nullptr;
    const uint8_t* up_data = nullptr;
    const uint8_t* down_data = nullptr;
    /// Opt-in hy4-flash CPU contract: per32 Q8_1 with half scale/sum, Q4_K/Q6_K integer dots.
    /// This changes CPU arithmetic; it does not promise CUDA reduction/transcendental bitwise equality.
    bool gpu_q8_1 = false;
    // Device-derived tables for Q8_1 quantization and SwiGLU.
    const uint32_t* q81_aligned_table = nullptr;
};
constexpr int kNvfp4Type = 40;
constexpr size_t kNvfp4Tail = 16;

/// Whether this build has the ggml-cpu path.
bool native_experts_available() noexcept;
/// Fills `f` for a layer; false (with a reason) when ggml-cpu has no dot product for a type.
bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err);

/// x (n_embd floats) -> the gate/up activation (act_bytes), rotated first when the layer is Hadamard-folded.
void native_quant_act(const NativeFmt& f, const float* x, void* dst);
/// h (n_ff floats) -> the down activation (h_bytes), rotated first when the layer is Hadamard-folded.
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

/// Same per-token result as the corresponding legacy nt=1 call on this build and CPU.
/// Kernels that already satisfy this contract retain their grouped execution.
void native_gu_rows_invariant(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt,
                              float* const* ff, int r0, int r1);
void native_down_rows_invariant(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt,
                                float* const* out, int r0, int r1);

}  // namespace strata::hy4_flash::cpu
