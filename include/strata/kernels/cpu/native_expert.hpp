// include/strata/kernels/cpu/native_expert.hpp - plan v0.3 P6: one routed expert in its GGUF form on the CPU.
//
// Native expert blobs keep the three GGUF slices back to back, [gate rows | up rows | down rows]. Existing i-quants
// use ggml-cpu's own activation quantizers and dot products. Strata's experimental Hadamard-INT2 extension uses its
// scalar reference decoder and the per-pack rotation seed.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::kernels::cpu {

/// Bytes reserved for the largest native activation, including 2560 Hadamard-INT2 values stored as FP32.
inline constexpr size_t kNativeActBytes = 16384;
/// Bytes reserved for the largest native down activation, including a transformed FP32 buffer when needed.
inline constexpr size_t kNativeHBytes = 16384;

/// One layer's native expert geometry.
struct NativeFmt {
    int gu_type = -1, d_type = -1;      ///< ggml types of gate/up and of down
    int gu_act = -1, d_act = -1;        ///< their vec_dot_type (the activation formats)
    int64_t n_embd = 0, n_ff = 0;
    uint64_t had2_seed = 0;
    size_t gu_row = 0, d_row = 0;       ///< bytes per weight row
    size_t up_off = 0, down_off = 0;    ///< inside the blob
    size_t bytes = 0;                   ///< the whole blob
    size_t act_bytes = 0, h_bytes = 0;  ///< required activation buffer sizes for gate/up and down
};

/// Whether this build has the ggml-cpu path.
bool native_experts_available() noexcept;
/// Fills `f` for a layer; false (with a reason) when ggml-cpu has no dot product for a type.
bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err,
                uint64_t had2_seed = 0);

/// x (n_embd floats) -> the gate/up activation (act_bytes).
void native_quant_act(const NativeFmt& f, const float* x, void* dst);
/// h (n_ff floats) -> the down activation (h_bytes).
void native_quant_h(const NativeFmt& f, const float* h, void* dst);

/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]) for rows r in [r0, r1), `nt` tokens.
void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1);
/// out[t][r] = down_r . hq[t] for rows r in [r0, r1).
void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1);

}  // namespace strata::kernels::cpu
