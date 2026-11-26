// include/strata/kernels/tq_i8_rule.hpp - STRATA_TQ_I8: the trellis (APR) expert formats on our own int8
// dot-product tile kernels (native_expert_i8, src/kernels/cuda/i8_expert.cuh).
//
// Strata's own kernels cover the i-quant and K-quant formats (i8_expert.cuh); the trellis formats - TQ2_T, TQK6,
// TQK7, the experts of Gyro-M and Gyro-S - were never in them, so on a card without llama.cpp's MMQ (every HIP build:
// 0.2.16 removed it, tools/check_no_mmq.py forbids the symbols) a folded trellis layer falls to the FP16 dequantize +
// FP16 GEMM path.  On RDNA2 (gfx1030 / 1031 / 1032) that costs the prefill about a third of its rate: the FP16 GEMM of
// a gate/up product of a Gyro-S layer measured 4,353 ms on an RX 6800 at engine 0.1.59 against 14,698 ms at 0.1.62,
// decode unchanged.
//
// The switch puts the trellis formats on the int8 kernels, which have DP4A on those cards (i8_dot0's v_dot4_i32_i8):
// the weight side decodes the int8 codebook (tq_lut_i8) into four int8 per trellis step and dots it against the q8_1
// activations, the same arithmetic the grouped trellis kernels (iq_kernels.cu's tq_load / tq_apply) do.  A Hadamard
// fold is covered too: the gate/up launch writes its h rows as FP32, the layer's hadamard_rows runs in place, and the
// rows are requantized into the kernels' own image before the down launch (the fused path's pattern, moe_fused_iq.cu).
//
// Opt-in: not set is OFF, the FP16 path stays exactly as it is today (byte-identical output and dispatch).  1: on.
// 0 or anything else: off, nothing said.  Nothing picks it on its own, on any card; a card without DP4A takes it as
// any other card does: the kernels decode and dot the same way, and i8_dot0 falls back to the platform's dp4a.
//
// Pure: no device query, no allocation, no logging.  The kernel file and the prefill plan ask it.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>

namespace strata::kernels {

/// The switch's state from its text: -1 not set (off, said nothing), 0 off, 1 on.  Anything else reads as off.
inline int tq_i8_switch_state(const char* env) noexcept {
    if (env == nullptr || env[0] == '\0') return -1;
    const char* s = env;
    while (*s >= '0' && *s <= '9') ++s;
    if (s == env || *s != '\0') return 0;
    return std::strcmp(env, "1") == 0 ? 1 : 0;   // (the i8_switch_parse of i8_expert_rule.hpp accepts more; this switch is 0 / 1 only)
}

/// A gate/up and down pair the trellis int8 kernels can take: both trellis (TQ2_T 144, TQK6 145, TQK7 146).  A mix
/// is fine (Gyro-S: TQK6 gate/up, TQK7 down); a trellis pair with an i-quant or K-quant partner is not this path.
inline bool tq_i8_pair(int gu, int dt) noexcept {
    return (gu == 144 || gu == 145 || gu == 146) && (dt == 144 || dt == 145 || dt == 146);
}

/// The fold this path covers: the pack's Hadamard fold of block 128 on both roles (gate == up: one input, and down),
/// which the kernels' FP32-h epilogue, hadamard_rows and requantize handle.  Any other block, a role missing, or a
/// fold without both roles: not this path (the FP16 path keeps it, as today).
inline bool tq_i8_fold_ok(int had_block, bool up_had, bool down_had) noexcept {
    return had_block == 128 && up_had && down_had;
}

/// Does the trellis int8 path take this layer?  `enabled`: the switch is on (tq_i8_switch_state == 1); `fold`: the
/// layer's fold state, allowed when there is none (had_block 0) or the one above.  The shapes the kernels need
/// (n_embd, n_ff, one blob's bytes) are checked by native_expert_i8_supported next to it.
inline bool tq_i8_takes(int gu, int dt, int had_block, bool up_had, bool down_had, bool enabled) noexcept {
    if (!enabled || !tq_i8_pair(gu, dt)) return false;
    return had_block == 0 || tq_i8_fold_ok(had_block, up_had, down_had);
}

/// The line for a STRATA_TQ_I8 that was asked for but not taken (not on (a pair that is not trellis, a fold this
/// path does not cover, a shape the kernels do not): said once by the prompt path, so a tuned switch is not left
/// silently doing nothing.  Empty for a switch that is off or not set.
inline std::string tq_i8_notice(const char* env, bool taken) {
    const int st = tq_i8_switch_state(env);
    if (st != 1 || taken) return std::string();
    return "strata: STRATA_TQ_I8=1 asks for the trellis experts on our int8 kernels; this layer is not a trellis "
           "pair this build and shape takes, so it keeps the FP16 path";
}

}  // namespace strata::kernels
