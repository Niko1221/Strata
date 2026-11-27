// include/strata/kernels/loop_gate.hpp - the recurrent-depth loop's per-layer message + gate (issue #7295).
//
// A PORT of Inkling2Loop's own mechanism (engine/src/inkling2/inkling2_loop.cpp: `attn_res_mix_f32`,
// `loop_control_f32`), not a from-scratch design - see LOOP_TRANSFORMER_PLAN.md §2a for the real schedule
// this is read out of, and §2b for the two honest adaptations made for the generic QSA-only path:
//
//   1. TWO gated channels, not four. Inkling2Loop separately norms an attn-query input, a K input and a V
//      input (three channels) plus an MLP input (a fourth) before its own bespoke projections. QSA's
//      mixer takes ONE normalized activation (`gr_read`'s `bb.mixed`) and derives Q/K/V from it
//      internally - there is nothing to gate separately for K and V. So this ports 2 channels: the QSA
//      mixer's input and the FFN/MoE input - the two points `block_layer_pre` already splits at.
//   2. The bank holds `bb.mixed` (plain n_embd vectors), not Strata's own hyper-connection residual `R`
//      (hc, n_embd) - because Inkling2Loop has no hyper-connection streams at all, and `bb.mixed` is
//      exactly its kind of thing (the plain activation a block's mixer/FFN actually reads).
//
// THE MESSAGE STEP (`loop_msg_attn`, ports `attn_res_mix_f32`): softmax attention over the PREVIOUS pass's
// bank of per-layer snapshots plus its end-of-core-range state, with ONE LEARNED QUERY PER LAYER
// (`loop_msg_query.weight`, [n_embd, 2] - one query row per channel), each state RMS-normalized before the
// score and the weighted sum (the normalization is folded into the score/sum math below rather than
// materialized, which is equivalent and avoids a scratch buffer the size of the whole bank).
//
// THE GATE STEP (`loop_ctrl_gate`, ports `loop_control_f32`): a GRU cell. `cat = rmsnorm(cur) ++
// rmsnorm(msg)` (2*n_embd), `gate = down @ cat` (low-rank, width `loop_gate_rank`), `g = tanh(gain @
// gate)`, `f = tanh(forget @ gate)`, `out = (1 - f) * cur + g * msg`. Called once per channel (`cur` =
// that channel's `bb.mixed` before the mixer/FFN reads it, `msg` = that channel's message), with its own
// `down`/`gain`/`forget` weight triple (`loop_{attn,mlp}_ctrl_{down,gain,forget}.weight`).
//
// Both steps are deliberately plain (serial per-thread reductions, no warp tuning) for the same reason
// `include/strata/kernels/loop_gate.hpp`'s predecessor in this repo was: there is no trained model yet to
// profile this against, and `gr.cu`'s own history is what tuning ahead of a measurement costs.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

struct LoopMsgShapes {
    int64_t n_embd = 0;
    int64_t bank_n = 0;   ///< entries in `bank` (the previous pass's per-layer snapshots), NOT counting `bank_out`
};

/// `msgs` is `(2, n_embd)`: one message per channel (0 = attn, 1 = mlp). `bank` is `(bank_n, n_embd)`,
/// `bank_out` is `(n_embd,)`, `query` is `(2, n_embd)` (`loop_msg_query.weight`, this layer's own). A
/// no-op when `s.bank_n <= 0` and there is no `bank_out` either - callers do not reach this on pass 0.
void loop_msg_attn(const float* bank, const float* bank_out, const float* query, const LoopMsgShapes& s,
                   float* msgs, void* stream);

struct LoopCtrlShapes {
    int64_t n_embd = 0;
    int64_t gate_rank = 0;
};

/// Bytes of device scratch `loop_ctrl_gate` needs (the normalized `cat` plus the low-rank `gate`
/// intermediate). 0 when the shape is degenerate.
size_t loop_ctrl_workspace_bytes(const LoopCtrlShapes& s);

/// `out = (1 - f) * cur + g * msg` - see the file comment for `f`/`g`. `cur`, `msg`, `out` are each
/// `(n_embd,)`; `out` may alias `cur` (every output element is computed from workspace and `msg` before
/// it is written, `cur` itself is read only while computing the workspace, earlier in the same call). A
/// no-op when `s.gate_rank <= 0` (mirrors the loop-level "no gate" fallback, same convention as before).
void loop_ctrl_gate(const float* cur, const float* msg, const uint16_t* down, const uint16_t* gain,
                    const uint16_t* forget, const LoopCtrlShapes& s, float* workspace, float* out,
                    void* stream);

}  // namespace strata::kernels
