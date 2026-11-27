// include/strata/kernels/loop_gate.hpp - the recurrent-depth loop's pass-boundary gate (issue #7295, M5).
//
// NEW CODE, NOT A PORT.  Earlier drafts of the loop-transformer plan claimed this already existed, ported from
// a model family called "Inkling2", under `src/inkling2/inkling2_loop.cpp`.  That file, that family and those
// kernel names do not exist anywhere in this repository - a search for `loop_control_f32`, `attn_res_mix_f32`,
// `gate_rank` and an `inkling2` directory across the whole tree (and the sibling `Strata-src` checkout) found
// nothing. So there is nothing to "move out" of, and what follows is a new design, built from the config field
// `ModelGeometry::loop_gate_rank` that M1 already reserved for it.
//
// THE DESIGN: a highway/GRU-style update gate, applied ONCE PER PASS BOUNDARY (not per layer), mixing the
// residual `R` the core range just produced with `R0` - the residual the core range READ on pass 0, i.e. its
// input before any pass ran:
//
//     g  = R @ down^T                      (hc, gate_rank)   - one low-rank projection per stream
//     z  = sigmoid(g @ up^T)                (hc, n_embd)      - broadcast back up to the residual's width
//     R' = z * R + (1 - z) * R0
//
// `gate_rank` keeps the two weight matrices small (hc * n_embd * gate_rank each, not hc * n_embd^2), matching
// the existing `hc_attn_down`/`hc_attn_up` low-rank convention in `gr.hpp` - but this is its own pair of
// matrices with its own name, not a reuse of `gr_read`/`gr_write`'s weights: `gr_write` ADDS a layer's output
// into every stream with a per-stream scalar weight (`include/strata/kernels/gr.hpp`'s `R += bo * 2*sigmoid(..)`),
// which has no notion of "the value before the whole core range ran" - there is nothing in it to blend `R`
// against. `R0` is new state the loop keeps for exactly this.
//
// `loop_gate_rank == 0` means "no gate": `session_loop` skips this file's kernels entirely and the passes
// accumulate through the ordinary per-layer `gr_write` alone, with zero extra code - the fallback this header's
// companion plan calls "simplest v1".  Deliberately UNOPTIMIZED (no warp-per-row, no fused kernels, a serial
// per-thread reduction for a small `gate_rank`): there is no trained model yet to profile this against, and
// `gr.cu`'s own history is a long chain of "tuned ahead of a measurement" mistakes this file does not repeat.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// `n_embd`/`hc` are the residual's shape (same as `GrShapes`); `gate_rank` is `ModelGeometry::loop_gate_rank`.
struct LoopGateShapes {
    int64_t n_embd = 0;
    int64_t hc = 0;
    int64_t gate_rank = 0;
};

/// Bytes of device scratch `loop_gate_mix` needs (the `g` intermediate, `hc * gate_rank` floats). 0 when the
/// shape is degenerate (any dimension <= 0) - the caller then owns skipping the call, same convention as
/// `gr_workspace_init`.
size_t loop_gate_workspace_bytes(const LoopGateShapes& s);

/// `R_out = z*R + (1-z)*R0`, `z = sigmoid(low_rank(R))` - see the file comment for the exact shapes. `R`, `R0`
/// and `R_out` are each `(hc, n_embd)` device buffers; `R_out` may alias `R` (the up kernel reads `R`/`R0` and
/// writes `R_out` per element, no element is read after it is written at that same index). `down` is
/// `(gate_rank, n_embd)` row-major bf16-in-u16 ("loop_gate_down.weight", n_embd contiguous); `up` is
/// `(n_embd, gate_rank)` row-major bf16-in-u16 ("loop_gate_up.weight", gate_rank contiguous) - the same
/// [out_features, in_features]-row-major, in-features-contiguous convention `gr.hpp`'s `hc_attn_down`/
/// `hc_attn_up` use. A no-op when `s.gate_rank <= 0` (the "no gate" fallback).
void loop_gate_mix(const float* R, const float* R0, const uint16_t* down, const uint16_t* up,
                   const LoopGateShapes& s, float* workspace, float* R_out, void* stream);

}  // namespace strata::kernels
