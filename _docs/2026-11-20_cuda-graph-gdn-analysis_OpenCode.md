# CUDA Graph / GDN engineering notes (2026-11-20)

## Current graph usage (direct source reading)

`src/qwen38/engine.cu`:
- `gdn_graphs[16]`, `ffn_graphs[16]`, `head_graph` (lines 75-78), captured in
  `ensure_decode_graphs()` (447-455) and launched per token at 479/488/494.
- Capture uses `cudaStreamBeginCapture`, `cudaGraphInstantiate`,
  `cudaGraphExecDestroy` (438-444). Graphs are decode-only; `conservative` and
  `fp32_decode` disable them.
- One decode token launches 16 GDN group graphs + 16 FFN group graphs + 1 head
  graph = 33 graph launches, with a `LayerFence` between groups for STOP,
  sticky numerical status and resource checks.
- Prefill has no graph: `run_prefill_chunk` calls cuBLAS and KV kernels directly
  with dynamic `n_tokens` and `pos_start`, so captured kernel parameters would
  be stale.

## Qwen3.5-specific structure

Geometry: 3:1 GDN-to-full-attention layers (48/16 for 27B, 24/8 for 9B).
GDN layers use fused kernels `fused_gdn_conv_l2(_batch)` and
`fused_gdn_step_norm_silu(_batch)` from `src/kernels/cuda/fused_gdn.cu`, called
at engine.cu:381-386 (decode) and 547-548 (prefill). Their Tensor Core
utilisation and launch geometry are being analysed by a subagent.

## Candidate engineering directions (not yet implemented)

1. Collapse the 33 per-token graph launches into a single captured step graph.
   Benefit: lower launch overhead. Risk: coarse STOP granularity; must preserve
   cancellation and resource fences. A hybrid (full-step graph plus a fence
   before/after) is the safe variant.
2. Capture a fixed-size prefill chunk graph keyed by `(chunk, kv mode)`.
   Benefit: remove CPU launch overhead for 848 GEMM calls. Risk: cuBLAS capture
   support, dynamic KV positions, varying `n_tokens` at chunk boundaries.
3. Evaluate whether GDN recurrence kernels can use `mma.sync` or warp-level
   primitives. Risk: recurrence is sequential; only the projection-heavy part
   is GEMM-shaped and already uses cuBLAS Tensor Cores.

## Detailed findings from the analysis subagent

- The GDN recurrence kernels (`src/kernels/cuda/fused_gdn.cu:21-227`) are scalar
  FP32 with only warp shuffles, no Tensor Cores. Their cost is ~1.0 ms/step
  versus ~11.5 ms FFN and ~9.3 ms projections, so rewriting them with
  `mma.sync` is explicitly **not** recommended: the GEMVs have too little N per
  head, the rank-1 delta update is dependency-coupled, SM120 lacks tcgen05, and
  prior WMMA/INT8 pilots were 1.9-4.5x slower.
- Decode already fuses alpha/beta via `mm_dual_h` (`engine.cu:417`), but prefill
  issues two skinny N=32/48 cuBLAS GEMMs (`engine.cu:594-595`). Fusing them into
  one wide GEMM is a candidate, though earlier call-count reductions (848->248)
  showed no time gain, so the win would have to come from the larger N.
- Decode captures 33 graph launches per token (16 GDN + 16 FFN + 1 head). The
  attention Q/K/V projection (`rms_h` + `mm_triple_h`) is capturable; the
  position-dependent RoPE and KV enqueue must stay outside. This is the lowest
  risk graph improvement and is now implemented in
  `H:/Strata/builds/q38-9b-graph-20261120` (verification pending).
- Hadamard of the shared hidden activation is now reused in prefill
  (`brot_hidden_ptr`), but decode still recomputes it per folded weight
  (`engine.cu:320`); that is a Hikari-only follow-up.

## Evidence status

The tiled GQA attention kernel (`attend_batch_gqa_tiled`, BR=4, batched
reductions) passed CTest 18/18 in `q38-9b-tiled-20261120`; A/B measurement is
in progress. The attention-projection graph build is also pending verification.
The accepted speedups remain the attention/softmax/RoPE/GQA and evolutionary
score-buffer results documented elsewhere.
