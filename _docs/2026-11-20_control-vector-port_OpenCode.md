# Control-vector (refusal-direction) feature ported to Q38 (2026-11-20)

## Source

Strata's existing feature is an app-owned boolean `refusal_direction_removal`
(`apps/host/server/native-settings.mjs:12`) that constructs these engine flags
(`apps/host/server/refusal-projection.mjs:62`):
`--control-vector-scaled FILE:1.0 --control-vector-layer-range 4 44
--cvec-mode project --cvec-dir per-layer`.
The CUDA math is in `include/strata/kernels/cvec.hpp`: after a steered layer's
FFN write, `h <- h - s_l (h . v_l) v_l` (project) or `h <- h + d_l` (add).

## Implementation in the dense Q38 runner

- `src/qwen38/cvec_kernels.cuh` (new): device projection kernel, host
  `CvecRuntime`, and a GGUF loader for `direction.<layer>` F32 tensors. The
  vector must have one tensor per layer whose width equals the model `hidden`;
  any mismatch is a hard, explicit error. Project mode normalises `d_l` and
  keeps `s_l = |d_l|`.
- `src/qwen38/engine.cu`: the four CLI flags above are parsed and passed to the
  Engine. The vector is applied after each layer's final FFN residual write in
  both the prefill loop and the decode layer emitters (which are captured in
  the CUDA graphs). No vector means the code path is skipped entirely, so the
  default output stays bit-identical.

## Verification

- Build (CUDA 13.5 / SM120 / 5 jobs): CTest 18/18.
- Without a vector: 9B fixture token sequence `4183,5543,401,632` — identical
  to the pre-port binary.
- Bundled Strata vector (Qwen4Exp, hidden 2560) against 9B (hidden 4096):
  rejected with `no direction.<layer> tensor with matching hidden size`.
- Synthetic matching vector (hidden 4096, layers 1..44, scale 1.0): loads,
  applies, and produces finite tokens (`4183,5543,401,632` — the tiny synthetic
  direction does not change the argmax). With scale 1000 the run is rejected by
  the existing non-finite guard, as expected for an out-of-range intervention.

## Limits

- Only `per-layer` direction lookup is implemented; `--cvec-dir single:L` is
  parsed but not supported.
- Only F32 `direction.<l>` tensors are loaded.
- No quality/benefit evaluation of any real refusal vector was possible: the
  bundled vector is for a different architecture (hidden 2560). The user must
  supply a control vector whose width matches 5120 (27B) or 4096 (9B) and
  validate behaviour themselves.
- This feature changes responses; neither answer quality nor speed is
  guaranteed, matching the upstream setting's own warning.
