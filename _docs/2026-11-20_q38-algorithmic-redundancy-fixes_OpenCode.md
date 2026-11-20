# Q38 exact algorithmic redundancy fixes (2026-11-20, OpenCode)

Trigger: code-analysis subagent found redundant computations that can be removed
without changing numerics beyond the current 3e-5/4e-5 tolerance contract.

## Findings and actions

1. **Softmax exponential recomputed 256x per position (largest).**
   `src/qwen38/kv.cu` `attend_batch` previously evaluated
   `expf(s[t]-maximum)/denominator` independently in all 256 threads of the
   value-accumulation loop, even though `s[t]` does not depend on the component
   index. Fixed by computing each probability once into dynamic shared memory
   (32 KiB cap, 8192 positions) and reusing it; positions beyond the cap keep
   the original expression as a fallback. The `maximum`/`denominator` reductions
   and the expf inputs/order are unchanged, so results are bit-identical.
   Same pattern remains in the legacy `attend`/`aggregate_kernel` paths; those
   are not on the 9B f16 prefill path and were left for a separate change.

2. **RoPE inverse frequencies recomputed per head/token.**
   `src/qwen38/ops.cu` `prep` computed `powf(base,-p/32)` for all 64 active lanes
   of every (head, token) block. Now the 32 values are computed once per block
   into `__shared__ inv_freq[32]` with the same device `powf`, then reused.
   Bit-identical angles; removes ~14.7M `powf` per 512-token, 16-layer chunk.

3. **Redundant output zeroing in decode `enqueue`.**
   `src/qwen38/kv.cu` `enqueue` zeroed `output` and `aggregate_` before kernels
   that fully overwrite both. Removed (one fewer kernel launch and one fewer
   full-buffer write per token).

NOT changed: the per-layer resource checks (`host_resources` + `cudaMemGetInfo`
in the fence lambda). They are host-side and cost little, but removing them
would weaken the configured RAM/commit/GPU floors, which the PC standard
requires to be maintained. Left intact deliberately.

## Verification

- Separate build directories were used so the running Shinka evolution's binary
  was not overwritten: `H:/Strata/builds/q38-9b-expf-20261120` (item 1) and
  `H:/Strata/builds/q38-9b-algo-20261120` (items 1+2+3).
- Both builds were configured with CUDA 13.5 / SM120, `--parallel 5`.
- CTest results to be recorded here when the background builds finish.
- End-to-end A/B on the pinned 9B 1024-token fixture (1024 ids, ctx 2048,
  `--kv f16`, `--seed 42`, separate build for each binary):
  - baseline: token `15217`, prefill 817.219 ms
  - expf fix: token `15217`, prefill 807.238 ms (about 1.2%, within run-to-run
    noise on this loaded machine). Numerics are identical as designed.
  - Interpretation: the removed `expf` was loop-invariant in the value component,
    so nvcc had likely already hoisted it; the manual cache is correctness-safe
    but not a meaningful speedup by itself.
- Full algorithmic set (expf cache + RoPE table + memset removal, build
  `q38-9b-algo-20261120`), single run: baseline 670.066 ms, new 607.918 ms,
  token `15217` for both.
- Three-run medians (all six runs produced token `15217`):
  - baseline: 563.3 ms (samples 563.3, 567.934, 556.9)
  - algo: 534.422 ms (samples 534.422, 535.526, 532.288)
  - median speedup **1.054x (5.4%)**, numerically identical. This is the
    accepted algorithmic gain from this session.

## Residual risk

- The shared-memory cache is capped at 8192 positions; prompts with absolute
  positions beyond that use the original path, so long-context speedup is not
  claimed.
- Item 1 changes launch-time dynamic shared memory size; `preload_kv` queries
  the kernel attributes, but no explicit `cudaFuncSetAttribute` is needed at
  32 KiB on SM120.
