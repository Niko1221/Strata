# Q38 9B/27B Tensor Core tuning — implementation record (2026-11-20, OpenCode)

## Overview

Follow-up to `_docs/2026-11-20_q38-9b-native-implementation_OpenCode.md`.
Profiled the 9B prefill, found the dominant cost was per-token launch overhead
in the f16 attention path, replaced it with batched kernels, and fixed a
critical 27B-hard-coded head-stride bug that corrupted 9B attention buffers.

## Background / requirements

User requested additional CUDA 13.5 / Tensor Core tuning to speed up both the
9B and 27B native paths. Earlier measurement: 9B 1000-token prefill 1623 ms vs
Prism 393 ms; the previous implementation log recorded the speed target as
unmet.

## Assumptions / decisions

Profiling used the built-in `STRATA_Q38_PROFILE=1` stage timers on the pinned
9B GGUF, 1000-token fixed prompt, ctx 2048, `--kv f16`, five-job build
`H:/Strata/builds/q38-9b-geometry-20261120`, CUDA 13.5.37 / SM120, RTX 5060 Ti
16 GB. No user process was stopped; GPU stayed free of competing compute jobs.

## Changed files

- `src/qwen38/kv.hpp`, `src/qwen38/kv_config.cpp`, `src/qwen38/kv.cu` — `KvShape`-driven batched f16 attention, bounded score staging, matching VRAM accounting
- `src/qwen38/ops.cu`, `src/qwen38/ops.cuh` — head-count parameters for prep/score/attend kernels and wrappers
- `src/qwen38/engine.cu` — pass geometry head counts; opt-in `STRATA_Q38_PREFILL_CHUNK` / `STRATA_Q38_PREFILL_TILE_MB` overrides
- `src/qwen38/numerics.cu`, `src/qwen38/kv_numerics.cu` — updated test call sites
- `src/qwen38/README.md`, this log

## Implementation details

Profile before the change (`gdn_mm=186.55 gdn_core=50.38 attn_mm=21.56
attn_kv=1177.57 ffn=186.46 fence=1.53`, total 1652.9 ms) showed attention KV
consumed 71% of prefill. The f16 branch of `KvCache::enqueue_batch` looped over
tokens and called the single-token `enqueue()` (memset, four finite checks, two
history stores, `score`, `attend`, final finite), producing tens of thousands of
launches for a 1000-token prompt and using only 24 blocks for the attend step.

New f16 batched path: one `store_f16_batch`, one `score_batch` and one
`attend_batch` per bounded sub-batch. `score_batch` / `attend_batch` reproduce
the original accumulation order and gate exactly. The score buffer is capped
at 32 MiB; the number of tokens per sub-batch is derived from it, so VRAM is
independent of context length and is included in KV scratch/admission.

Critical bug found during determinism checks: `prep`, `score` and `attend` in
`ops.cu` hard-coded 24 Q heads, 4 KV heads, 12288 Q rows and 6144 output rows.
For 9B (16 Q heads, 8192 Q rows, 4096 output rows) this wrote past the Q/gate
buffers and made logits nondeterministic. All three kernels and their wrappers
now take `q_heads` / `kv_heads` from `Geometry` / `KvShape`.

## Commands run

- `STRATA_Q38_PROFILE=1 strata-q38 --model <9B> --ctx-size 2048 --tokens <1000 ids> --n 1 --kv f16`
- `cmake --build H:/Strata/builds/q38-9b-geometry-20261120 --parallel 5` and `ctest --output-on-failure`
- Fixed-token runs with `--seed 42` on both profiles; PrismML 0.4.1 `llama-server` completion/timings for the same token IDs
- Tile/chunk sweep with `STRATA_Q38_PREFILL_CHUNK` / `STRATA_Q38_PREFILL_TILE_MB`
- FP8 run with `STRATA_FP8_PREFILL=1`

## Verification results

- Build and CTest: 18/18 passed, including the existing `kv_numerics` reference comparison of the batched path (max attention error 1.5e-06, tolerance 3e-5/4e-5) and `kv_cpu` exact-fit boundaries.
- 9B 1000-token prefill: 1652.9 ms → 618.4 ms (2.7x). Final stage profile `gdn_mm=179.00 gdn_core=45.62 attn_mm=18.19 attn_kv=149.22 ffn=171.11 fence=1.02`; attention KV fell from 1177.6 ms to 149.2 ms (7.9x).
- 9B determinism: three `--seed 42` runs now produce identical tokens `4183,5543,401,632`; before the head fix they differed every run.
- 27B regression: `4183,344,248046`, identical to the installed 0.6.3 binary; MTP path previously confirmed with `mtp_ready=1`.
- 140-token comparison (Prism ML 0.4.1 commit `1965755`): Prism prompt 204.25 ms, decode 1899.35 ms/128 tokens = 14.84 ms/token; native prefill 305.08 ms, decode 2034.72 ms/127 steps = 16.03 ms/token.
- Tile sweep (1024 tokens): 20 MiB 570.7 ms, 64 MiB 572.9 ms, 128 MiB 580.9 ms, 256 MiB 573.8 ms; chunk 512/64 MiB 658.0 ms, 512/128 MiB 679.5 ms, 256/128 MiB 875.6 ms. No gain over the built-in candidate.
- FP8 with an aligned 1024-token chunk: 616 ms, `fp8=784` GEMMs, slower than warm FP16 565–585 ms. Not adopted.
- FP32 accumulation diagnostic (`STRATA_Q38_FP32_ACCUM=1`, added this session): 912.9 / 832.2 ms versus FP16 638.3 / 590.4 ms on the same 1024-token fixture. Much slower; the opt-in remains only as a diagnostic and must not be enabled in production.

## Residual risks

- "Faster than llama.cpp" remains unmet: native is about 1.5x slower on prompt processing and about 8% slower on decode for this 9B checkpoint. Reaching that target needs a different weight math path (e.g. persistent dequantized weights or a direct quantized Tensor Core GEMM), not launch-count tuning.
- The 32 MiB batched score buffer is allocated for f16 KV even for decode-only sessions; it is accounted in admission but is a fixed overhead.
- The batched path covers f16 only; `k8v4`, Spin8 and Turbo modes still use the per-token loop.
- General model quality, full-window inference, concurrency and GUI/Control E2E remain unqualified. `releaseQualified=false` unchanged.
- Unrelated uncommitted work in the working tree was not staged or modified.

## Recommended next actions

1. If the speed target is mandatory, prototype a persistent FP16 weight arena for prefill (one dequantization per weight per session) or a direct Q4_K Tensor Core GEMM, and compare with the same fixed-token fixture.
2. Extend the batched attention approach to `k8v4`/Turbo modes if those modes are used in production.
3. Re-run GUI/Control acceptance only after a signed component build exists.
