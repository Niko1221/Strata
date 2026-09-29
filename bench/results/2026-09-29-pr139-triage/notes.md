# PR 139 triage: prefill BF16 on pre-Ampere GPUs

Date: 2026-09-29

Source: upstream pull request Niko1221/Strata#139, head commit
f4c86874a2758306d4d3b58dcbc990168f7b3554 (rluisr/Strata).

## Summary

PR 139 adds a BF16 prefill GEMM path for pre-Ampere GPUs (sm_70, sm_75).
The path expands BF16 to FP32 and runs tiled cuBLAS SGEMM.
It also lets `STRATA_PREFILL_RING=8` select routed-only expert staging
for large prompt chunks.

The fork already has a faster Volta GEMM path: BF16 weights convert to
FP16 and the product runs on FP16 tensor cores. The fork's measured
prompt speed is 3 to 5 times higher than the SGEMM path in PR 139.
The GEMM changes would also misinterpret the fork's FP16 activation
images as BF16 and produce wrong results. The GEMM part is not adopted.

The staging opt-in was tested on this fork. It gives parity at
`--prefill 4096` and is slower at the default `--prefill 8192`.
It is not adopted.

The testing exposed a build break on this fork's `main`:
`main` could not configure or link since the upstream merge
commit 4686c0f. This directory's parent commit fixes that break.

## A/B method

- Server: new build of `main` plus the two build fixes, fork engine.
- GPU: one Tesla V100-PCIE-16GB.
- Model: Qwen3.8-Flash-Next Q2_0, 262144-token context, INT8 KV.
- Requests: uncached API requests (`reused=0`), the exact row targets
  of `bench/run_v100_bench.py` (~8K and 32K rows), `max_tokens 64`.
- Each pass is one bench run (one request per row).
- The card cooled to 60 C or less before each configuration.
- Temperature and config order: control, ring 8, ring 8 with 4096,
  control again. The repeated control checks thermal drift.

## Results

Prompt tok/s, two passes per configuration:

| Configuration | ~8K prompt | 32K prompt |
|---|---:|---:|
| Control (default staging, `--prefill auto` = 8192) | 1162.7 / 1165.9 | 1454.4 / 1392.7 |
| Control, repeated after the test rows | 1161.4 / 1164.7 | 1453.4 / 1353.0 |
| `STRATA_PREFILL_RING=8` (routed-only staging, 8192 chunks) | 1075.8 / 1075.2 | 1305.5 / 1215.5 |
| `STRATA_PREFILL_RING=8` + `--prefill 4096` | 1165.5 / 1164.6 | 1457.4 / 1366.7 |

Routed-only staging at 8192-token chunks is 7 to 13 percent slower.
At 4096-token chunks it is parity with the default.
The default configuration remains the fastest.

## Correctness

A fixed prompt with temperature 0 produced byte-identical output in the
control and the ring-8 configuration. The tool-call diagnostics
(`tests/diagnose_openai.py`) pass 5 of 5 on the final control binary.

## Files

- `baseline-*.json`: the pre-merge engine (the README numbers).
- `ctrl-new-*.json`: new `main` build, default environment.
- `ring8-*.json`: `STRATA_PREFILL_RING=8` at the default 8192 chunks.
- `ring8-4096-*.json`: `STRATA_PREFILL_RING=8` with `--prefill 4096`.
- `ctrl-final-*.json`: control repeated after the test rows.
