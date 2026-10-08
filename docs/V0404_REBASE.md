# Concurrent throughput on upstream v0.1.40.4

This replaces PR #1209 with a performance-focused port onto pristine upstream
`6674a0065f` (v0.1.40.4). The previous head is preserved at
`archive/pr1209-before-v0404`, `0a3b624`. Experimental c8 and 24/32-row work is
excluded. The supported new path remains opt-in, c2-c4, at most 16 rows, on
eligible single-NVIDIA configurations. Existing AMD/multi-GPU paths remain;
hardware validation is limited to the local RTX 5090.

## What was retained, adapted or removed

| Area | Decision and reason |
| --- | --- |
| Wide speculative verification, per-slot suffix proposals, shared MTP weights/private state, overlap, merged experts, adaptive residency and optional scheduling | Retained from the performance commit. These are the throughput mechanisms being measured again. |
| Upstream batched QFUSE GDN correction (`0a22c46`) | Keep upstream's fix; no duplicate correctness commit. |
| Upstream shared MTP head type (`569c094`) | Keep upstream's copy. Retain our host vocabulary copy because per-slot drafting/suffix selection uses it. |
| Old QFUSE arithmetic and solo-commit correction | Not carried into this performance port. Upstream's batched repair covers only part of the old QFUSE commit; consumer-rounding and solo-commit differences remain separate issues, not claimed fixed here. Normal qualification uses QFUSE=0. |
| Python server and prefill telemetry | Omitted entirely, including protocol/docs/tests specific to that change. |
| New upstream interleaved MMVQ path | Preserve its dispatch. Give parallel dense groups separate interleaved scratch as well as separate ordinary quantization scratch; reset readiness when inputs change. |
| Upstream dense branches, PDL, Q4 query batching and PLE post-operations | Preserve upstream eligibility and code. Wide batch grouping does not activate solo-only optimisations. |
| Output projection | Tile wide rows into the existing supported kernel widths, retaining upstream interleaved dispatch where supported. |
| Solo and pipeline graph arrays | Bound capture/run widths to the existing eight-row solo limit; batch capacity is not a license to index solo graph arrays beyond eight. |
| Upstream request/pipeline-group changes | Retain current admission, group assignment, checkpoint, prefix reuse, cancellation and fallback implementation. |
| Old benchmark/qualification evidence | Do not present old measurements as fresh validation. New reports record new source and binary hashes. |

## Measurement plan

Use the same bounded essay and tool-driven coding fixture as PR #1209,
three alternating-order pairs per workload (seeds 123, 456, 789). The baseline
is unmodified upstream with `--batch-mtp`; candidate uses the documented
16-row configuration. Each process starts fresh and excludes warm-up.
Both allocate 98,304 context per request, request 2,560 MiB reserve, and use
the same explicit expert-cache capacity, model, prompts and sampling.
Hardware: RTX 5090 (undervolt removal previously reported by the user),
Ryzen 9950X3D, 48 GB DDR5-6000, Windows. No changed clock/voltage settings are
applied by this experiment.

Measure emitted-token throughput, not verification rows. Report stable c4
decode and whole-run throughput separately, plus native token gaps and
structural-check/turn-limit results. Generated code is not executed; unequal
outputs prevent a claim of equal-quality completion speed.

After performance screening, perform bounded native layout/state/expert
checks, controlled token comparisons and lifecycle checks. Keep normal fast
arithmetic performance separate from deterministic correctness controls.

## Completed qualification (8 October 2026)

- Twelve workload runs completed. Actual residency matched at 11,568 experts
  in every run; no cache retry or reserve shrink occurred.
- Median decode gains versus pristine v0.1.40.4: **1.31x essay, 2.07x coding**.
  Whole-run gains: 1.27x and 1.71x. See [full method and results](PR_WORKLOAD_BENCHMARKS.md).
- Native tests passed: 9,840 batch layouts, 100 bitwise partial state commits
  across 16 layouts, and the IQ2_XS/Q2_0 sixteen-row native expert fixture
  (zero failures).
- Four prompts x 128 tokens matched solo versus c4 within each build. The
  complete baseline/candidate dumps also matched: **512 solo plus 512 batch
  tokens** compared across builds.
- Both builds passed all eight lifecycle comparisons: concurrent admission
  during prompt work, continuation from a slot, a yielded prompt resuming,
  return from a batch to solo, and checkpoint reuse.
- These token tests control expert placement and CPU arithmetic: optimized
  IQ kernels disabled, IQ_MT_MIN=1, PCIe fraction zero, adaptation effectively
  off, no prefill borrowing. Performance runs use normal optimized arithmetic.
  This is bounded exact-token evidence, not bitwise internal-state equality
  across whole-model executions or universal normal-mode quality equivalence.
- Default interleaved MMVQ remains enabled in the tested builds. Newly added
  upstream opt-in PDL/dense-branch combinations, QFUSE-on, AMD, Intel and
  multi-GPU hardware are not separately qualified by this run.
- No Python server code changed, so the historical server-test counts are
  not recycled as new validation. No c=1 throughput speedup is asserted.

[Compact qualification](evidence/v0404-performance/qualification-summary.json),
individual native/lifecycle logs, token dumps, benchmark configurations and
artifact hashes accompany the PR. Runtime builds were produced in the same
local build checkout; the upstream executable built successfully before the
shared build wrapper requested candidate-only test targets absent upstream.
That wrapper's post-build target error did not change the baseline executable.
The candidate executable and relevant test targets all built successfully.
