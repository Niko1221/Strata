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
