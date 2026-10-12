# Async secondary refill: current-main refresh

Tested source `db5e8db73d47302027cc1b40418bae8f3c6fd10e`, merged with upstream
`61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406` and the refreshed #1336 cache
foundation. The new upstream async scheduler is retained. Secondary D2D copies
are queued by the owner before posting the existing upload worker; all-secondary
rounds still record completion/admit residency even with no H2D upload to enqueue.
The lifetime fixture now alternates mixed and all-secondary rounds. HIP error
aliases used by the snapshot header are mapped in both compatibility headers.

## Model checks and exactness limits

RTX PRO 6000 Blackwell 96 GB, Qwen3.8-Flash-Next Q8_0, FP16 KV, mmap PLE,
56 GiB resident-expert budget, 16 secondary-cache ways, adaptive swaps every four
windows (up to 96), PCIE fraction 0.55, 8192-token prefill chunks. Primary expert
capacity was 15,472 at 32K and 15,216 at 128K. Each long request reads its entire
prompt with no reuse and generates 1024 tokens. Cases use identical input hashes,
configuration and source/binary identity within each off/on pair.

| Input / width | Off decode tokens/s | Enabled tokens/s | Full output IDs |
|---|---:|---:|---|
| 32K / width 1 | 80.187 | 86.655 | Differ at token 218 |
| 32K / width 4 | 150.497 | 176.500 | Exact |
| 128K / width 1 | 93.437 | 84.233 | Differ at token 7 |
| 128K / width 4 | 134.417 | 170.312 | Exact |

Differences are zero-based. **Do not describe the width-1 rows as an exactness-
preserving speedup.** Each timed row is one fixed-order observation with warm
filesystem caches; output, admission timing and reported work can differ.
Dynamic CPU/GPU placement is a possible explanation, not a proved causal result.
These data do not establish a general speedup or model quality equivalence.

Four same-arm repeats used the identical binary hash, input hash, arguments and
environment. The 32K width-1 off arm changed at token 218 between repeats, while
its enabled arm matched itself. At 128K both arms matched their own repeats but
still differed off/on at token 7. Baseline variability therefore explains neither
all differences nor a general parity guarantee. The experiment remains opt-in.

An 8K/512-token default-off run matched current main exactly. Twelve primary
cases and four repeats completed and passed their pure arithmetic function
checks. Enabled width-1 and width-4 STOP runs cancelled after observing 7, 19 and
65 streamed tokens; all six drains completed and following function checks
passed. This does not prove a full internal KV/recurrent-state digest or exact
post-cancellation output in a dynamically placed Q8 model. Full IDs, native
statistics, environment, activation checks and individual cancellation results
are preserved in `model-results.json`.

## Component/backend checks

CUDA sm86 engine and read-only/compact byte fixtures passed on RTX 3070. The
secondary source-lifetime fixture passed, including snapshot invalidation,
mutation fencing, pending close and all-secondary completion. Compute Sanitizer
memcheck reported zero errors; racecheck reported zero hazards/errors/warnings.
These component checks do not substitute for whole-engine race analysis.

HIP gfx1012 and standalone SYCL builds passed. The snapshot compatibility fix
was rebuilt successfully on HIP. The async secondary-refill implementation is
CUDA-only; those other backend builds establish compatibility, not feature
runtime validation there. Main default-off parity is scoped to the model case
above; it is not a cross-hardware identity claim.

## Reproduction

Use the preserved `../reproduce/case.py` and `plan.json`, replace source/binary/
model paths, and run each labelled case on an idle GPU in new result directories.
Keep cache/snapshot settings equal and toggle only `STRATA_Q8_ASYNC_REFILL`.
`STRATA_Q8_CACHE_TAG_SNAPSHOT=1` and native `--adapt-async 1` are required. Layer
splits, pipeline windows, batched requests, helpers and device planning remain
refused. No production service or cache was modified.

Historical measurements remain separate at the original source. A laptop
restart interrupted one private case; completed rows were preserved and all
remaining cases ran detached in a fresh scratch directory. The corrected full
matrix took 1165.735 seconds unattended, and four same-arm repeats took 453.369
seconds. Setup/engineering time was not instrumented and is excluded.

## Tail-skip-off follow-up (2026-10-11)

A fresh four-case width-1 replay keeps the same current-main candidate binary,
inputs and settings, adding only `STRATA_ROUTE_TAIL_SKIP=0` to each arm. The 32K
off/on pair now matches all 1024 output IDs. The 128K pair still differs at
zero-based token 7. All four requests complete and pass their function checks;
full records, logs, exact configuration and comparisons are retained under
`tail-skip-off-followup/`.

This control removes the CUDA policy whose expert decisions depend on all rows
in a window. It supports a narrower statement for the 32K case; it does not
establish that tail skipping caused the remaining 128K difference. The original
rank-7 results and same-arm repeats remain above, with their original settings.
Async placement/admission and CPU/GPU execution remain separate factors. No
unconditional token parity, timing gain or answer-quality equivalence is claimed.

## Async admission diagnostic follow-up

The reproduction harness runs a 128-token function check before the long
benchmark, with adaptive placement active during that first request. A sparse
analysis-only CUDA build buffered eviction/admission event windows and selected-
swap hashes, emitting them after requests. It reproduced the 128K off/on output
difference at token 7 while exposing different history before the benchmark:
round 10 admitted the same 79-expert plan at window 43 in the off arm versus
42 in the enabled arm. At window 49, round 12 selected 10 versus 11 swaps with
different plan hashes. Both initial function outputs were identical (128 tokens)
and passed the arithmetic check. Placement history was therefore not held fixed
by this throughput comparison; the difference cannot be interpreted as a
same-placement numerical comparison.

A heavier diagnostic that printed full residency hashes every window made the
two 1024-token outputs and 1152 logged residency rows match; both matched the
original enabled arm. With logging disabled on that exact same diagnostic
binary, token-7 divergence returned. The observer changes timing. This control
supports timing sensitivity but does not prove that admission differences are
the sole cause, or rule out every possible engine-state defect.

All six diagnostic requests completed and passed their function checks. Full
plans, outputs, events, executable hashes, status records and the two minimal
analysis-only source patches are under `async-admission-diagnostics/`. They are
CUDA diagnostic builds, separate from the production PR source. The sparse
patch removes per-window I/O; it still adds instrumentation and does not prove
unobserved execution follows an identical schedule. The existing byte-copy
lifetime fixture and sanitizer results remain separate memory/fence evidence.

The opt-in async feature is retained as experimental. Exact output or quality
equivalence and a general speedup remain unestablished; disabling adaptation to
obtain equality would also remove the feature being measured and is not used
as its correctness certification.
