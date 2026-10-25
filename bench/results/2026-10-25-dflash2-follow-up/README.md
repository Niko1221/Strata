# DFlash2 projection and selection follow-up, 2026-10-25

This continues the [earlier code-path study](../2026-10-25-dflash2-code-optimization/README.md).
All additions remain opt-in. Component improvements do not establish an increased
whole-model advantage over MTP. Short full-request studies completed; the final
256K study was interrupted without a completed control result.

Hardware: RTX 4070 Ti SUPER 16 GiB / SM89, driver 610.57.04, Ryzen 9 5900X,
about 125 GiB RAM, Linux, GCC 14.2 and CUDA 13.4.92. Desktop applications remain
open and clocks are not locked. These are local experiments, not standard
benchmark-protocol results. The target uses the local
`Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata` pack and dense GGUF,
with BF16 embeddings, FP8 PLE and the official DFlash2 Q8_0 artifact.

## Implemented changes

`STRATA_DFLASH2_PROMPT_GRID=1` combines the existing eight-row Q4_0/Q5_0/Q8_0
projection groups into one grid per matrix. It requires
`STRATA_DFLASH2_PROMPT_BATCH=1`. Per-row dot products and reduction order stay
the same. Tails retain the original smaller projection. Unsupported layouts,
including gfx906's wave64 dispatch, retain the old path.

`STRATA_DFLASH2_WARP_TOPK=1` reduces parallel top-k candidates within warps,
then merges eight warp winners. Separate winner storage protects readers across
iterations. `STRATA_DFLASH2_MERGE_TOPK=1` merges already sorted tile lists by
keeping their current heads, rather than rescanning every candidate at each
selection. It handles up to 64 tiles; other shapes retain the old merge. Both
preserve finite-value filtering, smaller-token tie breaking and score bits.

## Component measurements

| Component and comparison | Before | After | Lower latency |
| --- | ---: | ---: | ---: |
| 512-row context: matrix grouping → one-grid projections | 9.424 ms | 7.752 ms | 17.7% |
| Top-k: original parallel → warp reduction + list-head merge | 89.118 µs | 70.231 µs | 21.2% |
| Top-k: existing register tiles → register tiles + list-head merge | 53.370 µs | 38.741 µs | 27.4% |

Context measurements use the real official Q8_0 weights and synthetic padded
target-feature planes. They alternate 24 paired samples after warmup, exclude
input upload, and use cached graphs in both grouped variants. See
`grid-component-summary.json` and `grid-paired-q8.log`. The earlier original
encoder → grid prototype comparison is preserved separately in that JSON;
it must not be combined with the incremental percentage above.

Top-k uses seven rows, vocabulary 248320, K=16, synthetic logits and hot data.
A graph contains ten top-k calls. Each timing round replays it 200 times;
the reported value is the median of eight rounds. These are component timings,
not whole-model throughput. The final V5 samples are in `production-v5-topk.json`.
Earlier forward/reverse process sweeps are retained in `topk-graph-results.json`
and `topk-register-sweep.json`; rounds within a process are not independent
hardware trials. The gains are not additive.

Reproduce with an idle CUDA GPU and an existing local checkpoint:

```sh
cmake --build build-dflash2 --target strata dflash2_extension_parity dflash_prompt_fusion_parity -j4
build-dflash2/dflash_prompt_fusion_parity /path/to/dflash2-Q8_0.gguf --grid --bench
build-dflash2/dflash2_extension_parity --topk-bench
STRATA_DFLASH2_WARP_TOPK=1 STRATA_DFLASH2_MERGE_TOPK=1 \
  build-dflash2/dflash2_extension_parity --topk-bench
STRATA_DFLASH2_REGISTER_TOPK=1 STRATA_DFLASH2_MERGE_TOPK=1 \
  build-dflash2/dflash2_extension_parity --topk-bench
```

## Vector selector

`STRATA_DFLASH2_VECTOR_SELECTOR=1` reads four BF16 codebook values and four
hidden values per iteration. Four separate score updates retain the original
FMA order. Dispatch requires rank divisible by four, eight-byte codebook alignment
and sixteen-byte hidden alignment; other inputs retain scalar reads. Reference
and serial modes remain unchanged. The option also supports the existing lattice
selector. It remains off by default.

Final CUDA Graph measurements, with confidence enabled:

| Selector | Scalar | Vector reads | Lower latency |
| --- | ---: | ---: | ---: |
| Existing parallel candidates | 62.290 µs | 41.387 µs | 33.6% |
| Existing lattice | 21.543 µs | 18.795 µs | 12.8% |

The fixture uses synthetic hot data, eight rows, rank 256 and K=16. Each graph
contains ten calls, with 20 warmup replays and eight timing rounds of 100 replays.
Two processes per configuration run in forward/reverse order. Each process is
summarized by its median round; the table uses the median of those two values.
Individual rounds are not independent trials. `selector-graph-final.json`, its
logs and `selector-graph-metadata.json` preserve samples and source/binary hashes.
The earlier host-enqueued screen is retained in `vector-selector-results.json`.

The expanded fixture checks ranks 1/3/4/12/256/260/1024, rows 2/3/8 and K up to 64;
IDs and confidence match the scalar reference exactly. Misaligned pointers,
nonfinite hidden values and negative anchors retain the expected behavior.
Ordinary full extension fixtures pass with vector reads both with and without
lattice. Vector Memcheck reports zero errors; vector+lattice Racecheck reports
zero errors and warnings. The graph fixture was added after those sanitizer
runs; it changes measurement orchestration, not the tested device algorithm.

SM89 instruction bytes remain identical for both original scalar selector kernels
(7552 and 2560 bytes), in addition to the 414 original MMVQ kernels and original
top-k. `default-v7-text.json` records this final check. The V7 engine SHA256 is
`ed1a9578fe55c5dd5a1dabc2f968301fd0816b6d4dc8458122d5e886f97de7d9`.
The full-model `vector384-smoke` run used one short-code prompt, a warmup, one
measured 384-token request and a same-process target-only check; every token also
matches the common separate control. This validates that case, not a model-level
speed claim. CUDA extension ON/OFF builds passed; HIP/SYCL were not available.

```sh
STRATA_DFLASH2_VECTOR_SELECTOR=1 \
  build-dflash2/dflash2_extension_parity --selector-bench
STRATA_DFLASH2_VECTOR_SELECTOR=1 STRATA_DFLASH2_LATTICE_SELECTOR=1 \
  build-dflash2/dflash2_extension_parity --selector-bench
```

## Short-request screen

The completed 128-output screen used four fixed prompts of 27, 2427, 2425 and
7227 input tokens. Each task had one warmup and three measured requests in each
process, followed by a target-only check in that process. All token sequences
also matched a common, separate target-only process. Context/resident limits were
8192, expert cache 2900 slots, reserve 480 MiB, prefill 512, short-read 64,
prompt cache disabled and speculative allocation 8. DFlash2 used the official
Q8_0 artifact, block 7 and floor 0.7; MTP used max-T 4 and floor 0.5.

The prompt batching ABBA screen gave these median prompt times, in milliseconds:

| Arm, in execution order | Short code | Long code | Cache | Very long code |
| --- | ---: | ---: | ---: | ---: |
| Original A | 439.3 | 5916.6 | 5897.3 | 16851.4 |
| Batch A | 438.9 | 5903.7 | 5881.4 | 16797.8 |
| Batch B, valid retry | 438.9 | 5908.7 | 5893.5 | 16832.2 |
| Original B | 438.4 | 5907.6 | 5893.4 | 16841.7 |

These whole-prompt differences are small compared with the component gain and
some of the process variation. They do not establish a general prompt speedup.

The separate grid screen's decode medians were 47.676/41.808/46.862/46.017 tok/s
before, 46.315/41.459/48.008/47.159 with the grid, and
47.126/41.440/47.945/47.517 in the closing baseline. The order follows the table's
tasks. The changing baseline prevents interpreting these differences as a stable
decode gain. MTP's same-day medians were 36.198/37.105/46.189/43.328 tok/s;
these establish context for the existing DFlash2 advantage, not an increase due
to this change. The MTP arms were not interleaved with every later screen.

A selector-overlap prototype was tested twice between two baselines. Its medians
were 47.106/41.428/48.052/47.458 and 47.062/41.378/47.956/47.495 tok/s;
the surrounding baselines were 47.142/41.470/48.008/47.548 and
47.220/41.464/47.976/47.538. No consistent gain was found. The prototype was
removed. The historical `pipeline128` arm also contains that removed selector;
it is not the submitted production profile. The current profile is measured
separately by the 384-output and final 256K studies.

Every DFlash2 arm retained accepted/offered counts 92/113, 79/113, 80/95 and
86/111 for the four tasks. `full-request/` preserves all warmups, samples,
controls, command arguments, flags, hashes and token arrays. `summary.json`
contains descriptive medians and ranges; it is not a confidence interval.
`collect_full_request.py` checks token arrays against the common controls and
archives only completed profiles. It records hashes of the original JSON files
before replacing absolute local data paths with `${STRATA_DATA}`.

The batching screen used binary SHA256 `0a3d3a5e…47520`; grid/overlap screens
used `8dff202104…80744`, which still contains the rejected overlap option.
Full hashes are in each raw JSON. No scratch-reuse claim relies on these older
executables.

## 384-output selection study

This study used three tasks (27/2427/2425 input tokens) and 384 outputs, with the
same allocation settings and floors as the short screen. Every process had one
warmup and three measured requests per task, same-process target-only checks,
and the same separate target-only control. The fixed binary is V5, SHA256
`17d3264b2537ffc2e40855604b3b39ac0292544a719c4540f7ec561371e08917`.
It contains the submitted grid and top-k code, before scratch reuse and the later
vector selector. All eleven processes completed and all controls passed.

Median decode tok/s, in actual process order:

| Arm | Short code | Long code | Cache |
| --- | ---: | ---: | ---: |
| default384-a | 43.549 | 42.615 | 45.255 |
| topk384-a | 45.495 | 43.368 | 44.374 |
| register384-a | 45.658 | 43.471 | 45.611 |
| register-merge384-a | 44.621 | 42.677 | 44.594 |
| register-merge384-b | 44.685 | 42.691 | 44.779 |
| register384-b | 44.487 | 42.697 | 44.817 |
| topk384-b | 44.238 | 42.393 | 44.811 |
| default384-b | 44.602 | 42.641 | 44.824 |
| pipeline384-a | 44.197 | 42.530 | 44.639 |
| pipeline384-b | 44.373 | 42.691 | 44.905 |
| default384-c | 44.536 | 42.395 | 44.451 |

`default` retains the existing tiled top-k; `topk` adds warp reduction and list
merge; `register` uses the existing register tile kernel; `register-merge` adds
the new merge to it. `pipeline` enables prompt batch/grid, direct quantization,
context graph, BF16 tile, register top-k and list merge. It does not enable the
removed selector overlap or the later vector selector.

No stable whole-model throughput gain is established. For example, the first
register process is faster than its later repeat, while adding the merge does
not consistently improve the full request despite its lower isolated latency.
The closing baselines also drift. All individual values and ranges are retained
in `full-request/summary.json`; no best-run percentage is used as a speed claim.
Accepted/offered counts stay exactly 279/368, 246/322 and 266/327 for the tasks.
The final scratch-reuse executable is evaluated separately by the 256K study.

The existing runtime timing diagnostics explain the limited impact. A measured
short-code request in `default384-c` reports 72.91 ms/window: verification
68.65 ms (including 54.05 ms waiting for CPU expert jobs) and draft 4.23 ms.
The long-code request reports 65.20 ms/window: verification 61.00 ms, CPU jobs
46.92 ms and draft 4.17 ms. These diagnostic categories are the engine's own
accounting, not an independent hardware profiler. `full-request/decode-timing.json`
retains every timing line and the matching request labels; source log hashes are
included. Saving tens of microseconds in a selection kernel does not remove the
measured CPU expert wait.

## Prompt scratch memory

After FC normalization has been converted to BF16, its float buffer is dead.
The official geometry can reuse it for K/V outputs. Other geometries keep separate
allocations. BF16 artifacts also omit the unused Q8_1 buffer. The existing prompt
batch option remains required; defaults allocate neither form of extra scratch.

The drafter's accounted extra device bytes for a 512-row batch changed as follows:

| Artifact | Before | After | Saved |
| --- | ---: | ---: | ---: |
| Q8_0 | 31.0547 MiB | 27.0547 MiB | 4 MiB |
| BF16 | 31.0547 MiB | 20.0234 MiB | 11.0313 MiB |

These are measured `vram_bytes()` counter differences between two uploaded
instances, not measurements of driver allocation rounding. `memory-v6-summary.json`
and the before/after logs preserve the values. The V6 grid timing repeat is also
in that JSON. Q8/Q4/Q5 against the unbatched encoder, BF16, distant positions and
Memcheck all passed after the buffer reuse. Only the owner allocation is freed.

```sh
build-dflash2/dflash_prompt_fusion_parity /path/to/dflash2-Q8_0.gguf --memory-only
```

## Validation

Passed on the tested revisions:

- CUDA extension ON/OFF builds, including the final scratch-reuse candidate.
- Final focused CTest: seven passed, one model-argument skip with the extension
  enabled; six passed with it disabled. Model-dependent fixtures ran explicitly.
- Exact raw projection checks for Q4_0/Q5_0/Q8_0, widths 96/2560/7680/12800,
  odd output widths, 16–512 rows and tails.
- Q8_0/Q4_0/Q5_0 model-dependent full physical KV comparison; graph-disabled
  Q4_0 and distant-position Q5_0 checks. The Q8_0 grid fixture compares 928,153,600
  bytes across ring wraps and absolute positions 262141/263679, exactly.
  These synthetic features do not constitute a full 256K prompt generation.
- Independent CPU top-k oracle, signed-zero ties, nonfinite and all-invalid
  rows, fewer finite values than K, and 4096-token / 64-tile boundaries,
  including the 65-tile fallback.
- Final top-k Racecheck: zero errors or warnings; Memcheck: zero errors.
  The compact checks cover 4097 and 262144 vocabularies. The larger ordinary
  oracle suite is separate from these sanitizer runs.
- Prompt-grid Memcheck: zero errors, including the final scratch-reuse revision.
- Static SM89 comparison reconfirmed on final V6: all 414 original MMVQ kernels
  (5,528,960 bytes) and the original top-k kernel (9,856 bytes) retain identical
  instruction bytes. This is not a claim that the entire executable is identical.
  `compare_default_text.py` reproduces this comparison from preserved executables;
  `default-v6-text.json` records their hashes and the final comparison.

Not run: HIP/SYCL builds or hardware execution, Metal, Windows, other GPUs,
concurrent requests, vision or multi-GPU. The completed 384-output screen does not establish a stable full-model speedup.
The final 256K attempt did not produce a completed result. No default changed.

## 256K attempt and limits

The final V6 scratch-reuse engine was queued for seven cold-process arms:
target-only, baseline A, optimized A, MTP A, optimized B, baseline B and MTP B.
The prompt contains 261747 input tokens, with an output budget of 384, context
262144, requested resident limit 32768, expert cache 2700 slots, reserve 480 MiB,
prefill 512 and short-read 64. V6 excludes the later vector selector.

Only the target-only arm started, at 19:25:26 UTC. At 21:43:36 UTC, after the
renewed five-hour deadline 21:38:15 UTC, no benchmark processes remained and
neither a completed target result nor a failure record existed. The cause of
interruption is not established. Startup logged 2700 resident experts and 1703 MiB
free VRAM, but startup alone cannot establish prompt completion, performance or
token parity. `long-study-status.json`, the exact command and startup log preserve
this incomplete attempt. No 256K throughput or comparison against MTP is claimed.
The earlier distant-position synthetic KV fixture is separate from this attempt.

## Rejected experiments

All three shared-GQA attention variants produced bit-identical dumped outputs
for all 30 timing cases in four processes each and passed the FP64 oracle.
They were slower and are absent from production. For page size 4, eight rows,
end position 2053, medians of two process measurements were approximately:

| Prototype | Baseline | Candidate |
| --- | ---: | ---: |
| Shared K/V, 64 cells | 157.8 µs | 212.4 µs |
| Shared K only, 64 cells | 160.4 µs | 173.1 µs |
| Shared K only, 32 cells | 156.5 µs | 172.1 µs |

The JSON files retain every shape and sample. Experimental patches apply to
`f2543017`; the two `complete` patches also contain the then-uncommitted grid,
selector and top-k prototypes. They are reproduction artifacts, not production
options. The `SHARED_GQA` variable is absent from submitted runtime code.

A biased unsigned AVX2 NVFP4 dot retained fixture parity but was slower in the
eight-token DRAM screen. L2 and non-temporal prefetch hints retained parity;
the small timing differences did not establish a useful model-level gain.
These CPU tests use synthetic weights/activations and eleven threads over a
4.2 GB working set. They are not model throughput or quality measurements.
`biased-bw.json` and `prefetch-results.json` preserve all runs. No CPU kernel
change is submitted.

One attempted second batching-arm startup overlapped the still-running grid
sanitizer. Both processes for that arm were stopped before measured repetitions;
its directory is marked invalid locally. A fresh `prompt-batch128-b-retry`
completed against the common target-only control. The invalid start is excluded.
