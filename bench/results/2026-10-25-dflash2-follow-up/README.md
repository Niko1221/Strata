# DFlash2 projection and selection follow-up, 2026-10-25

This continues the [earlier code-path study](../2026-10-25-dflash2-code-optimization/README.md).
All additions remain opt-in. Component improvements do not establish an increased
whole-model advantage over MTP. Full-request studies are running at this snapshot.

Hardware: RTX 4070 Ti SUPER 16 GiB / SM89, driver 610.57.04, Ryzen 9 5900X,
about 125 GiB RAM, Linux, GCC 14.2 and CUDA 13.4.92. Desktop applications remain
open and clocks are not locked. These are local experiments, not standard
benchmark-protocol results.

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
concurrent requests, vision or multi-GPU. Full-model speed for the new selection
kernels and the combined profile is pending in this snapshot. No default changed.

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
