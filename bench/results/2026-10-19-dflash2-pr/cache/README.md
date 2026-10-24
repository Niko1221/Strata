# Cache parity and QFUSE commit correction — 2026-10-19

Hardware: RTX 4070 Ti SUPER 16 GiB, Ryzen 5900X, 125.7 GiB RAM, Linux,
CUDA 13.4, GCC 14.2, architecture 89. Branch codex/dflash2-main rebased on
fork/feat/dflash bc29f62f. Target: local Qwen3.8 NVFP4 pack, Q5_K target
head, FP8 PLE, BF16 embeddings, int8 KV. No weights downloaded.

## Findings and changes

The previous rebase cache failure compared different expert placements. A request
for 3500 expert slots exceeded available VRAM; the engine reduced it and reported the actual capacity, which the old test ignored:

| Drafter | Cached actual slots | Fresh actual slots |
| --- | ---: | ---: |
| Plain Q8 DFlash | 3169 | 3161 |
| Official Q8 DFlash2 | 3059 | 3065 |

The GPU and CPU expert paths round differently. These were not fixed-residency
cache comparisons. The original uncontrolled runs are archived locally; the table above is retained to identify why they were excluded. A diagnostic run
with extra fingerprints happened to pass with 3064/3065 slots, illustrating why
successful output alone does not establish a controlled comparison.

The corrected cache gate defaults to 2000 slots, captures native INFO, and
rejects a clamped budget or mismatched engine settings **before** comparing
outputs. It checks cached output against both a fresh drafter-enabled engine
and a fresh target-only engine. The sampling, greedy-probe and math benchmark
tools also validate actual placement. The old math --expert-cache-mib spelling
is retained as an alias but correctly documented as counting slots. Automatic
capacity benchmarks remain explicit and record the resolved capacity.

A separate runtime bug affected STRATA_QFUSE=1: record_window disabled the
one-token GDN state write, but commit still assumed the write had occurred and
skipped its graph. The commit condition now uses the same QFUSE exclusion.
Default QFUSE-off execution takes the same branch as before. The regression
compares the complete consumed target-state fingerprints for an identical
prefix read through single-token decoding and teacher-forced split windows.
The pre-fix binary fails with different GDN, PLE, indexer and KV state.
The corrected binary passes exact state and output comparisons for both the
consumed prefix and its continuation (qfuse-before/ and qfuse-after/).

CUDA build passed. All 39 Python tests and all ten selected C++/CUDA tests
passed, including GDN bitwise kernel parity. See the recorded exit codes; fresh checks for the prepared PR are listed in its validation package.

Other corrections: the output gate read DONE field 15 (offloaded experts) as
prompt tokens read; it now reads field 14. Engine cleanup no longer masks the
original startup error with a second BrokenPipeError from closing stdin.

## Real-model checks

Plain Q8 DFlash and official Q8 DFlash2 each pass eight cases of 32 output
IDs: **512/512 IDs exactly equal** to their fresh drafter-enabled and target-only
controls. Actual expert capacity is 2000 in all six arms. Reuse: live continuation 55, parked continuation 88,
checkpoint/rewind 16, sampled-to-greedy 72. Sampled requests offer zero drafts;
greedy drafting resumes afterward. No near-tie exemption is used.

The cases cover initial generation, live continuation, a different conversation,
RAM parking/restoration, pinned prompt restoration, rewind, sampling and a
subsequent greedy continuation. Full IDs, timings, counters, native INFO and
logs are in the per-run directories. Summary: summary.json.

Additional validation results are recorded in run-exit-codes.json,
the original external archive; they are historical checks, not fresh final-branch validation.

Commands (from repository root):

```sh
.venv/bin/python tools/dflash_cache.py --config bench/results/2026-10-19-dflash2-quant-mtp/target-config.json --engine build-dflash2/strata --output bench/results/2026-10-19-dflash2-pr/cache/official-fixed
.venv/bin/python tools/dflash_cache.py --config PATH_TO_PLAIN_DFLASH_CONFIG.json --engine build-dflash2/strata --output bench/results/2026-10-19-dflash2-pr/cache/plain-fixed
.venv/bin/python tools/verify_qfuse_commit.py --config bench/results/2026-10-19-dflash2-quant-mtp/target-config.json --engine build-dflash2/strata --output bench/results/2026-10-19-dflash2-pr/cache/qfuse-after
```

These results cover the supplied NVFP4 fixture with one proposal, split windows,
fixed placement and PCIe fraction 0. They do not establish equality for other
arithmetic paths or wider speculative windows. HIP/SYCL compilation, Windows,
multi-GPU and new performance measurements were not tested here. Sampled
requests retain the upstream target-only fallback and gain no draft acceleration.
