# DFlash2 attention query cache, 2026-10-26

This follows the [previous DFlash2 study](../2026-10-25-dflash2-follow-up/README.md).
The new `STRATA_DFLASH2_CACHE_QUERY=1` path keeps each lane's four query values
in registers while its attention tile scans the keys. The dot products, warp
reductions, softmax and value sums retain their previous order. It works with the
existing small-page specialization and general page addressing. It is off by
default; the five original SM89 attention kernels retain identical instruction
bytes (94,848 bytes), as recorded in `attention-default-text.json`.

Hardware: RTX 4070 Ti SUPER 16 GiB / SM89, driver 610.57.04, Ryzen 9 5900X,
125 GiB RAM, Linux, GCC 14.2 and CUDA 13.4.92. Desktop applications remain open;
clocks are not locked. These are local experiments, not standard protocol results.
The checkout starts at `3f08f4c2`; unrelated CPU/zram changes in another checkout
are excluded. Full binary/source hashes are in `metadata.json`.

## Component measurements

Final source, existing small-page specialization enabled, page size 4, eight rows,
32 query heads and eight KV heads, 2048-cell attention window:

| End position | Before | Query cache | Lower latency |
| --- | ---: | ---: | ---: |
| 32 | 8.674 µs | 8.569 µs | 1.2% |
| 128 | 16.965 µs | 14.674 µs | 13.5% |
| 512 | 42.693 µs | 37.794 µs | 11.5% |
| 1024 | 80.197 µs | 68.908 µs | 14.1% |
| 2053 | 156.688 µs | 134.147 µs | 14.4% |

Two processes per variant run in ABBA order. Each process checks the FP64 oracle,
then measures all 30 cases: pages 4/64, rows 2/4/8 and five end positions. A graph
contains one complete attention call, with 20 warmup replays and 200 timed replays.
The table uses the median of the two process values. Every individual sample is
in `query-final-summary.json` and its logs. The earlier duplicate-kernel prototype
also checks the general-page path with the small-page option disabled; those
samples are retained separately. They are not pooled into the final table.

These are hot synthetic component timings, not request throughput. No percentage
is added to an earlier component gain or presented as a larger advantage over MTP.
The completed full-model decode screen below does not establish an increased
advantage over MTP. A separate 32K prompt comparison is running at this snapshot.

## Full-model decode screen

The fixed prompts contain 27/2427/2425 input tokens. Every process has one warmup
and three measured 384-token requests per task. DFlash2 processes also perform
same-process target-only checks; every warmup, measured sequence and control
matches a common separate target-only process. All nine profiles, including that
control, completed. The table shows decode speed, excluding prompt processing.
Raw files also retain prompt time and request wall time.

Settings: context 8192, requested resident 8192, expert cache 2900 slots, reserve 480
MiB, prefill 512, short-read 64, prompt cache disabled, speculation allocation 8,
fixed routing profile, no adaptive swaps or PCIe tier. DFlash2 uses the official
Q8_0 artifact, block 7 and floor 0.7. MTP uses max-T 4 and floor 0.5. The target is
`Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata`, with the same pack, dense
GGUF, BF16 embeddings, FP8 PLE and interleaved shared-head settings for every arm.

Median decode tok/s in execution order:

| Arm | Short code | Long code | Cache |
| --- | ---: | ---: | ---: |
| mtp-a | 39.444 | 39.180 | 41.726 |
| original-a | 44.602 | 42.926 | 45.194 |
| tuned-a | 44.981 | 43.289 | 45.434 |
| query-a | 44.958 | 43.254 | 45.320 |
| query-b | 44.663 | 43.300 | 45.438 |
| tuned-b | 44.682 | 43.239 | 45.363 |
| original-b | 44.698 | 43.007 | 45.097 |
| mtp-b | 39.455 | 39.087 | 41.755 |

`original` uses the starting branch's behavior without the optional bundle.
`tuned` enables prompt batch/grid, direct quantization, context graph, BF16 tiles,
register top-k, list merge, vector selector and the existing small-page attention.
`query` adds only the new query cache to that bundle. Original/tuned use the
baseline binary; query uses the candidate whose full hash is in the metadata.

The query cache reduces isolated attention latency but does not produce a stable
increase in full-model decode speed over the tuned profile here. The bundle's
small differences from original also need to be considered with process variation.
The existing DFlash2 advantage over these fixed MTP settings is visible, but it
must not be attributed to the new cache. Individual samples and ranges are in
`whole-model/summary.json`; rounds are not independent hardware trials.
All DFlash2 processes retain accepted/offered counts 279/368, 246/322 and 266/327.

`collect_full_screen.py` archives only zero-exit completed profiles and checks
all token sequences, prompt/profile hashes and actual placement metadata against
the common control. Original JSON hashes are recorded before absolute local
paths become `${REPO}` / `${STRATA_DATA}`. Full commands and environment flags
are retained. The source runner is the existing
`bench/results/2026-10-21-dflash2-long-goal/run.py`.

## Validation

Passed:

- Fresh CUDA builds with DFlash2 enabled and disabled.
- Final independent FP64 attention oracle for pages 4/64/96, heads 4/32,
  rows 1/2/4/8, full/local windows, distant positions and ring aliases.
- Invalid-value checks, including a visible NaN query.
- Every dumped output is bit-identical between variants for all 30 timing cases:
  2,293,760 bytes per process. Eight prototype processes and four final-source
  processes completed. Hashes and per-case fingerprints are retained.
- Query cache with the small-page option: Memcheck zero errors; Racecheck zero
  errors and warnings. The sanitizer fixture used rows 2/4/8; rows 1 was added to
  the later ordinary oracle without changing the device algorithm.
- Static comparison of all five original attention kernels: identical SM89
  instruction bytes. This is not a claim that the entire executable is identical.

Not run: other GPUs, Windows, Metal, multi-GPU, concurrent requests or vision.
DFlash2 in this tree requires the NVIDIA CUDA backend; HIP/SYCL execution is not
claimed. The short full-model token and speed screen completed as described
above; the separate 32K comparison is pending in this snapshot.

Reproduce the component check in an enabled CUDA build:

```sh
cmake --build build-opt --target dflash2_extension_parity -j4
STRATA_DFLASH2_SMALL_PAGE_ATTENTION=1 STRATA_DFLASH2_CACHE_QUERY=0 \
  build-opt/dflash2_extension_parity --attention-bench /tmp/attention-before.bin
STRATA_DFLASH2_SMALL_PAGE_ATTENTION=1 STRATA_DFLASH2_CACHE_QUERY=1 \
  build-opt/dflash2_extension_parity --attention-bench /tmp/attention-after.bin
cmp /tmp/attention-before.bin /tmp/attention-after.bin
```

`compare_attention_text.py BASELINE CANDIDATE OUTPUT.json` reproduces the static
SM89 comparison using CUDA cuobjdump and c++filt. `prompts.json` is the exact input
file for the full-request screen; its hash is retained in the metadata.
