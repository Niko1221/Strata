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
The full-model comparison is running at this snapshot.

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
claimed. Full-model token and speed comparisons are pending in this snapshot.

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
