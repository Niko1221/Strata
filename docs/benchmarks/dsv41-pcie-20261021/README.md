# DeepSeek registered host memory: historical PCIe measurements

These measurements were collected on 2026-10-21 with a separate experimental
DeepSeek engine. They motivate the optional RAM-store allocation change in this
PR. **They are not before/after results for this PR, and do not measure inference
or the upstream store's mapped-memory GPU reads.**

## Measured result

An RTX PRO 6000 Blackwell Workstation (96 GB) on PCIe 5.0 x16 transferred the
same 156,662,231,040 bytes per pass from a 156,711,780,352-byte host arena:

| Arm | CUDA registrations | Actual CPU page backing | Wall-clock GB/s | Six timed passes, min–max |
| --- | ---: | --- | ---: | ---: |
| A | 1 | 4 KiB; 0% THP | 23.952108 | 23.903775–23.971078 |
| B | 40 | 4 KiB; 0% THP | 23.915125 | 23.906953–23.920579 |
| C | 1 | 2 MiB; 100% THP | 24.258977 | 24.242281–24.267592 |
| D | 40 | 2 MiB; 100% THP | 45.104124 | 44.904047–45.141330 |

The combined change D/A was **+88.31% (1.883x)**. Segmentation alone B/A was
-0.15%, and huge pages alone C/A was +1.28%. The large gain appeared with both
settings together on this machine. The small differences do not meet the
experiment's predeclared 5% threshold for a material gain.

Each arm has two independent processes, one complete warmup and three complete
timed passes per process. The headline is the mean of the two process medians,
in decimal GB/s. Every process transferred all 10,715 nonresident experts from
the same seven-shard DeepSeek-V4.1-Flash-1017 UD-IQ3_XXS set. The initial 4,645
resident IDs, source offsets, padding and order stayed fixed. A single CUDA
stream copied individual experts into an 8 x 16 MiB device ring, synchronizing
per layer. Wall time includes submission and those synchronizations; allocation,
file reads and host registration are excluded. CUDA-event time is separate in
the CSV. There was no profiler, model request, prefill or decode phase.

The 40 registrations cover balanced ranges of one contiguous mapping, with no
expert straddling a range. They are not one allocation per layer. The largest
range was 3,927,965,696 bytes. All four arms had identical layouts. Linux
`/proc/self/smaps` showed the stated CPU THP coverage after allocation, before
and after registration, and after measurement. This does not establish the
GPU/IOMMU mapping size or the driver's internal reason for the gain.

Hardware: Intel i7-14700KF, 192 GB DDR5-4800, Linux 6.8.0-138, NVIDIA driver
595.104.02, CUDA 13.2. All 258 samples inside timed passes reported an **active**
Gen5 x16 link; samples were about 0.5 seconds apart. This is sampled evidence,
not continuous link tracing.

## Evidence and limits

- [All 24 timed passes](measurements.csv), [allocation/registration coverage and
  provenance](provenance.json), and the [original measured program](historical_transfer.cpp)
  are preserved. `python3 docs/benchmarks/dsv41-pcie-20261021/recompute.py` checks
  file hashes and recomputes rates from bytes and time without a GPU.
- Run order was D1/C1/A1/B1/B2/A2/C2/D2. D1/C1 were retained from an interrupted
  stage using the same binary, followed by a service load/unload interval. Before
  A1, the controller stopped requiring absolutely zero system swap activity and
  instead recorded it. This was **not an uninterrupted ABBA run**. Every arm's
  two process medians drifted by less than 0.05% (predeclared limit: 3%).
- Other CUDA compute processes were absent at the before/after checks. Background
  swap-in and disk activity were nonzero; the largest sampled disk-read interval
  was 45.38 MB/s. Desktop and other system activity were not suppressed. These
  conditions limit precision and were not used to explain the gain.
- **The 24 passes did not validate transferred bytes.** An earlier, separate
  integrity-enabled attempt failed a byte comparison at layer 28/expert 310 after
  7,759 experts passed. Its root cause remains unresolved. Those timings are
  excluded here; the later bandwidth-only run did not resolve that failure.
  This table therefore reports throughput only, not correctness or model quality.
- The measured source was an uncommitted experimental snapshot, identified by
  source-manifest, loader-library and executable SHA-256 values in the provenance
  file. `historical_transfer.cpp` is archival source using that engine's loader,
  not a build target for today's `src/dsv41` engine. The model-loader snapshot,
  frozen hot-ID input and weights are not included: this bundle supports numeric
  recomputation, not turnkey hardware replay.

For reference, the original command shape was `transfer MODEL HOT_IDS.tsv ARM`
(`ARM` was A, B, C or D). The original build used `g++ -std=c++17 -O3 -Wall
-Wextra`, the experimental engine's include paths and
`libstrata_deepseek41_model.a`, CUDA 13.2 headers/runtime, `-lcudart -pthread`.
These path placeholders document the archived run; they are not a claimed
successful build command for this PR.

## What the current patch does

With `DS4_STORE_ZC=1 DS4_STORE_HUGE_PAGES=1`, Linux CUDA uses 2 MiB-aligned chunks,
requests THP and faults the whole committed chunk before the existing
`cudaHostRegister` call. Chunk padding counts against the RAM budget. Each expert
slot stays inside one registered range, at most 4 GiB minus 2 MiB. That bound is
an allocation policy motivated by the experiment, **not a CUDA API limit**.

The upstream store was already registered in chunks. This patch adds page
geometry and prefaulting to that existing path; it does not port the other
engine's scheduling or GPU-only execution. Registration count follows the RAM
budget, slot size and existing `DS4_STORE_CHUNK_SLOTS` setting (default 64).
`DS4_STORE_CHUNK_SLOTS=4096` requests larger chunks, capped by this bound; it does
not promise exactly 40 registrations or the historical bandwidth.

`MADV_HUGEPAGE` is advisory. Successful advice and a startup log do not prove
100% THP coverage. Inspect the store's actual mapping in `/proc/PID/smaps` before
comparing performance. No global THP, huge-page pool or driver setting changes.
If aligned reservation fails, the original allocation path is used; if page
advice fails, committed ordinary pages remain usable. Existing failed CUDA
registration and device-alias handling stay in place. Full-page faulting adds
startup work and padding can slightly reduce slot capacity.

Without the new variable, the old layout and lazy commit path are retained.
The engine enables the new variable only for the existing single-GPU,
single-shard-per-layer CUDA zero-copy path. Windows, HIP, Metal, SYCL and normal
CPU execution do not enable it. This is source-level scope, not a claim that all
backends have been built or tested.

## Validation of this patch

On macOS/Apple clang, the following passed with AddressSanitizer and UBSan:

```sh
c++ -std=c++17 -Wall -Wextra -Werror -fsanitize=address,undefined \
  -Isrc/dsv41 tests/dsv41/store_pages_test.cpp -o /tmp/ds4-store-pages-test
/tmp/ds4-store-pages-test
```

The portable checks cover budget accounting, tail padding, the registration
bound, invalid inputs and 10,000 deterministic packing cases. Linux additionally
enables a synthetic file-backed store test for exact matrix bytes, nonidentity
device aliases, registration failure, pinning, shrink/regrow and cleanup. That
Linux part **has not run**; its mapping callback is a CPU test double, not CUDA.
It is also registered as the `ds4_store_pages_test` CTest target.

Before this draft is ready for review, a Linux CUDA/PCIe 5.0 tester needs to build
`strata-dsv41`, run the Linux store test, check actual THP coverage and byte-exact
GPU reads, and compare the same upstream build with the option off/on/off.
Keep model/quantization, prompt hash, output length, resident expert set, cache,
thread count, chunk size and resources fixed. Preserve every run and failure.
The default-off model output must be compared with the base revision; a source
inspection does not establish byte-identical runtime output.

CUDA and HIP builds, GPU execution, full-model correctness, default-output parity
and a current-branch performance comparison are **not run**. No matching
prefill/decode timelines or Top10 activity tables exist for this patch; they
remain pending. Historical H2D rates cannot be converted into tokens/s.
