# Dual PCIe V100: current-main compatibility and B6-only comparison

Community diagnostic, not the standard benchmark protocol. Measured on 2026-10-31.
This follows up on [#3411](https://github.com/Niko1221/Strata/pull/3411) and the B6-only
subset in [#4767](https://github.com/Niko1221/Strata/pull/4767).

## Code and hardware

- Tested merge commit `cb656b83fe152d404a8466f22cc7c2334b6c85e5`, based on upstream
  `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`. Later report-only commits do not change tested code.
- One compiled binary for all arms: SHA-256
  `0068c2d3f1bbadfee78e22cb514107c35ff176a0fd327a22211ef35932d124cb`.
- Ryzen 9 9950X, 128 GB RAM, two Tesla V100 PCIe 32 GB, each capped at 150 W.
  Ubuntu 24.04.5, GCC 13.3, CUDA 12.8.93, sm_70.
- Qwen3.8-Flash-Next GSQ-RCO IQ3_S, MTP4/min-p 0.5, INT8 KV,
  32,768 context, auto layer split, two slots/two batch groups,
  later-GPU reserve 650 MiB. Same existing pack and expert profile for all arms.
- `STRATA_PF_WMMA=0 STRATA_PF_WMMA2=0` in all arms. FP16 remains out of this comparison.

## Protocol and scope

Three cycles of baseline, B6-only, full table; fresh process before every arm.
Before loading, both cores had to be at most 50 C and HBM at most 52 C.
There was one short warmup, then the same four synthetic prompts per arm:
Japanese/code (256 output tokens), 1,551-token summary and 15,352-token prefill
(each 64 output tokens). Temperature 0, seed 42, thinking off; all measured
requests reported zero cached tokens. Individual data includes engine timings,
TTFT, usage and decoded-text SHA-256. This is a fixed-order screening, not a
randomized or broad quality evaluation. Short-prompt prefill rates are not used
as evidence of a prefill improvement.

- Baseline: `STRATA_SM70_PERF=0`, all 13 manual variables unset.
- B6-only: same baseline, with `STRATA_B6_MMVQ_ROWS=1`. This isolates exactly the
  setting proposed by #4767 on the same current tree; it is not a separate build
  of that PR or a reproduction of its kernel microbenchmark.
- Full: `STRATA_SM70_PERF=1`, all 13 manual variables unset; startup confirmed all
  13 defaults applied automatically. The full table includes B6 once.

The remaining flags in the full arm are SIDE_MIXER=2, GR_UP75=1, GU_SWIGLU=1,
QSA_ROWS=1, ATTN_DAILY=1, SM70_TABLE=1, GR_NORM_FAST=1, ARGMAX_ROWS_SM70=1,
QIL=1, EXPDEC=1, GR_DOWN_TG=1 and DEQUANT_WIDE=1 (all prefixed STRATA_).

## Results

Median of three runs, tok/s:

| Arm | Japanese decode | Code decode | 15,352-token prefill |
|---|---:|---:|---:|
| baseline | 73.7 | 108.4 | 1504.6 |
| b6 | 75.9 | 113.1 | 1507.8 |
| full | 81.6 | 122.2 | 1517.9 |

b6: Japanese +2.99%, code +4.34% versus baseline; all 12/12 corresponding decoded texts matched baseline.

full: Japanese +10.72%, code +12.73% versus baseline; all 12/12 corresponding decoded texts matched baseline.

This supports the direction of the B6-only result on a different V100 topology.
On these dual PCIe cards the combined table adds a further benefit; it does not
show that every additional flag contributes independently. In particular this
run does not isolate GR_NORM_FAST and does not contradict the SXM2 report's
failure to resolve its individual benefit. Decoded-text equality is not a
bitwise kernel, token-ID or exact_pp/logit identity gate.

## Long-input concurrency

After the last full-table arm, four requests in two concurrent pairs each used
32,000 input + 512 output tokens. All returned both start/end sentinels.

Pair 1: request completion times 43.081 / 43.106 seconds.

Pair 2: request completion times 43.174 / 43.199 seconds.

Sentinels test limited recall/liveness, not general conversation quality.

## Compatibility and validation

Passed:
- Built the current source `strata` and `nv_defaults_test` targets on CUDA 12.8/sm_70.
- The combined host test passed on Linux/GCC and macOS/Clang. It retains the GB10
  four-switch/STRATA_DEFAULTS=026 checks and adds Volta opt-out, user-override and
  idempotence checks. Ampere/Pascal tables are unchanged.
- Resolved the startup-message conflict while preserving GB10 labels and the
  removal of its exactness claim. Volta now says speed switches, not exact speed
  switches, since our validation is decoded-text equality only.
- Removed the duplicate CMake nv_defaults_test target introduced by merging the
  independently added tests. This leaves one registered test.
- All 36 measured requests, plus four concurrent requests, completed. Temperature
  and fan guard remained active. No OOM or thermal cutoff occurred.
- Normal production Strata and the recovery timer restored after testing; LINE
  was on its cloud backend during GPU testing.

Not run:
- HIP/SYCL builds or hardware; no such toolchains available. NVIDIA dispatch is
  guarded out of HIP/Metal, and no non-NVIDIA kernels were changed.
- Single V100, SXM2, 16 GB, other packs, or kernel-level B6 bitwise comparison.
- Per-flag ablation of the other 12 settings, broad quality, or prolonged endurance.

There is still no maintainer scope agreement for a broad sm_70 default. The B6-only
proposal is a strict subset; either it can land first and the remainder be judged
separately, or its independent evidence can support the B6 row here. This report
supports the measured dual-PCIe combination without claiming it is optimal on
all Volta systems.

## Reproduction

Build:

```sh
cmake -S . -B build-v100 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DSTRATA_EXPERIMENTAL_SM60=ON -DSTRATA_MMQ_TAB=OFF \
  -DSTRATA_MMQ_KQUANTS=OFF -DSTRATA_BUILD_TESTS=ON
cmake --build build-v100 --target strata nv_defaults_test -j 12
./build-v100/nv_defaults_test
```

Choose the CUDA 12.8 nvcc path with CMAKE_CUDA_COMPILER. Start the server with the
configuration and arm variables above, using the same binary each time. Run the
included measure.py with STRATA_API_KEY set privately and BENCH_OUTPUT_DIR set:
`python3 measure.py LABEL 8080 /path/to/engine.log`. Repeat each arm three times
with a fresh process, then run `python3 stability.py` with the same environment for the long-input pair check. Credentials
and production configuration files are not part of this report.

Long-input telemetry: maximum core temperatures [50.0, 53.0] C, maximum HBM [52.0, 54.0] C; minimum free VRAM [4347.0, 205.0] MiB. GPU1 headroom remains narrow.
