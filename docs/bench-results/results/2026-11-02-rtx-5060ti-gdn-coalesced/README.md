# RTX 5060 Ti GDN coalesced experiment, 2026-11-02

The opt-in candidate improves the separate native recurrence in these tests.
It does NOT beat the existing fused recurrence-plus-output-norm path.
Do not turn off fusion for ordinary use based on these results.

Full rationale and exact scope:
../../../_docs/2026-11-02_gdn-coalesced-experiment_Codex.md

full-r2.jsonl and full-r3.jsonl contain all final samples, correctness gates and
device properties. summary.json retains medians, min/max, paired ratios and
deterministic bootstrap intervals. gpu-*.csv records concurrent GPU load; this
was not an exclusive-GPU benchmark and is not a tokens-per-second measurement.
The one-layer case has 3 MiB state; 48 synthetic state buffers have 144 MiB.
The compiled model geometry has 36 GDN layers, not 48.

memcheck.log, racecheck.log and synccheck.log show zero sanitizer errors in the
quick workload. environment-*.jsonl verifies process-local configuration.
receipt.json records final source hashes and final executable identity. The
performance runs preceded whitespace-only clang-format of the harness; the
final rebuild then passed all three environment smoke cases.

Build/run on the measured Windows/sm_120 target:
powershell -File tools/benchmark_native_gdn_coalesced.ps1

The wrapper applies fast math to native adapters only, matching the source
CMake definitions. build.log is the final successful compiler output.
initial-all-fastmath-r1.jsonl is exploratory data and is excluded from summary.

No model weights or private user data are used. No installed engine, service,
GPU setting, dependency or default was changed. Complete engine integration,
real-model output/quality/speed and HIP/SYCL/Metal qualification are not done.
