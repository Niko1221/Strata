# Follow-up audit, 2026-11-02

See ../../../../_docs/2026-11-02_gdn-followup-audit_Codex.md.

full-r4.jsonl and full-r5.jsonl are completed microbenchmarks using the identical
binary as the previous final receipt. All four correctness cases pass, and the
existing fused kernel remains fastest. summary.json includes all 15 samples per
configuration in each run; GPU telemetry is sampled concurrently, not exclusive
GPU ownership. These timings are not native-model throughput.

math.json contains constructed SO(8)/quantization counterexamples and the
conditional 6-MiB / 448-GB/s calculation. profiler-access.json is a clearly marked
transcription of tool observations. Nsight hardware counters were denied; the
CUDA-only Nsight Systems smoke timed out. No privileges were changed.

baseline-r1 and baseline-r2 contain ABORTED native-model diagnostics. Both
stopped at the predeclared physical-RAM floor, with no completed DONE request.
Job cleanup confirmed zero active processes. r1 and r2 use different settings;
they are not an A/B performance pair. No instrumented native run was launched.

analyzer-tests.log is a synthetic reporting-logic test run, not a GPU benchmark.
The five cases check refusal of invalid evidence and honest parity/count labels.

The follow-up receipt records current source/artifact hashes. The earlier parent
receipt remains the immutable record of the original experiment. No release,
installation, service or runtime-default change was made.