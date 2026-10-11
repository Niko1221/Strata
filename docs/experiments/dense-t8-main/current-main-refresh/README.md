# Current-main refresh, 2026-10-11

The eight-column Q8 QKV path remains experimental and default off. Refreshed to
upstream main `61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`; tested native source
`b57144b620daddf912255d5b692d6c3c6a3c3f48`. The current main QKV fork and its
`mm()` fallback are preserved. No other performance experiment is included.

CUDA, HIP and standalone SYCL engine builds passed on the identical source
archives. CUDA SM120 tested the feature; HIP/SYCL build results establish fallback
compatibility, not execution of this CUDA-only kernel. CUDA synthetic scalar
bounds, unsupported formats/shapes, zero row, graph input mutation, independent
owners/streams and early close passed. Compute Sanitizer memcheck reported zero
errors. Raw records include exact source/archive and engine hashes.

The isolated warmed graph stage, including quantization and repacking, measured
20.543751 µs off versus 14.330000 µs on, with 32 warmups and 256 timed replays
per arm. This is a single ordered observation using repeatedly reused weights;
it does not establish a full-request or cold-memory gain.

Eight model cases completed at 8,192 prompt + 512 generated tokens, with FP16 KV,
8,192-token prefill chunks, fixed cache 15,472, mmap PLE and adaptation disabled.
Approximate route-tail skipping was explicitly disabled in every arm. All eight
pure-arithmetic function checks passed; the long generated module was not scored.

| Comparison | Full benchmark token IDs | Benchmark eight-column MMQ windows |
|---|---|---:|
| Main / candidate default off | All 512 match | 0 / 0 |
| Same binary, T8 off / on | First difference at index 16 | 0 / 57 |
| Same binary, T1 off / on | All 512 match | 0 / 0 |
| Same binary, T4 off / on | All 512 match | 0 / 0 |

The T8 model observations were 70.98 off versus 68.28 on committed output tok/s.
Output and draft work changed, so this is not an isolated acceleration estimate
or a quality-equivalence result. The earlier observed apparent request speedup
did not reproduce in this single refresh pair. Keep the kernel default off.

**An MTP cap is not a global feature-off control.** The short 44-token arithmetic
prompt goes through verifier prompt reads with `--spec 8`. Five eight-column
launches occur before the benchmark even in the T1/T4 feature-on arms, and their
128-token function outputs first differ from off at index 26. The current
`read_windows` path chunks prompt reads using `S`, independently of the MTP cap.
The benchmark-window counts above subtract this preceding request. Thus the
long-prompt fallback matches do not establish unchanged behavior for every
short prompt or for the entire preceding request history.

The arithmetic reduction order changes while Q8 activation codes and promoted
half scales are preserved. The synthetic tolerances and model observations do
not prove broad quality equivalence or identify every downstream state effect.

[All timings](measurements.md), [full-output comparisons](current-audit.json),
[source provenance](source-manifest.json) and individual result/log/memory files
are retained. Startup took about 69 seconds per arm and is excluded from decode
and prompt-plus-decode rates; file-cache and clock conditions can vary across
these single observations. Setup/engineering time was not instrumented.

Reproduce using the parent `reproduce/case.py` and `memory_samples.py`, with this
directory's `plan.json` and local paths substituted. Run cases serially on an
idle SM120 GPU. Build with `STRATA_BUILD_TESTS=ON`; run `dense_q8_t8_test`,
Compute Sanitizer memcheck and `dense_q8_t8_bench` before the model suite.
