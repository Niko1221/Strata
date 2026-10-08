# Fresh c4 throughput against v0.1.40.4

Measured 8 October 2026 on RTX 5090, Ryzen 9 9950X3D and 48 GB DDR5-6000,
Windows, NVIDIA driver 610.88, reported power limit 600 W. The owner previously
reported removing the undervolt; the voltage curve was not independently
verified. No clocks or power settings were changed for this comparison.

Three paired runs per workload, seeds 123/456/789, order AB/BA/AB. Each arm
starts a fresh process and excludes its 256-token warm-up and startup. These
are medians of the three runs, with ratios calculated from those medians.

| Workload | Measurement | Upstream TPS | Candidate TPS | Ratio |
| --- | --- | ---: | ---: | ---: |
| Essay writing | Four-stream decode | 154.8 | 203.1 | 1.31x |
| Essay writing | Whole bounded run | 149.6 | 190.5 | 1.27x |
| Tool-driven coding | Four-stream decode | 162.5 | 336.0 | 2.07x |
| Tool-driven coding | Whole bounded run | 136.3 | 233.5 | 1.71x |

Baseline source is pristine upstream v0.1.40.4, `6674a0065f`, with
`--batch-mtp` enabled. Unlike PR #1209's older reference, no baseline patch is
needed. Candidate runtime source is `ae084bc`; later evidence commits do not
change the runtime. Both are Release builds using CUDA 13, MSVC 14.32,
sm120, the same CMake cache and GGML source
`3cf03257f219afbe7334045ff7c6a06ac68c627d`.

Swift 1.5 IQ2_XS, c4, 98,304 allocated context per request, int8 KV, 2,560 MiB
requested reserve, requested cache 11,022. Profile sizing rounded both builds
to **11,568 resident experts in every run**. There were no expert-cache retry
cuts or reserve shrinks. Whole-arena host pinning used the known successful
sliced fallback; this is distinct from GPU expert-cache allocation failure.
This is not a 96K-prompt benchmark: the report records actual prompt lengths.

Temperature .85, top-p .95, top-k 20, thinking disabled by the fixture. Normal
optimized CPU kernels are enabled. QFUSE is off, as in the original NVIDIA
qualification. Candidate settings: `--batch-spec 4 --batch-rows 16
--batch-spec-fixed --batch-draft-overlap --batch-suffix --batch-expert-merge
--batch-adapt --batch-pcie-frac 0` and `STRATA_BATCH_DENSE_PARALLEL=1`.
The speedup belongs to this combined opt-in configuration; it is not an
individual attribution to every switch. Upstream's current defaults remain
in place for its other optimisations, including interleaved MMVQ.

## Latency and output limitations

| Median of per-run metrics | Essay upstream | Essay candidate | Coding upstream | Coding candidate |
| --- | ---: | ---: | ---: | ---: |
| Native token-gap p95 | 47 ms | 47 ms | 62 ms | 47 ms |
| Turn TTFT p95 | 0.313 s | 0.344 s | 0.891 s | 0.813 s |

Gaps pool adjacent native receipt timestamps within turns, including zero-gap
speculative bursts. They are not HTTP/UI latency or worst-stream guarantees.
Four-stream decode counts emitted tokens in intervals with all four slots
active and no admission. Whole-run throughput includes prompt reads, tools,
and periods with fewer active requests.

The coding fixture is an in-memory read/write/check tool loop, not DSH or
SWE-bench. **12/12 modules passed syntax/structure checks in each arm**.
Nine agents in each arm reached the six-turn cap; only three in each finished
naturally. Generated code and unit tests are not executed. Outputs and work
amounts differ: these numbers establish throughput, not equal-quality task
completion speed, normal-mode token identity or universal quality equivalence.

## Reproduction and evidence

The fixture in `tools/benchmark_pr_workloads.py` is unchanged from PR #1209.
`tools/benchmark_v0404.py` runs the bounded sequence; configs and all twelve
compact summaries are under [evidence/v0404-performance](evidence/v0404-performance).
Adjust the model, tokenizer, executable and dependency paths to the local
installation and place the configs under `build-validation/baseline.json`
and `candidate.json`. Build the recorded source commits before running the
campaign; do not compile or run other inference concurrently.

`tools/summarize_v0404.py` records actual residency, timing and artifact hashes.
Raw reports, generated text, engine logs and binaries remain locally under
`build-validation`; the public manifest identifies their exact bytes.
For correctness controls and exclusions, see [V0404_REBASE.md](V0404_REBASE.md).
