# RTX 3090 / IQ2_XS: official resident exchange rotation, same-build ABBA

Custom workload report under the community measurement guide. This is **not** a `strata bench run` standard-protocol submission and has no maintainer approval status. AI assisted the experiment orchestration and report preparation; the attached records are from actual hardware execution.

With rotation enabled, median fixed-cap code decode increased **118.5 → 137.1 tok/s (+15.7%)**, and Chinese decode **86.1 → 107.5 tok/s (+24.8%)**. Code ranges slightly overlap; Chinese ranges are separate. All four repair outputs stopped naturally and passed 14/14 checks. Their lengths were A: 499/311 and B: 352/1070 tokens: repair E2E median increased from 7.415 to 9.004 s despite higher median token throughput. Whole-request outputs vary within each arm, so these are workload-level measurements, not fixed-token-trace replay measurements.

## What was measured

Engine **0.1.55**, current main at [`c62d5e3a15b40ea51c96aae9e8d36928965f53ab`](https://github.com/Niko1221/Strata/commit/c62d5e3a15b40ea51c96aae9e8d36928965f53ab). A is `STRATA_EXCHANGE_ROTATE=0`, B is `=1`. One untouched source build, four fresh processes in **A → B → B → A** order. See [official rotation documentation](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/docs/EXCHANGE_ROTATION.md) and the existing [mixed-size implementation discussion #2042](https://github.com/Niko1221/Strata/pull/2042).

The source uses its stock **16 GPU staging slots**, not the earlier local 64-slot modification. No local kernel, early-fetch, fetch-grid, instrumentation, or bucket-rotation patches were applied. CPU fallback remains possible; it is recorded below.

## Hardware, build and settings

- Single RTX 3090, 24 GiB VRAM, 350 W power limit, Linux VM, 32 visible CPU threads; Intel(R) Xeon(R) Platinum 8260 CPU @ 2.40GHz; 62.76 GiB system-visible RAM. Host RAM speed and physical CPU topology are not measured.
- The GPU retains a previous hardware unlock experiment state. Selectors were read before/after and were unchanged. This is not a stock-versus-unlocked GPU benchmark; no new TFLOPS calibration was run.
- GCC 11.4 / CUDA 12.8, Release, `CMAKE_CUDA_ARCHITECTURES=86`. Pinned ggml `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Full [build commands](build.json), [build receipt](build-completed.json), [environment](environment.json).
- Binary SHA-256: `43b39750bcee469d2a35d31997d2484755be46cb721abf9ede4ff816c89fb2f8`.
- Original Flash-Next IQ2_XS local GGUF/native expert pack, matching PLE shard and MTP pack. Exact [model and pack SHA-256 values](model-hashes.json) are attached; publisher/tokenizer revisions are unknown.
- One active request; 32,768 context; int8 KV; prefill 2048; expert cache requested 18,432 MiB; PCIe fraction 1 and kernel mode; 7 CPU workers; spec 4 / spec-min-p 0.5; resident experts / RAM PLE; adaptive E4/S96/decay0.7/async0.
- Tail skip disabled; prefill CPU share disabled; prompt and conversation caching disabled. Greedy temperature 0, top_p 0.95, seed 20261013, reasoning effort none. These are custom settings, not model-card/default benchmark sampling.
- Effective fixed engine geometry: `{"expert_slots": 13503, "expert_cache_mib": 18530, "arena_mib": 16461, "spec": 6, "mtp_max": 4, "pool_workers": 7, "pcie_frac": "1.00", "context": 32768, "kv": "int8"}`. Every arm began from the same attached [frozen expert profile](frozen-expert-profile.bin).
- Resource sampling every 2 s was identical in both arms. No Nsight, GPU event profiler, logits export, or shadow kernel ran during timed requests. Loading and file hashing precede the timed requests.

## Workloads and timing

The attached [cases.json](cases.json) preserves all prompts. Every process first warms the code and Chinese cases once at a 512-token cap. Then each case runs twice at the same cap. A retry-function repair runs once per process with a 2048-token cap and natural stopping. Total: **28 requests = 8 warm-up + 16 fixed-cap performance + 4 repair**. There are no invisible retries.

- `decode-0`: sustained Python source generation; 512-token output cap.
- `decode-2`: Chinese technical explanation; 512-token output cap.
- `repair_a`: fix `retry_delays(attempts, base, cap)`, including bool/non-integer rejection, negative arguments, zero cases, cap handling, and 10,000 attempts without overflow. The [14-check grader](grade_code.py) is fixed and runs generated code in a separate CPU/address-space-limited process. This one task is not a general coding benchmark.

Decode throughput is `1000 * timings.predicted_n / timings.predicted_ms`; TPOT is the inverse in ms/token. Timings come from the engine's SSE `timings`, with raw usage retained. Client `perf_counter` measures time to first nonempty content (TTFT, reasoning disabled) and complete streaming E2E. SSE chunks are never counted as tokens. Engine prompt throughput is also included in results.json. Startup and prefill are excluded from decode throughput. MTP is enabled, so TPOT is per accepted output token across the active decode pipeline.

Statistics below are **per-request medians [min,max]**. Fixed-cap cases have 4 formal requests per arm; repair has 2 per arm. Warm-up requests are excluded from the speed summaries. Per-process results and aggregate token/time rates remain in [results.json](results.json) and [requests.csv](requests.csv).

| Case | Metric | A rotation off: median [min,max] | B rotation on: median [min,max] | Median B/A change |
|---|---|---:|---:|---:|
| decode-0 | decode_tps (tok/s) | 118.540 [105.332, 132.197], n=4 | 137.112 [131.687, 142.436], n=4 | +15.67% |
| decode-0 | tpot_ms (ms/token) | 8.479 [7.564, 9.494], n=4 | 7.300 [7.021, 7.594], n=4 | -13.91% |
| decode-0 | ttft_s (s) | 0.320 [0.306, 0.350], n=4 | 0.314 [0.306, 0.321], n=4 | -1.82% |
| decode-0 | e2e_s (s) | 4.659 [4.166, 5.174], n=4 | 4.031 [3.901, 4.196], n=4 | -13.47% |
| decode-2 | decode_tps (tok/s) | 86.113 [78.349, 88.881], n=4 | 107.488 [102.259, 111.639], n=4 | +24.82% |
| decode-2 | tpot_ms (ms/token) | 11.624 [11.251, 12.763], n=4 | 9.317 [8.957, 9.779], n=4 | -19.84% |
| decode-2 | ttft_s (s) | 0.313 [0.297, 0.347], n=4 | 0.316 [0.308, 0.327], n=4 | +0.75% |
| decode-2 | e2e_s (s) | 6.269 [6.053, 6.833], n=4 | 5.075 [4.883, 5.323], n=4 | -19.06% |
| repair_a | decode_tps (tok/s) | 61.196 [50.618, 71.774], n=2 | 82.291 [69.841, 94.740], n=2 | +34.47% |
| repair_a | tpot_ms (ms/token) | 16.844 [13.933, 19.756], n=2 | 12.437 [10.555, 14.318], n=2 | -26.17% |
| repair_a | ttft_s (s) | 0.885 [0.879, 0.890], n=2 | 0.853 [0.853, 0.854], n=2 | -3.54% |
| repair_a | e2e_s (s) | 7.415 [7.005, 7.825], n=2 | 9.004 [5.877, 12.131], n=2 | +21.43% |

## Full repair outcomes

| Process | Output tokens | Finish | Checks | Decode tok/s | E2E s | Final text SHA-256 |
|---|---:|---|---:|---:|---:|---|
| 0-A | 499 | stop | 14/14 | 71.77 | 7.825 | `54d39fa72330772bc15b0f9e8aeecbe0d6085669a99434723fe438bf65a2c964` |
| 1-B | 352 | stop | 14/14 | 69.84 | 5.877 | `0d46347459028a27616659c09c79fe563c9d8618f81f2a0c0b32a6c23b6100e9` |
| 2-B | 1070 | stop | 14/14 | 94.74 | 12.131 | `1c7b7c87c29822810b1b48d72272c60dc8c756e7da525d467eae25e8b5bcd59d` |
| 3-A | 311 | stop | 14/14 | 50.62 | 7.005 | `515c9c1e5dcfe7c7dfd2f07e4928114900215a9818db70572e57b2c8556014cd` |

Formal final-text hash equality by task: `{"decode-0": false, "decode-2": false, "repair_a": false}`. This checks final UTF-8 text and output lengths; native token IDs and per-window logits were not exported. Differences, if present, are retained in the outputs and are not scored away. E2E for unequal output lengths measures completion of the generated answer, not equal work.

## Path activation and resource tradeoff

The B startup logs must contain `exchange buffer rotation enabled`; A must not. Each B request retains the official cumulative rotation count and avoided host-copy payload. The report does not count read-plus-write traffic twice and does not attribute GPU transfers to avoided RAM copies.

| Process | Resident exchanges, cumulative | Rotated blocks, cumulative | Host memcpy payload avoided, bytes | Non-hit CPU-path routed entries across requests |
|---|---:|---:|---:|---:|
| 0-A | 13442 | 0 | 0 | 543 |
| 1-B | 11387 | 11387 | 16369740800 | 350 |
| 2-B | 12962 | 12962 | 18629324800 | 212 |
| 3-A | 12093 | 0 | 0 | 767 |

CPU-path entries are derived from the official per-request cache log as `lookups - hits` under this single-GPU, tail-skip-off configuration; PCIe entries are reported separately by the engine. This is a decode route count, not sampled CPU utilization or prefill CPU work. Per-request values are attached. The stock16-slot build is allowed to use CPU fallback; a PCIe fraction of1 is not treated as proof that fallback was zero.

The pack has three expert block sizes: **1,177,600 / 1,305,600 / 1,510,400 bytes**. With 96 swaps and async 0, the off path allocates `96 * max(sizes)` = **138.281 MiB**, and rotation allocates `96 * sum(sizes)` = **365.625 MiB**, an extra **227.344 MiB pinned host RAM**. This is allocator geometry checked against the pack metadata hash, not an RSS estimate; [expert-block-sizes.json](expert-block-sizes.json) records the byte counts. Both arms use the official allocator; no enlarged equal-RAM pool was injected into the off arm. The enabled-pool startup message, loaded process status and system Mlocked values are retained. Linux VmLck/RSS alone do not account for all CUDA-pinned allocations. GPU cache geometry matched across all starts.

All requests reported zero expert file-tier blob reads. The repair requests nevertheless reported 20.1/21.9/44.5/47.9 MB of total process storage reads and 6/7/6/6 major faults. These are distinct engine/OS counters; this run does not attribute those residual reads to a particular model component or quantify their critical-path cost.

The general formula, including async bounce buffers and the scope of the memory estimate, is documented in [memory-formula.md](memory-formula.md).

## Validation, failures and limits

- Official ownership CTest and explicit `file_expert_source_test --rotation-gpu`: [commands and return codes](tests.json), [CTest output](ctest.log), [GPU fixture output](rotation-gpu.log). Hardware execution is distinct from compilation.
- Streams:28/28 complete. Payload-cache reuse: none. Same binary/profile and actual engine geometry across all four starts: verified.
- Source files: unchanged before/after. Original service files and selectors: unchanged; the original service was restored loaded and idle. New NVRM Xid lines during this experiment: **0**. An old historical Xid line was already present before the run.
- Initial remote `git clone` stalled and was stopped before timed work. The tested source is the immutable GitHub commit archive. A first admission attempt loaded the model but the observer only searched the server main thread's child list; the new server created its engine from another thread, so observer PID lookup failed before any inference request. That attempt's [error and logs](failed-attempt-1/) are preserved. The observer was fixed to search all server threads, the original service was restored, and the full ABBA began in a new directory. No inference request was retried or replaced.
- Full weight/pack hashing completed during that zero-request preparation attempt. The successful attempt reused those hashes after checking every file's size and modification/change timestamps, and checked the same metadata again before every arm; see model-hashes-reuse.json. The build logs retain compiler warnings and exact commands. The experiment did not modify the model files.
- Four starts, one GPU/VM, one quantization, one fixed repair task; no long-context, concurrency, broader quality, stock hardware, Windows, HIP, or SYCL performance claim. Compute Sanitizer and native token-ID/logit parity are not run in this report.
- The new engine warns that 7 workers exceed the 4 cores of the main thread's exposed CPU socket and recommends 3. It also reports 295 MiB free VRAM at request start and recommends more reserve. Both settings were held fixed to preserve the planned comparison. Actual GPU paging was not measured. This is not a claim that the configuration is the best tuned 3090 deployment.
- Results include runtime adaptive expert placement, MTP/lookup acceptance and CPU/GPU numerical paths. Source-level unchanged arithmetic alone does not prove identical whole-request outputs. Every saved output and quality outcome is retained.

A standalone [HTML explainer](rotation-explainer.html) illustrates buffer ownership, the memory formula and these measured results. Download and open it locally. Its animation is schematic, and slider settings other than 96 are formula estimates. Interaction/data checks used DOM stubs; browser screenshot verification was not completed.

## Evidence and reproduction

`abba/<index>-<arm>/` includes sanitized configuration, fingerprint, effective environment, schedule, full SSE request records, output text, engine/server logs, resource samples and repair grades. Local machine paths are replaced by `<SOURCE>`, `<MODEL_ROOT>`, `<BENCH_ROOT>` and `<EXISTING_INSTALL>`; the machine IP and GPU UUID are omitted. Replace those path placeholders with your own locations before launching.

Rebuild with the exact source/ggml commits and flags in build.json. Run each saved config from its `cwd`, using the source server and that config's binary. Apply its fingerprint environment. Restore the initial profile before each process; wait for `/health` loaded=true. Submit requests in its schedule via [replay_api.py](replay_api.py), with the attached seed and reasoning settings. Retain each fresh process and warm-up separately. Do not reuse a process between arms or let prompt/conversation caching silently change the work. [machine_orchestration.py](machine_orchestration.py) preserves the actual orchestration used here, with machine paths replaced; its original-service management and hardware audit helpers need adaptation before running elsewhere.

The included [analyze.py](analyze.py) is the actual summary code used for this run against the private raw archive. That archive additionally retains before/after service and hardware audits; those private host snapshots are not published. To verify the self-contained public package, run `python3 verify_public.py .` from this directory. It checks artifact hashes, raw timing/output evidence, score counts and the summary formulas. [SHA256SUMS](SHA256SUMS) hashes the final sanitized submission bytes.
