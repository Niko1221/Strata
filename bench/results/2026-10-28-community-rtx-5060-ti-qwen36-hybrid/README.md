# Qwen3.6 hybrid tradeoffs on Windows RTX 5060 Ti

Measured on 28–29 October 2026 by [Thymos2027](https://github.com/Thymos2027). Independent validation of rynwang's [#4335](https://github.com/Niko1221/Strata/pull/4335) and [#4347](https://github.com/Niko1221/Strata/pull/4347), tested separately. No engine changes are proposed here.

**At a configured 32K context, chunk4096 reduced the 16K-input fixed-history request cost, but increased the 1K-input cost. Releasing packed copies improved the 16K generation component, while its prefill cost increased enough to make the combined request slower.** These are native, teacher-forced comparisons with frozen expert residency, not standard benchmark results, API/free-generation speed, or general quality equivalence.

## Hardware, software and provenance

| Item | Measured setup |
| --- | --- |
| GPU | One RTX 5060 Ti16GB, sm120 |
| CPU / RAM | i7-12700,128GB DDR4 |
| OS / driver | Native Windows11, driver610.88 |
| Build | MSVC19.42, CUDA13.2, Ninja Release, CUDA architecture120 |
| Source baseline | `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d` |
| #4335 head | `bdf0bce2e5520680524efd3a5cf79db984c3902c`, parent is the baseline |
| #4347 head | `7be0a3a17aeea0a7c0bf7bc63ae56c70e53311d4`, parent is the baseline |
| ggml source | `3cf03257f219afbe7334045ff7c6a06ac68c627d` |
| Model | `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`,22,853,663,008 bytes |
| Model SHA-256 | `55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1` |
| Model revision | Not recorded; use the exact size/hash, rather than a moving model-repository revision |

These measurements test the pinned heads above. The report branch was later based on upstream `46c152a1`; that is **not** a newly measured engine. Default clocks/power settings were not changed. Background model downloading was suspended only for formal performance runs and resumed afterwards. The daily model was unloaded for experiments and restored afterwards. RAM/VRAM peaks, per-request clocks/temperatures, CPU waiting and GPU critical paths were not measured in these fixed-history runs.

## Method and timing boundaries

Both independent experiments use the same [synthetic input fixtures](fixtures/manifest.json):1025 and16385 prompt tokens, followed by the same96 forced token IDs per input. They contain fictional English/Chinese/Python ledger records, not personal prompts. The configured context stays32768. KV is FP32, MTP0, pool-workers7 plus the calling thread, `STRATA_MMQ_KQ=1`, `STRATA_Q8_PACKED=1`, GPU0 and `CUDA_MODULE_LOADING=EAGER`. `STRATA_Q36_FIXED_SLOTS=1` freezes residency **after automatic initial budgeting**; no explicit slots/VRAM budget bypasses the policy.

Every input/arm/repetition starts a new process, with only the engine's built-in one-chunk/one-decode-step warmup. Five pairs per input: AB order for pairs0/2/4, BA for pairs1/3; case order alternates too. Each experiment has20 processes. All samples are retained. Outputs are checked against the corresponding arm's prior numerical-check argmax IDs; different layouts are not required to have identical output IDs.

The native CLI evaluates97 argmax positions: the initial prompt's prediction plus96 actual single-token forwards. `generation_ms` includes the final printing of those IDs. No full-logit dump is written during performance measurement. Tables report native prefill, generation and their sum; load/warmup/HTTP/queueing are excluded. This sum is **not measured TTFT or user-facing total latency**, and97/time is not presented as standard decode throughput. Forced histories also make this a different workload from normal sampled or greedy free generation.

Rules were recorded before each experiment; original plans are in [historical-plans](historical-plans). The original #4347 free-generation experiment failed its identity gate and stopped; the later fixed-history protocol is a separate experiment, not a reinterpretation of those timings.

## #4335: chunk2048 versus4096

Same #4335 binary, packed copies retained; no #4347. Automatic selection chose4096 at configured32K context. Explicit2048/4096 left63/50 GPU expert slots per layer respectively. Larger-chunk buffer costs and the resulting expert placement are part of the policy comparison, not separately isolated factors.

| Input tokens | Stage | 2048 median ms | 4096 median ms | Median paired2048/4096 (range) | 4096 faster pairs |
| --- | --- | ---: | ---: | --- | ---: |
| 1025 | prefill | 809.1 | 808.5 | 1.0004 (1.0001–1.0016) | 5/5 |
| 1025 | generation | 2519.2 | 2594.6 | 0.9705 (0.9564–0.9742) | 0/5 |
| 1025 | sum | 3327.8 | 3403.1 | 0.9779 (0.9667–0.9804) | 0/5 |
| 16385 | prefill | 7165.2 | 5502.7 | 1.3029 (1.3005–1.3045) | 5/5 |
| 16385 | generation | 3319.7 | 3393.0 | 0.9791 (0.9557–0.9897) | 0/5 |
| 16385 | sum | 10478.0 | 8898.4 | 1.1788 (1.1697–1.1846) | 5/5 |

Ratios are computed per pair before taking the median, so they differ from dividing the two displayed medians. A ratio above1 means4096 is faster. The median paired sum ratio corresponds to about15.2% less time at16K and2.3% more time at1K. Both chunks cover the1K prompt in one segment, where prefill differences are tiny; the generation cost then dominates this comparison. No statistical significance or universal recommendation is inferred.

Evidence: [all10 pairs](data/pr4335/pairs.csv), [summary](data/pr4335/summary.json), [per-run outputs](data/pr4335/recorded-runs.json), [redacted engine logs](logs/pr4335).

## #4347: packed copies retained versus released

Same #4347 binary, chunk2048 fixed; no #4335. A sets `STRATA_Q36_KEEP_PACKED=1`; B leaves it unset. The policy released1.78GiB, increasing resident GPU slots from63 to89 per layer. Automatic initial budgeting remains enabled.

| Input tokens | Stage | A kept median ms | B released median ms | Median paired A/B (range) | B faster pairs |
| --- | --- | ---: | ---: | --- | ---: |
| 1025 | prefill | 808.5 | 839.5 | 0.9626 (0.9594–0.9767) | 0/5 |
| 1025 | generation | 2519.9 | 2518.2 | 1.0050 (0.9990–1.0458) | 3/5 |
| 1025 | sum | 3328.4 | 3360.5 | 0.9949 (0.9892–1.0252) | 2/5 |
| 16385 | prefill | 7161.8 | 7478.3 | 0.9591 (0.9556–0.9602) | 0/5 |
| 16385 | generation | 3322.9 | 3187.1 | 1.0395 (1.0270–1.0642) | 5/5 |
| 16385 | sum | 10484.7 | 10644.3 | 0.9834 (0.9795–0.9906) | 0/5 |

The16K generation direction is consistent in these five pairs, but does not offset prefill cost for this history length. The1K sum is mixed. This does not establish the critical-path cause or show that the policy is generally slower on other CPUs, prompt lengths, contexts, cache histories or output lengths.

Evidence: [all10 pairs](data/pr4347/fixed-history-pairs.csv), [summary](data/pr4347/fixed-history-summary.json), [per-run outputs](data/pr4347/recorded-runs.json), [redacted engine logs](logs/pr4347).

## Correctness, failures and environment controls

**Passed:**

- Baseline and both unchanged PR heads built natively; each ran the same16 selected host/layout/sampling tests, all passed. These are not the full CTest suite.
- #4335: for each prompt, five full-logit arms (main explicit2048/4096, PR explicit2048/4096 and PR auto). Same explicit chunk gives bitwise-identical main/PR logits; PR auto equals main explicit4096. Each comparison covers97×248320 FP32 values, all finite. Cross-chunk top1 agrees94/97 at1K and93/97 at16K; differences remain and are not converted into a quality-equivalence claim.
- #4347: main packed on/off at identical63 slots gives bitwise-identical full1K logits. At1K/16K, candidate kept63 and released89 separately equal the main controls at matched63/89 placements, bitwise across full97-row logits. Thus these two frozen histories reproduce the policy's numerical difference through existing main placements; this does not identify a particular rounding operator or validate arbitrary cache trajectories.
- Deterministic task checks: #4335 is24/24 (12 per chunk); #4347 is48/48 (12 per each of four layout controls). Tasks cover arithmetic JSON, Chinese formatting, four executed pure Python functions with three inputs each, and short/long retrieval. Full task inputs, expected values and token hashes are in [fixtures/tasks.json](fixtures/tasks.json). These are small regression checks, not a model-quality benchmark.
- All40 formal fixed-history requests exited0, matched their arm's prior97 argmax IDs and logged zero expert slot replacements.

**Failed / corrected:**

- The earlier #4347 dynamic free-generation pair diverged at output76 for1K and19 for16K (one-based). Remaining pairs were stopped under the original identity rule. Its16K state followed a different1K generation length from the earlier64-token gate. The output difference alone is not proof of a packed-read bug or quality loss. [Failure record and all actual outputs](data/pr4347/identity-failure.json); [original summary and numerical diagnosis](data/pr4347/summary.json). Those failed-identity timings are excluded from both tables above.
- The first full-logit adapter link attempt omitted the CUDA library search path; after adding it, the unchanged libraries linked and the entry matched archived original-engine logits. This was preparation failure, not a model failure. Original main/PR executables and libraries were not changed.
- Initial task scoring mistakenly included GEN's emitted EOS token248046, rendered as `<|im_end|>`. Re-scoring removed exactly that terminal transport marker, preserving raw answers and original judgments in [recorded-task-runs.json](data/pr4347/recorded-task-runs.json). No answer content or expected task result was changed.

| Experiment | HtoD before GB/s | HtoD after GB/s | Relative drift |
| --- | ---: | ---: | ---: |
| #4335 fixed history | 24.833714 | 24.855839 | 0.0891% |
| #4347 fixed history | 24.860822 | 24.835640 | 0.1013% |

Both passed the predeclared>=15GB/s and<=15% drift checks. These controls exclude the machine's previously observed≈1.7GB/s transfer fault, not every source of noise. Download processes and the daily model were restored, with the daily configuration hash unchanged.

**Not run:** broad quality equivalence, new dynamic-cache performance pairs, sampled/free-generation performance acceptance, longer output histories, other configured contexts, other VRAM sizes, batch/MTP paths, API/concurrency, long stability, critical-path profiling, and HIP/SYCL/Metal. The two PRs were not combined. Default behavior and published headline speed claims are unchanged by this report.

## Reproduce and audit

See [REPRODUCE.md](REPRODUCE.md). All archived inputs, per-run outputs and timing lines are included; full vocabulary dumps and model files are omitted. `python verify.py` checks the package and recomputes every paired statistic without loading a model.

[historical-drivers](historical-drivers) preserves the executed driver source for audit, with model paths parameterized and original/exported SHA-256 values in [provenance.json](provenance.json). They retain historical lab dependencies and are not a standalone installation path. `replay.py` and `build_adapter.py` were added during publication to remove those local dependencies. Their offline/compilation validation is documented in REPRODUCE.md; the portable GPU replay itself is not claimed as an executed measurement. Original plans are retained for chronology, including superseded failed gates.
