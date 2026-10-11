# ROG Flow Z13 128 GiB: Windows HIP, Huihui UD-IQ4_XS, 256K configuration report

Measured on October 10, 2026 (America/Los_Angeles; UTC records extend into October 11), submitted by xszwow. This is a results-only report for Strata **0.1.42**, commit **61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406**. It records a local configuration search for coding and reasoning, including unsuccessful answers and failed attempts. It does not establish a global optimum or a win over a different online setup.

The final same-machine comparison improved weighted harmonic decode throughput from **46.41 to 48.87 tok/s (+5.31%)** across six tasks. Each case has **two measured repetitions**, with 512 output tokens. Completed xhigh coding answers were slower than those truncated throughput prefixes. Three full uncached prompts near 261K tokens ran successfully; a fourth request tested prefix reuse.

## Hardware and software

- ASUS ROG Flow Z13 2025, AMD Ryzen AI Max+ 395, Radeon 8060S (`gfx1151`), 128 GiB installed unified memory; 96 GiB assigned to the GPU. That GPU allocation is part of the 128 GiB, not additional RAM. Windows reports 33,185,604 KiB of visible physical RAM (about 31.65 GiB).
- Windows 11 Home, 10.0.26200/build 26200; AMD display driver `32.0.31032.1003`; bundled ROCm `10.2.0a20260930`; hipBLASLt `100500`, with the attached matching 32-row tuning table.
- Windows HIP release engine; executable and preparation asset hashes are in [provenance.json](provenance.json). No engine source modification was used for these text measurements.
- KIOXIA KBG60ZNS1T02 NVMe SSD, 1,024,209,543,168 bytes. PCIe link width/speed: not measured.
- AC power and Armoury Crate Turbo. Sustained watts, temperatures and aggregate peak GPU memory: not measured. Normal desktop background activity was not exhaustively logged.

## Model and exact configurations

- Model: [huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF), revision `c219df4f109fa42c402c44c702c4e33e2ce05fe2`, `unsloth-UD-IQ4_XS`, all three `Qwen3.8-Flash-Next-UD-IQ4_XS-0000{1,2,3}-of-00003.gguf` shards. Total download: 93,682,584,224 bytes (87.25 GiB).
- Native pack prepared with `tools/iq_pack.py --compat-bf16`. Expert and PLE weights remain native in the GGUF shards; no full `experts.bin` copy. Dense compatibility conversion includes BF16 rounding; this is not a claim that every tensor remains bit-identical to its GGUF representation. Conversion and pack hashes are recorded.
- Official MTP tensors from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`; Strata's `mtp_pack.py`/`mtp_rt.py`, Q2_0 experts, official CJK-inclusive 106,299-token draft vocabulary. This is the official draft layer used with the Huihui target, not a separately abliterated MTP layer.
- Original frequency-ranked `data/expert-profile.bin`; 24,576 of 24,576 experts loaded and slot zero verified. Experimental sorted profile was measured but not adopted.
- No additional calibration file was used. The engine automatically selected its gfx1151 defaults and logged its existing experimental native Q5_K output head. This report does not claim an entirely unquantized or uniformly IQ4 target.
- One GPU, one session/slot, 262,144-token **total context capacity**, int8 KV. Input and output must fit together; 256K capacity does not mean 256K input plus unlimited output.

| Setting | Original comparison | Selected daily text deployment |
| --- | --- | --- |
| Speculative window / acceptance floor | `--spec 4`, request floor 0 | `--spec 6 --spec-min-p 0.4` |
| Prefill | 8,192 | 16,384 |
| Expert cache | 24,576 | 24,576 |
| CPU pool workers / host | 8 / first | 8 / first |
| MTP history window | 8,192 | 8,192 |
| GPU reserve | 4,096 MiB | 4,096 MiB |
| Host expert budget | implicit | requested 1 GiB; effective 0 in the audited startup |
| Direct file loading | original buffered setting | `STRATA_UNBUFFERED_LOAD=1` |
| Prompt cache | 0 in comparison | 6 in daily deployment; **0 in scored comparisons** |
| Parked conversation cache | 0 MiB | 0 MiB |
| CPU prefill / adaptive swaps | `--pcie-frac 0 --adapt-every 0` | same; `--no-prefill-borrow` |
| Vision | disabled | disabled for this report's measurements |

See [config-original-comparison.json](config-original-comparison.json), [config-selected.json](config-selected.json) and the exact per-profile configurations in [profiles.json](profiles.json), rather than treating this table as the full command. `${WORKSPACE}` replaces the private deployment root and must be resolved before use. [REPRODUCE.md](REPRODUCE.md) describes preparation and replay.

**Reasoning settings:** the saved server fallback is `reasoning_effort: high`. The xhigh tests explicitly send `reasoning_effort: xhigh`; they are not evidence that every client automatically requests xhigh. Daily deployment sets `reasoning_budget_tokens: 8192`. Completed quality tests used 16,384 total output tokens and an explicit 8,192 thinking budget. The 12-turn test used an 8,192 total output cap; the server reserves 2,048 answer tokens and clamps its thinking budget to 6,144. An 8,192 thinking budget is a tested latency/answer-completion choice, not proof of maximum reasoning accuracy.

## Method and data

The search includes **780 formal throughput requests**, split as follows: screen 128; refine 90; refine2 72; refine3 45; startup 12; startup order 18; prefill 24; large prefill 12; workers 45; drafting 108; draft temperature 48; window 18; KV 16; interactions 72; final generalization 72. Frozen run manifests, candidate configurations and selection decisions are included in [profiles.json](profiles.json) and [decisions-and-audits.json](decisions-and-audits.json).

Each candidate ran alone. Kernel-capture/initial sampled-request warmups were excluded from throughput summaries (64-token greedy and xhigh warmups). Prompt reuse was disabled in those measurements: `cache_prompt=false`, checkpoint false, engine prompt cache 0. Repetition order was shuffled with seed `1977 + repetition`. Expert caches were preloaded/warmed and retained within a profile. OS/SSD page caches were not cleared; “uncached” here refers to prompt reuse. Model loading is excluded from prompt/decode throughput and reported separately for fresh process starts.

Long-code fixtures are public llama.cpp `src/*.cpp` text at pinned revision `3cf03257f219afbe7334045ff7c6a06ac68c627d`, concatenated in filename order with control literals escaped. The exact messages are included; attribution/license is recorded in [REPRODUCE.md](REPRODUCE.md). Other long-context records are synthetic sensor logs with three markers. Short throughput prefixes use greedy/T=0, sampled code/T=0.6, and xhigh/T=1.0. Full per-request sampling, seed, output cap and thinking flags are retained.

- [runs.jsonl](runs.jsonl): 809 recorded requests: 780 formal throughput, 10 completed quality, 4 full-context, 12 conversation turns, 3 deployment checks. Response text/reasoning, engine timings, grading, latency and host-memory samples are retained. `Request.MessagesRef` resolves into [messages.json](messages.json); stability requests are reconstructed from the stored transcript and original driver parameters.
- [timings.csv](timings.csv): one row per record; milliseconds for engine times, seconds for client latency, GiB for host memory. Empty cells mean unavailable, not zero.
- `prompt_n` is freshly processed tokens; `cache_n` is reused tokens; their sum is total input for a normal prompt. `predicted_n` and decode throughput include reasoning tokens. Quality forced thinking-wrap can introduce additional internal prompt processing.
- Host headroom uses `GlobalMemoryStatusEx`, sampled every 0.5 seconds where present. `MaximumCommitTotalGiB` is the **system commit limit/capacity**, not peak committed usage. Available commit and available RAM are different counters.
- TTFT was **not measured**: these requests were non-streaming. Prompt processing duration must not be called TTFT.

## Final same-machine comparison

Each row: 2 runs per configuration, actual input counts shown, 0 reused tokens, 512 generated tokens per run. Median decode and range are in tok/s. These are throughput prefixes, not completed-answer accuracy tests.

| Case | Input tokens | Original decode median (range) | Selected decode median (range) | Original prompt median | Selected prompt median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy code | 83 | 48.15 (48.0–48.3) | 51.90 (51.8–52.0) | 135.35 | 133.25 |
| Sampled code | 67 | 47.45 (47.3–47.6) | 49.60 (49.3–49.9) | 111.40 | 112.15 |
| xhigh prefix | 108 | 47.50 (47.4–47.6) | 51.15 (51.1–51.2) | 147.45 | 147.20 |
| Long C++ 8K | 8,313 | 49.05 (48.9–49.2) | 48.10 (48.0–48.2) | 993.05 | 1,091.30 |
| Long C++ 32K | 32,889 | 43.90 (43.8–44.0) | 46.30 (46.1–46.5) | 1,034.45 | 1,076.55 |
| Long C++ 128K | 131,193 | 42.15 (42.0–42.3) | 45.45 (45.3–45.6) | 1,001.25 | 1,026.85 |

The weighted harmonic score uses weights 0.35 for each short coding case and 0.25 for xhigh and each long-code case: `sum(w) / sum(w / median_decode)`. It is a local workload preference, not a universal benchmark score. The final selected profile improved this score by 5.3059% over the original comparison; its 8K decode median was 1.94% slower. Two repetitions are below the community guide's suggested three and are not a statistical confidence claim.

Selected prefill remained 16,384. A 32,768 alternative improved prompt score only about 0.62% while regressing 32K decode about 7.95%; it was rejected. Four workers improved the score only 0.30%, below the 3% adoption threshold; eight were retained. MTP draft sampling modes, temperature gating, 4K/16K MTP windows, alternate KV formats and coupled interaction candidates did not clear their adoption/5% regression guards. The sorted startup profile's 8.57% gain did not meet its 15% adoption threshold. All measured candidates, including rejected ones, remain in the data.

One short no-speculation control removed MTP and disabled suffix/lookup drafts, with draft counters zero. Greedy output was identical: 52.8 vs 23.2 tok/s (2.28×). This compares **MTP plus suffix speculation** against no speculation, not an isolated MTP-only gain.

## Completed quality, full context, and startup checks

The quality arm used one completed answer per task/profile, xhigh, T=1, top-p=.95, top-k=20, seed=12345. Three Python functions passed 94 tests per profile (topological sort 26, edit distance 38, interval scheduling 30), including non-mutation/contract review. Arithmetic result 234168 passed in both profiles. The exact-count DP task **failed in both**: expected **601380693** (independently checked with two recurrences and a combinatorial cross-check), original answered 241183, selected answered 22006125. All ten completed with normal stop. These results support no observed regression on this small set, not broad accuracy equivalence.

| Completed task | Original / selected decode tok/s | Original / selected generated tokens |
| --- | ---: | ---: |
| Topological sort | 33.4 / 33.4 | 8,619 / 8,752 |
| Edit distance | 36.1 / 38.4 | 1,240 / 681 |
| Interval scheduling | 32.7 / 33.0 | 7,975 / 3,481 |
| Arithmetic reasoning | 44.0 / 46.3 | 2,008 / 1,856 |
| Exact-count DP (both wrong) | 31.0 / 32.2 | 9,133 / 9,307 |

Output lengths differ, so complete latency reductions cannot all be attributed to engine speed. [code_oracle.py](code_oracle.py) is the final restricted algorithm grader; stored grading and review evidence are included.

| Full-context case | Total input | Reused | Newly processed | Output | Prompt tok/s | Decode tok/s | Client seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Three-marker recall | 260,934 | 0 | 260,934 | 31 | 977.2 | 34.3 | 268.078 |
| BFS implementation | 261,056 | 0 | 261,056 | 364 | 941.8 | 44.0 | 285.703 |
| Cache first request | 260,934 | 0 | 260,934 | 31 | 933.2 | 43.4 | 280.500 |
| Cache repeat | 260,934 | 260,927 | 7 | 31 | 79.7 | 41.4 | 0.969 |

All recall replies contained ORCHID7319, MAPLE2846 and CEDAR9052. The full-context BFS passed 41 tests and manual O(V+E) review. Cache-repeat prompt throughput reflects just seven fresh tokens, not rereading 261K. Each row is one measured request, not a median.

Twelve successive coding turns in one conversation passed **376 checks**, all normal stops, with prefix reuse on all eleven follow-ups. Decode ranged 32.6–39.6 tok/s; largest input was 41,118 tokens. This does not cover a multi-turn conversation near 256K or editing a large real project.

Three fresh native process starts became ready in **92.531, 91.407, 91.281 seconds**; each passed a complete 26-test topological-sort check and verified actual command/defaults/context capacity. They are fresh process starts with retained OS/SSD caches, not cold-disk timings.

## Observed memory and CPU/GPU placement

[startup-allocation.log](startup-allocation.log) is the final audited text startup segment, not a tail combining multiple launches. It records:

| Component / placement | Observed allocation or counter |
| --- | --- |
| GPU expert cache | 24,576 / 24,576 slots, **55.43 GiB**; prefilled in 76.2 s at 781 MB/s |
| Expert RAM tier | requested 1 GiB clamped by commit guard to **0 GiB**; resident/pinned complement 0; no missing GPU experts |
| Token embedding | Q8_0 in mapped host memory, 644 MiB |
| Dense weights | 1,416 MiB pack weights; 495 native projections, 3,619.14 MiB |
| Decode matrix packing | 84 Q8_0 matrices, +1,912.5 MiB; HC Q8 projections 0.63 GiB |
| MTP draft layer | 1,090 MiB reported, including 675 MiB experts and 111 MiB dense |
| Main head | Q6 packed head +497.3 MiB; native Q5_K representation 521,472,000 bytes |
| MTP draft head | 106,299-token head 212.9 MiB; extra Q6 packed representation +212.9 MiB |
| Verification buffers | up-to-eight-token allocation, 100 MiB device buffers; selected speculation length remains 6 |
| Free GPU memory | 30,313 MiB after loading; 30,073 MiB after the logged graph captures |
| CPU participants | 8 workers on logical processors 2,4,6,8,10,12,14,16; draining host thread on 0 |

These are log component counters, some with overlapping representations; **do not sum them into an asserted peak VRAM usage**. KV type/window/context are recorded, but KV bytes were not measured separately. Every expert is on GPU; the engine reports no CPU prefill expert share and zero request-time expert file reads in the audited startup request. CPU work, mapped PLE/embedding access and file mappings still exist.

Across the full-context requests, minimum available host RAM was **16.287 GiB**, minimum available commit **0.160 GiB**; maximum observed system commit limit was **104.107 GiB**. The conversation test reached minimum available RAM **15.140 GiB** and commit **0.195 GiB**. The final startup logged only 0.9 GiB commit free out of a 96.7 GiB limit and emitted its commit warning. These are observed constraints, not a comfortable memory margin or a recommendation to shrink the page file.

## Preserved failures and limits

- Invalid `--spec 1` attempted as an off control failed before formal requests; the corrected no-speculation control removed the draft paths. An initial spec7 allocation failure also had zero completed requests; spec7 was successfully rerun later. [failures.json](failures.json) retains these attempts outside successful throughput summaries.
- An earlier unbounded-thinking topological-sort run exhausted 16,384 tokens without an answer; the protocol was changed to an explicit thinking budget. The older failure/protocol record is preserved in the audit export.
- Initial graders incorrectly rejected a fenced result, `dict.fromkeys`, and `list.insert`. Original records/hashes and explicit regrade evidence are preserved. Corrections regraded stored replies; they were not replacement model answers. The exact-count DP failure remains a failure.
- The engine warns that large Windows HIP prefill chunks have crashed on another GPU (#1630). This machine's accepted profile logged no fallback and completed the recorded 256K tests; that does not establish stability on other hardware or remove the small Windows commit margin.
- No measured overall win over the Linux 70 W online Strix Halo report: OS, Huihui vs original target, output length, sampling, context allocation and repetition protocol differ. No llama.cpp/Halogen paired comparison is claimed here.

## Later vision configuration (separate evidence)

[config-vision-later.json](config-vision-later.json) documents the subsequent daily configuration with a Vulkan vision encoder, BF16 mmproj, GPU enabled, 16 threads, min/max image tokens 1,024/8,192. It is **not** the configuration used for the 780-request text search, and the text throughput/memory figures above must not be attributed to simultaneous vision usage.

Related [PR #1946](https://github.com/Niko1221/Strata/pull/1946) addresses GPU vision warm-up above 4,096 image tokens. Its separate Windows Vulkan comparison recorded encoder GPU allocations at READY of 1,699,991,552 bytes before and 2,305,748,992 after the patch (+605,757,440 bytes preallocated); the first 4K image reached 2,305,748,992 in both. One prewarm observation was 8.2 vs 33.2 seconds. That change and its visual correctness/build evidence belong to #1946, not this results-only text configuration report. The later vision configuration has no equivalent 780-run tuning/stability coverage in this submission.
