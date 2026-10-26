# Strata standard benchmark: Tesla V100-PCIE-16GB

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261026T034014Z-2c720199`, started 2026-10-26T03:40:14Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: Tesla V100-PCIE-16GB (16.0 GiB dedicated)
- CPU: Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz; RAM: 123.4 GiB
- Storage of the model files: nvme SAMSUNG MZVLB256HAHQ-000L2 (nvme)
- OS: Ubuntu 25.10 6.17.0-41-generic; drivers: gpu-driver 580.159.03; runtime: cuda 13.0
- Strata commit: `562a0c2ed7169ff6373f893b162106d55a828297`; engine 0.1.50; release build
- Background work: gpu_monitor sampler on port 8888 (nvidia-smi polling)

## Model and configuration

- Model: swift-1.5-iq3_s; quantization IQ3_S; revision ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF (local copy); source revision not measured
- Files: `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `46cb3996cd1a17fe...`; `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `70ce6fb48a94fe34...`; `expert-profile-learned.bin` `a8ad5807559c2dc1...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (V100-16G single card; engine 0.1.50; Swift 1.5 IQ3_S derivative; custom tuning: ple-io mmap, spec 5, spec-min-p 0.7, prefill auto:16384, prompt-cache 64+spread, park 16G/6, STRATA_SM70_TABLE=1, STRATA_ADAPT_OVERLAP=1, kv q4_0 resident 32768, 256K ctx)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 14 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 54.9 (48.3-56.2) n=3 | 1.01 (0.78-1.04) n=3 | 2.60 (2.31-9.17) n=3 | 8.1 (7.0-11.6) n=3 | 1/3 |
| prompt-4k | 932.5 (926.5-933.9) n=3 | 52.0 (51.2-55.1) n=3 | 4.42 (4.42-4.45) n=3 | 10.76 (8.86-11.46) n=3 | 14.3 (13.7-14.4) n=3 | 3/3 |
| prompt-16k | 1399.4 (1396.5-1400.8) n=3 | 54.1 (48.3-56.8) n=3 | 11.75 (11.74-11.78) n=3 | 15.92 (15.87-15.98) n=2 | 20.7 (19.7-22.3) n=3 | 2/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 4 | 8 | 0 | 0 |

Sanity checks: **12 of 12**.

| Case | Result | Reason |
| --- | --- | --- |
| I1 | pass | the expected JSON value |
| I2 | pass | the expected JSON value |
| I3 | pass | the expected JSON value |
| I4 | pass | the exact expected text |
| R1 | pass | the exact expected text |
| R2 | pass | the exact expected text |
| R3 | pass | the exact expected text |
| R4 | pass | the expected JSON value |
| T1 | pass | the expected tool call |
| T2 | pass | the expected tool call |
| T3 | pass | the expected tool call |
| T4 | pass | the expected tool call |

Concurrency rounds (four requests released together):

| Round | Wall s | Output tokens | Aggregate tok/s | Failed |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 41.208813 | 1958 | 47.514108 | 0 |
| 2 | 39.318888 | 1881 | 47.839603 | 0 |
| 3 | 44.846055 | 2048 | 45.667339 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
