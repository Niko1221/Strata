# Strata standard benchmark: NVIDIA GeForce RTX 3060

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261121T012734Z-d34f6cd3`, started 2026-11-21T01:27:34Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3060 (12.0 GiB dedicated)
- CPU: AMD Ryzen 7 255 w/ Radeon 780M Graphics; RAM: 59.6 GiB
- Storage of the model files: nvme Samsung SSD 980 1TB (nvme)
- OS: Ubuntu 26.04.1 LTS 7.0.0-43-generic; drivers: gpu-driver 595.99.02; runtime: cuda 13.2
- Strata commit: `12d491da4a6904eb49aaa6f2f63ec4b1a17582e3`; engine 0.1.63; release build
- Background work: GNOME Shell; nvidia-smi sampling every 2 seconds; other GPU clients not running

## Model and configuration

- Model: qwen3.8-flash-next-q2_0; quantization Q2_0; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision de4b8e4d43b917e7706784d8bb445c9af86a3540
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` `69820c02ec7d0b45...`; `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (Q2_0; context=64k; parallel=1; vision=off; drafting=off; GPU Unlock=on; eGPU PCIe Gen4 x4; Strata engine 0.1.63; untuned; custom experimental config)
- Context 65536 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 43 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 47.1 (46.8-47.5) n=3 | 1.64 (1.38-1.71) n=3 | 3.48 (3.28-4.19) n=3 | 8.3 (7.3-11.7) n=3 | 0/3 |
| prompt-4k | 775.2 (772.8-775.7) n=3 | 47.1 (45.9-53.6) n=3 | 5.31 (5.31-5.33) n=3 | 14.12 (14.12-14.12) n=1 | 16.2 (14.8-16.4) n=3 | 3/3 |
| prompt-16k | 1319.4 (1318.2-1325.5) n=3 | 46.5 (45.7-47.5) n=3 | 12.46 (12.42-12.47) n=3 | 17.82 (15.70-21.17) n=3 | 23.1 (21.0-23.6) n=3 | 2/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 4 | 5 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 7 | 5 | 0 | 0 |

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
| 1 | 40.411015 | 1579 | 39.073505 | 0 |
| 2 | 49.354136 | 1988 | 40.280312 | 0 |
| 3 | 43.735062 | 1768 | 40.425231 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 1 request(s) before this run.
