# Strata standard benchmark: NVIDIA GeForce RTX 3060

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261121T005301Z-2660aa1d`, started 2026-11-21T00:53:01Z. **Status: complete.**

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
- Track: **custom** (Q2_0; context=32k; parallel=1; vision=off; drafting=off; GPU Unlock=on; eGPU PCIe Gen4 x4; Strata engine 0.1.63; untuned; custom experimental config)
- Context 32768 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 43 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 47.7 (46.5-48.2) n=3 | 1.63 (1.37-1.69) n=3 | 3.65 (2.69-5.52) n=3 | 9.3 (8.2-9.4) n=3 | 0/3 |
| prompt-4k | 780.3 (777.0-781.7) n=3 | 46.3 (45.1-48.5) n=3 | 5.27 (5.27-5.30) n=3 | 13.38 (11.64-15.11) n=2 | 16.3 (15.8-16.6) n=3 | 3/3 |
| prompt-16k | 1378.0 (1368.3-1378.9) n=3 | 46.8 (45.5-47.0) n=3 | 11.93 (11.92-12.02) n=3 | 20.67 (19.50-21.83) n=2 | 22.9 (22.8-23.1) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 10 | 2 | 0 | 0 |

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
| 1 | 44.516135 | 1760 | 39.536227 | 0 |
| 2 | 42.372957 | 1669 | 39.38833 | 0 |
| 3 | 39.520746 | 1548 | 39.169301 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 1 request(s) before this run.
