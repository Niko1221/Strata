# Strata standard benchmark: NVIDIA GeForce RTX 2080 Ti

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261026T220124Z-6af0a23e`, started 2026-10-26T22:01:24Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 2080 Ti (11.0 GiB dedicated)
- CPU: AMD Ryzen 5 3600X 6-Core Processor; RAM: 31.9 GiB
- Storage of the model files: ssd Netac NVMe SSD 1TB (nvme)
- OS: Windows 10.0.26200; drivers: gpu-driver 621.01; runtime: cuda 13.5
- Strata commit: `562a0c2ed7169ff6373f893b162106d55a828297`; engine 0.1.50; release build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-gyro-gyro-s; quantization not measured; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf` `c6ea1b08ca312f99...`
- Track: **custom** (A3b: budget 14 + UNBUFFERED_LOAD=0)
- Context 32768 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 6 min 24 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 29.7 (25.4-30.0) n=3 | 1.25 (1.22-1.53) n=3 | 5.03 (3.23-5.94) n=3 | 15.0 (13.5-15.6) n=3 | 0/3 |
| prompt-4k | 553.2 (294.8-553.7) n=3 | 32.9 (32.3-36.6) n=3 | 7.44 (7.44-13.95) n=3 | 16.98 (10.91-19.25) n=3 | 23.3 (21.4-28.4) n=3 | 2/3 |
| prompt-16k | 536.1 (528.8-553.2) n=3 | 31.6 (31.2-32.1) n=3 | 30.62 (29.67-31.06) n=3 | 41.74 (38.42-45.06) n=2 | 47.0 (45.6-47.2) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 4 | 5 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 0 | 0 | 0 | 0 | 0 |

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

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 1 request(s) before this run.
