# Strata standard benchmark: NVIDIA Thor

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261107T142633Z-3acef9db`, started 2026-11-07T14:26:33Z. **Status: nonconforming.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

This run is not a complete standard result:

- no model file could be hashed

## Hardware and software

- GPU: NVIDIA Thor
- CPU: aarch64; RAM: 122.8 GiB
- Storage of the model files: not measured
- OS: Ubuntu 24.04.4 LTS 6.8.12-tegra; drivers: gpu-driver 580.00; runtime: cuda 13.0
- Strata commit: `5819cc0b7c24c3716eed0e50de1e674ecafd3171`; engine 0.1.56; source build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision not measured; source revision not measured
- Files: not measured
- Track: **custom** (no notes given by the operator)
- Context 32768 tokens; server capacity 4 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 4 min 0 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 41.3 (40.7-43.3) n=3 | 0.89 (0.86-0.95) n=3 | 2.84 (1.93-2.88) n=3 | 8.7 (8.0-9.6) n=3 | 0/3 |
| prompt-4k | 972.7 (898.3-984.6) n=3 | 35.9 (33.6-37.6) n=3 | 4.25 (4.20-4.60) n=3 | 16.10 (16.10-16.10) n=1 | 18.4 (17.8-19.8) n=3 | 3/3 |
| prompt-16k | 1128.1 (1053.8-1132.5) n=3 | 36.5 (35.9-38.2) n=3 | 14.58 (14.52-15.61) n=3 | 25.56 (25.56-25.56) n=1 | 28.8 (28.0-29.6) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 3 | 9 | 0 | 0 |

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
| 1 | 35.71571 | 1898 | 53.141881 | 0 |
| 2 | 37.030958 | 2038 | 55.035033 | 0 |
| 3 | 36.650721 | 1964 | 53.58694 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 1 request(s) before this run.
