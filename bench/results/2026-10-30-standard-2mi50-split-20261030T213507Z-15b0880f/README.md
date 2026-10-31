# Strata standard benchmark: AMD Radeon Graphics

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261030T213507Z-15b0880f`, started 2026-10-30T21:35:07Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon Graphics
- CPU: AMD Ryzen Threadripper 1920X 12-Core Processor; RAM: 94.3 GiB
- Storage of the model files: not measured
- OS: Ubuntu 24.04.4 LTS 6.8.0-138-generic; drivers: gpu-driver 6.10.5; runtime: rocm 7.14.0
- Strata commit: `93f866a78de921a39b92ca6bf0eb45d112682c50`; engine 0.1.52; source build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (Strata 0.2.7 + #3986 (gfx906 build), setup a1 of the split-serving measurements)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 7 min 25 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 40.9 (40.7-46.2) n=3 | 2.55 (2.24-2.60) n=3 | 5.49 (3.54-5.56) n=3 | 10.1 (9.3-15.0) n=3 | 1/3 |
| prompt-4k | 293.4 (234.2-318.6) n=3 | 43.1 (40.0-47.2) n=3 | 14.08 (12.97-17.62) n=3 | 20.55 (16.99-24.11) n=2 | 26.8 (23.8-29.5) n=3 | 3/3 |
| prompt-16k | 378.4 (377.9-378.9) n=3 | 40.5 (39.2-45.3) n=3 | 43.50 (43.43-43.54) n=3 | 52.52 (50.34-54.71) n=2 | 56.1 (54.8-56.5) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 2 | 7 | 0 | 0 |
| sanity | 12 | 11 | 0 | 1 | 0 |
| concurrency | 0 | 0 | 0 | 0 | 0 |

Sanity checks: **10 of 12**.

| Case | Result | Reason |
| --- | --- | --- |
| I1 | pass | the expected JSON value |
| I2 | pass | the expected JSON value |
| I3 | pass | the expected JSON value |
| I4 | pass | the exact expected text |
| R1 | pass | the exact expected text |
| R2 | pass | the exact expected text |
| R3 | pass | the exact expected text |
| R4 | fail | the request failed |
| T1 | pass | the expected tool call |
| T2 | fail | expected one structured tool call, got 0 |
| T3 | pass | the expected tool call |
| T4 | pass | the expected tool call |

## Failures, exclusions and limitations

- sanity-R4 failed: server_error (the engine stopped unexpectedly (exit code 1); the next request restarts it)
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
