# Strata standard benchmark: Apple M5 Ultra

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261109T155531Z-a0a20381`, started 2026-11-09T15:55:31Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: Apple M5 Ultra (96.0 GiB shared)
- CPU: Apple M5 Ultra; RAM: 96.0 GiB
- Storage of the model files: not measured
- OS: macOS 27.2; drivers: not measured; runtime: metal 27.2
- Strata commit: `37d5c9e84514150ee74d69f4f0f0466ba4b91b51`; engine 0.1.59-metal; source build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (Setup's IQ3_S config with one change: drafting dflash2 (setup's default for IQ3_S is MTP))
- Context 65536 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 24 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 153.8 (144.1-155.0) n=3 | 0.21 (0.20-0.22) n=3 | 0.78 (0.73-0.79) n=3 | 2.5 (2.0-3.4) n=3 | 0/3 |
| prompt-4k | 3005.8 (2512.1-3006.2) n=3 | 137.1 (136.4-143.5) n=3 | 1.37 (1.37-1.64) n=3 | 4.27 (3.76-4.56) n=3 | 5.1 (4.9-5.4) n=3 | 3/3 |
| prompt-16k | 3110.0 (3088.4-3110.4) n=3 | 130.9 (126.3-132.8) n=3 | 5.29 (5.29-5.32) n=3 | 8.42 (8.41-8.50) n=3 | 9.2 (9.1-9.3) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
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
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
