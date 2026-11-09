# Strata standard benchmark: Apple M5 Ultra

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261109T155927Z-876abcf7`, started 2026-11-09T15:59:27Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: Apple M5 Ultra (96.0 GiB shared)
- CPU: Apple M5 Ultra; RAM: 96.0 GiB
- Storage of the model files: not measured
- OS: macOS 27.2; drivers: not measured; runtime: metal 27.2
- Strata commit: `37d5c9e84514150ee74d69f4f0f0466ba4b91b51`; engine 0.1.59-metal; source build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-iq4_xs-v31; quantization UD-IQ4_XS-v31; revision d96201194f103c906994f283f504209dc8d2a10c; source revision not measured
- Files: `unsloth-ud-iq4_xs-v31-metal.gguf` `6182804a8766314d...`
- Track: **default** (M5 Ultra 96 GB, source build v0.2.13 (37d5c9e), setup defaults, Unsloth v3.1 UD-IQ4_XS, setup chose SSD streaming, DFlash2, 64K)
- Context 65536 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 9 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 65.8 (65.0-75.2) n=3 | 0.46 (0.37-0.49) n=3 | 1.72 (1.30-2.62) n=3 | 5.7 (4.1-6.0) n=3 | 0/3 |
| prompt-4k | 2370.0 (2310.7-2389.9) n=3 | 81.2 (79.4-81.5) n=3 | 1.75 (1.73-1.79) n=3 | 4.83 (4.32-5.17) n=3 | 8.1 (8.0-8.2) n=3 | 3/3 |
| prompt-16k | 2423.9 (2398.0-2429.4) n=3 | 65.0 (57.9-74.0) n=3 | 6.79 (6.77-6.86) n=3 | 12.84 (12.84-12.84) n=1 | 14.6 (13.7-15.7) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
