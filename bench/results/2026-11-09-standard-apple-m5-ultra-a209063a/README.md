# Strata standard benchmark: Apple M5 Ultra

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261109T155736Z-a209063a`, started 2026-11-09T15:57:36Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: Apple M5 Ultra (96.0 GiB shared)
- CPU: Apple M5 Ultra; RAM: 96.0 GiB
- Storage of the model files: not measured
- OS: macOS 27.2; drivers: not measured; runtime: metal 27.2
- Strata commit: `37d5c9e84514150ee74d69f4f0f0466ba4b91b51`; engine 0.1.59-metal; source build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-q3_k_xl; quantization UD-Q3_K_XL; revision d96201194f103c906994f283f504209dc8d2a10c; source revision not measured
- Files: `unsloth-ud-q3_k_xl-metal.gguf` `9cb1688806851133...`
- Track: **default** (M5 Ultra 96 GB, source build v0.2.13 (37d5c9e), setup defaults, Unsloth v3.1 UD-Q3_K_XL, resident, 64K)
- Context 65536 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 17 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 132.8 (126.4-135.0) n=3 | 0.21 (0.19-0.21) n=3 | 0.87 (0.80-1.65) n=3 | 2.9 (2.6-3.3) n=3 | 0/3 |
| prompt-4k | 3092.6 (2527.9-3098.9) n=3 | 114.4 (99.1-136.6) n=3 | 1.34 (1.33-1.63) n=3 | 5.47 (2.51-6.79) n=3 | 5.8 (4.3-6.8) n=3 | 2/3 |
| prompt-16k | 3214.8 (3194.4-3222.8) n=3 | 118.4 (118.4-122.8) n=3 | 5.11 (5.10-5.15) n=3 | 8.00 (7.65-8.35) n=2 | 9.4 (9.3-9.4) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
