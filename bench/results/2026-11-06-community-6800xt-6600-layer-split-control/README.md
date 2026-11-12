# Strata standard benchmark: AMD Radeon RX 6800 XT, AMD Radeon(TM) Graphics

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261106T192622Z-ba785608`, started 2026-11-06T19:26:22Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Context

This is the control run for [the helper-cache benchmark](../2026-11-06-community-6800xt-6600-helper-cache/README.md):
the same machine, model and engine, minutes apart, with the RX 6600 as a layer-split stage (layers
0-37 / 38-47, found by the engine's own search) instead of an expert-cache helper. It exists so that the
helper numbers are compared against this machine on the same day — not against the 2026-11-05 report,
whose expert arena ran on large pages. Same protocol, same corpus, same client; `strata bench run
--suite standard-v1`. The split run also ran its expert arena on 4 KB pages.

## Hardware and software

- GPU: AMD Radeon RX 6800 XT; AMD Radeon(TM) Graphics
- CPU: AMD Ryzen 7 9700X 8-Core Processor; RAM: 61.7 GiB
- Storage of the model files: ssd NVMe KINGSTON SFYRD20 (nvme)
- OS: Windows 10.0.26300; drivers: not measured; runtime: not measured
- Strata commit: `c3b3f5035506f3e966f1b4b7b188aeaabae92f1d`; engine 0.1.57; release build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs; quantization IQ3_XXS; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `219ea929900dfa9e...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (no notes given by the operator)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 6 min 20 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 42.8 (40.2-46.7) n=3 | 2.03 (1.89-2.09) n=3 | 3.88 (3.31-8.66) n=3 | 12.0 (9.2-14.0) n=3 | 1/3 |
| prompt-4k | 375.5 (370.5-378.2) n=3 | 37.4 (36.1-39.2) n=3 | 10.95 (10.86-11.11) n=3 | not measured | 24.6 (23.9-25.2) n=3 | 3/3 |
| prompt-16k | 640.4 (637.8-643.5) n=3 | 38.6 (36.0-43.9) n=3 | 25.64 (25.50-25.73) n=3 | 30.33 (30.33-30.33) n=1 | 38.7 (37.4-39.8) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 2 | 7 | 0 | 0 |
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
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 1 request(s) before this run.
