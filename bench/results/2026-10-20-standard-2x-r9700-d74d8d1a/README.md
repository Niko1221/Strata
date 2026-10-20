# Strata standard benchmark: 2x AMD Radeon AI PRO R9700

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261020T124137Z-d74d8d1a`, started 2026-10-20T12:41:37Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: 2x AMD Radeon AI PRO R9700 32 GB (gfx1201; Navi 48), layer split across both cards
- CPU: AMD EPYC 7452 32-Core Processor; RAM: 123.4 GiB
- Storage of the model files: nvme Fanxiang S500Pro 2TB (nvme)
- OS: Ubuntu 26.04.1 LTS 7.0.0-38-generic; drivers: not measured; runtime: not measured
- Strata commit: `9ae93f7b1fe989d04a8e404d32cddaeeff10505f`; engine 0.1.46.1; release build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (no notes given by the operator)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 48 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s |
| --- | --- | --- | --- | --- | --- |
| short | 192.2 (191.7-194.6) n=3 | 87.4 (87.4-94.4) n=3 | 0.43 (0.42-0.43) n=3 | 1.33 (1.28-1.98) n=3 | 4.5 (3.8-5.8) n=3 |
| prompt-4k | 2313.2 (2301.2-2333.3) n=3 | 84.0 (69.3-92.8) n=3 | 1.80 (1.78-1.81) n=3 | 5.14 (4.12-6.15) n=2 | 7.9 (7.3-9.2) n=3 |
| prompt-16k | 3138.1 (3130.5-3157.1) n=3 | 63.8 (61.8-71.6) n=3 | 5.26 (5.23-5.27) n=3 | 11.09 (11.09-11.09) n=1 | 13.2 (12.4-13.5) n=3 |

Cells are median (min-max) n=samples.

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
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 48 request(s) before this run.
