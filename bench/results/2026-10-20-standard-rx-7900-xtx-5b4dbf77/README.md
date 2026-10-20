# Strata standard benchmark: AMD Radeon RX 7900 XTX

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261020T105417Z-5b4dbf77`, started 2026-10-20T10:54:17Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon RX 7900 XTX (24.0 GiB dedicated)
- CPU: AMD Ryzen 9 5900X 12-Core Processor; RAM: 63.9 GiB
- Storage of the model files: ssd KINGSTON SKC3000D2048G (nvme)
- OS: Windows 10.0.19044; drivers: gpu-driver 32.0.31036.15; runtime: not measured
- Strata commit: `90767252f420e25223c739d604145ad15911d48b`; engine 0.1.45; source build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs; quantization IQ3_XXS; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `219ea929900dfa9e...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **default** (0.2.1 release, tuned config (issue 2274 flags), engine 0.1.45)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 5 min 4 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s |
| --- | --- | --- | --- | --- | --- |
| short | 80.2 (79.6-81.2) n=3 | 63.3 (54.6-108.9) n=3 | 0.99 (0.98-1.00) n=3 | 3.19 (3.19-3.19) n=1 | 6.7 (5.7-7.2) n=3 |
| prompt-4k | 847.3 (768.5-864.8) n=3 | 85.4 (73.5-109.0) n=3 | 4.86 (4.76-5.35) n=3 | 8.95 (8.95-8.95) n=1 | 10.7 (10.0-11.8) n=3 |
| prompt-16k | 1822.6 (1714.7-1825.0) n=3 | 80.6 (76.2-84.6) n=3 | 9.04 (9.03-9.60) n=3 | 12.37 (12.37-12.37) n=1 | 15.6 (15.4-15.7) n=3 |

Cells are median (min-max) n=samples.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 1 | 8 | 0 | 0 |
| sanity | 12 | 10 | 2 | 0 | 0 |
| concurrency | 0 | 0 | 0 | 0 | 0 |

Sanity checks: **0 of 12**.

| Case | Result | Reason |
| --- | --- | --- |
| I1 | fail | the answer is not one strict JSON value (Expecting value: line 1 column 1 (char 0)) |
| I2 | fail | the answer is not one strict JSON value (Expecting value: line 1 column 1 (char 0)) |
| I3 | fail | the answer is not one strict JSON value (Expecting value: line 1 column 1 (char 0)) |
| I4 | fail | the answer text differs from the expected text |
| R1 | fail | the answer text differs from the expected text |
| R2 | fail | the output cap was reached before the answer was complete |
| R3 | fail | the answer text differs from the expected text |
| R4 | fail | the answer is not one strict JSON value (Expecting value: line 1 column 1 (char 0)) |
| T1 | fail | expected one structured tool call, got 0 |
| T2 | fail | expected one structured tool call, got 0 |
| T3 | fail | expected one structured tool call, got 0 |
| T4 | fail | the output cap was reached before the answer was complete |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-short-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-short-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The runner could not tell whether the server was freshly loaded.
