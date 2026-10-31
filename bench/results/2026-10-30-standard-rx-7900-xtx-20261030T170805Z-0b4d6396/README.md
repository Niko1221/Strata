# Strata standard benchmark: AMD Radeon RX 7900 XTX

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261030T170805Z-0b4d6396`, started 2026-10-30T17:08:05Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon RX 7900 XTX (24.0 GiB dedicated)
- CPU: AMD Ryzen 9 5900X 12-Core Processor; RAM: 63.9 GiB
- Storage of the model files: ssd KINGSTON SKC3000D2048G (nvme)
- OS: Windows 10.0.19044; drivers: gpu-driver 32.0.32015.7029; runtime: not measured
- Strata commit: `870aedd9e1ebe1c91f94da93b59af17729f6c5c0`; engine 0.1.54; source build
- Background work: llama-swap proxy idle with a 230M llama.cpp title generator loaded on the CPU (-ngl 0); qBittorrent; ProtonVPN service; Signal; opencode; Microsoft SQL Server (sqlservr); WizTree; Windows Defender

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs; quantization IQ3_XXS; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `219ea929900dfa9e...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **default** (release 0.2.8 (installer clone of tag v0.2.8), setup choices --context 131072 --kv int8 --vision no)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 5 min 43 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 78.7 (77.6-87.6) n=3 | 1.12 (1.00-1.14) n=3 | 1.14 (1.14-1.14) n=1 | 7.6 (6.8-7.7) n=3 | 3/3 |
| prompt-4k | 746.3 (741.0-748.8) n=3 | 90.2 (89.4-93.7) n=3 | 5.53 (5.51-5.57) n=3 | 10.48 (10.17-10.80) n=2 | 11.2 (11.0-11.3) n=3 | 3/3 |
| prompt-16k | 878.1 (869.9-882.0) n=3 | 85.9 (76.3-87.7) n=3 | 18.70 (18.62-18.87) n=3 | 23.49 (20.45-24.15) n=3 | 24.6 (23.2-25.4) n=3 | 2/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 1 | 8 | 0 | 0 |
| sanity | 12 | 11 | 1 | 0 | 0 |
| concurrency | 0 | 0 | 0 | 0 | 0 |

Sanity checks: **0 of 12**.

| Case | Result | Reason |
| --- | --- | --- |
| I1 | fail | the answer is not one strict JSON value (Expecting value: line 1 column 1 (char 0)) |
| I2 | fail | the answer is not one strict JSON value (Expecting value: line 1 column 1 (char 0)) |
| I3 | fail | the answer is not one strict JSON value (Expecting value: line 1 column 1 (char 0)) |
| I4 | fail | the answer text differs from the expected text |
| R1 | fail | the answer text differs from the expected text |
| R2 | fail | the answer text differs from the expected text |
| R3 | fail | the answer text differs from the expected text |
| R4 | fail | the answer is not one strict JSON value (Expecting value: line 1 column 1 (char 0)) |
| T1 | fail | the output cap was reached before the answer was complete |
| T2 | fail | expected one structured tool call, got 0 |
| T3 | fail | expected one structured tool call, got 0 |
| T4 | fail | expected one structured tool call, got 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-short-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-short-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
