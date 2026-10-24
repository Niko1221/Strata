# Strata standard benchmark: AMD Radeon RX 6900 XT

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261023T175937Z-8f3712c4`, started 2026-10-23T17:59:37Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon RX 6900 XT (16.0 GiB dedicated)
- CPU: Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz; RAM: 63.8 GiB
- Storage of the model files: ssd WDC WDS200T2B0C-00PXH0 (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 32.0.21045.16010; runtime: not measured
- Strata commit: `7b3e2eaa0b69ff212a6aa5752d9020bce9ddc1e1`; engine 0.1.48; release build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision de4b8e4d43b917e7706784d8bb445c9af86a3540
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (the default-track run (same release, package, model files and launch settings) with one engine option added: --vram-reserve-mib 700 (the Linux default; Windows keeps 2560 MiB on a card this size, #1376) to see what the 16 GB card loses to that reserve)
- Context 65536 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 13 min 6 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 17.2 (14.9-35.3) n=3 | 2.52 (2.41-2.62) n=3 | 10.86 (3.44-12.22) n=3 | 13.1 (11.6-16.9) n=3 | 1/3 |
| prompt-4k | 460.4 (458.0-465.2) n=3 | 18.7 (18.2-30.2) n=3 | 8.95 (8.84-9.00) n=3 | 18.37 (14.02-22.72) n=2 | 36.2 (18.5-37.2) n=3 | 2/3 |
| prompt-16k | 487.2 (484.4-488.4) n=3 | 24.8 (16.7-30.5) n=3 | 33.70 (33.61-33.89) n=3 | 42.02 (37.27-46.78) n=2 | 50.6 (46.4-54.2) n=3 | 2/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 4 | 5 | 0 | 0 |
| sanity | 12 | 10 | 2 | 0 | 0 |
| concurrency | 12 | 2 | 10 | 0 | 0 |

Sanity checks: **0 of 12**.

| Case | Result | Reason |
| --- | --- | --- |
| I1 | fail | the answer is not one strict JSON value (Expecting value: line 1 column 1 (char 0)) |
| I2 | fail | the answer is not one strict JSON value (Expecting value: line 1 column 1 (char 0)) |
| I3 | fail | the answer is not one strict JSON value (Expecting value: line 1 column 1 (char 0)) |
| I4 | fail | the output cap was reached before the answer was complete |
| R1 | fail | the answer text differs from the expected text |
| R2 | fail | the answer text differs from the expected text |
| R3 | fail | the output cap was reached before the answer was complete |
| R4 | fail | the answer is not one strict JSON value (Expecting value: line 1 column 1 (char 0)) |
| T1 | fail | expected one structured tool call, got 0 |
| T2 | fail | expected one structured tool call, got 0 |
| T3 | fail | expected one structured tool call, got 0 |
| T4 | fail | expected one structured tool call, got 0 |

Concurrency rounds (four requests released together):

| Round | Wall s | Output tokens | Aggregate tok/s | Failed |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 51.891 | 1434 | 27.63485 | 0 |
| 2 | 47.812 | 1615 | 33.778131 | 0 |
| 3 | 67.188 | 2048 | 30.481634 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c1: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r1-c2: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r2-c1: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r2-c2: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r2-c3: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
