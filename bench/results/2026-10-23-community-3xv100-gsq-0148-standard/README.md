# Strata standard benchmark: Tesla PG500-216, Tesla PG500-216, Tesla V100-PCIE-32GB

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261023T164654Z-04c0b605`, started 2026-10-23T16:46:54Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: Tesla PG500-216 (32.0 GiB dedicated); Tesla PG500-216 (32.0 GiB dedicated); Tesla V100-PCIE-32GB (32.0 GiB dedicated)
- CPU: Intel64 Family 6 Model 63 Stepping 2, GenuineIntel; RAM: 127.9 GiB
- Storage of the model files: not measured
- OS: Windows 10.0.19044; drivers: gpu-driver 582.78; runtime: cuda 12.9, cuda-driver-supports 13.0
- Strata commit: `5c8dd322528cc84f68f461ad54a506b9da8fa8d4`; engine 0.1.48; source build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (no notes given by the operator)
- Context 262144 tokens; server capacity 2 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 35 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 78.9 (76.7-84.0) n=3 | 1.20 (0.84-1.28) n=3 | 2.22 (2.14-2.39) n=3 | 5.2 (5.1-5.8) n=3 | 0/3 |
| prompt-4k | 1138.7 (1129.5-1141.5) n=3 | 71.4 (70.1-72.0) n=3 | 3.64 (3.64-3.66) n=3 | 9.62 (9.56-9.69) n=2 | 10.8 (10.8-10.9) n=3 | 3/3 |
| prompt-16k | 1541.7 (1539.0-1549.4) n=3 | 72.7 (67.9-74.0) n=3 | 10.70 (10.66-10.72) n=3 | 15.14 (13.81-17.44) n=3 | 17.7 (17.5-18.2) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 5 | 7 | 0 | 0 |

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
| 1 | 24.375 | 1732 | 71.05641 | 0 |
| 2 | 25.187 | 1853 | 73.569699 | 0 |
| 3 | 25.781 | 1970 | 76.412862 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 171 request(s) before this run.
