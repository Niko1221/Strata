# Strata standard benchmark: AMD Radeon RX 6800 XT

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261104T060442Z-9723209f`, started 2026-11-04T06:04:42Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon RX 6800 XT (16.0 GiB dedicated)
- CPU: AMD Ryzen 9 3950X 16-Core Processor; RAM: 127.9 GiB
- Storage of the model files: ssd WDC WDS500G2B0C-00PXH0 (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 32.0.21045.11001; runtime: not measured
- Strata commit: `c5ab5d4003161333044f6f16f5dacd40b7be6d9a`; engine 0.1.56; release build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-q4_k_xl; quantization UD-Q4_K_XL; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` `4448186216b3af4c...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf` `3f342f1c1580473f...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf` `56758f40269cad5c...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf` `753bda48b98ba4f1...`
- Track: **custom** (Strata 0.2.10 / engine 0.1.56 hip-rdna2 / 2x RX 6800 XT 16 GB x16 plus x4 layer split K=42 / Ryzen 9 3950X / 128 GB DDR4-3200 4x32 / Windows 11 / UD-Q4_K_XL 32K ctx / --kv int8 / STRATA_RDNA2_SGEMM=1 / local patch allows 2 AMD cards on Windows)
- Context 32768 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 7 min 58 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 29.6 (28.8-32.8) n=3 | 3.33 (3.14-3.39) n=3 | 7.81 (5.83-12.59) n=3 | 17.2 (14.3-20.5) n=3 | 0/3 |
| prompt-4k | 331.3 (330.7-332.1) n=3 | 26.4 (26.1-27.6) n=3 | 12.42 (12.39-12.44) n=3 | 28.89 (28.89-28.89) n=1 | 31.8 (30.9-31.9) n=3 | 3/3 |
| prompt-16k | 460.6 (460.1-461.5) n=3 | 28.5 (25.7-30.6) n=3 | 35.64 (35.58-35.69) n=3 | 48.55 (46.17-50.94) n=2 | 53.6 (52.3-55.5) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 1 request(s) before this run.
