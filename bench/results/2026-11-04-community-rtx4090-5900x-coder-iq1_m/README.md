# Strata standard benchmark: NVIDIA GeForce RTX 4090

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261104T170032Z-dc084574`, started 2026-11-04T17:00:32Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 4090 (24.0 GiB dedicated)
- CPU: AMD Ryzen 9 5900X 12-Core Processor; RAM: 63.9 GiB
- Storage of the model files: ssd Samsung SSD 980 PRO 1TB (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 621.01; runtime: cuda 13.5
- Strata commit: `c5ab5d4003161333044f6f16f5dacd40b7be6d9a`; engine 0.1.56; release build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-coder-iq1_m; quantization IQ1_M; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf` `bfa3644db4adfff0...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (no notes given by the operator)
- Context 131072 tokens; server capacity 3 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first <key>mpty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 16 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 108.5 (95.5-114.7) n=3 | 0.55 (0.42-0.58) n=3 | 1.39 (1.16-5.33) n=3 | 4.1 (3.4-5.9) n=3 | 1/3 |
| prompt-4k | 4052.9 (4006.5-4071.0) n=3 | 99.9 (96.2-117.9) n=3 | 1.05 (1.03-1.05) n=3 | 4.00 (3.02-4.99) n=2 | 6.2 (5.3-6.3) n=3 | 2/3 |
| prompt-16k | 5389.9 (5345.3-5430.5) n=3 | 117.4 (105.0-120.7) n=3 | 3.08 (3.05-3.09) n=3 | 5.31 (5.17-6.64) n=3 | 7.4 (7.3-7.9) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 290 request(s) before this run.
