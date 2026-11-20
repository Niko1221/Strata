# Strata standard benchmark: NVIDIA GeForce RTX 4080

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261119T231845Z-49c4bc57`, started 2026-11-19T23:18:45Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 4080 (16.0 GiB dedicated)
- CPU: AMD Ryzen 9 5950X 16-Core Processor; RAM: 125.7 GiB
- Storage of the model files: not measured
- OS: Garuda Linux 7.3.5-zen1-1-zen; drivers: gpu-driver 620.73.04; runtime: cuda 13.5
- Strata commit: `64524fb9fc82222ddce554c0643b64f056ec8366`; engine 0.1.62; release build
- Background work: KDE Plasma desktop, browser, discord, telegram (GPU-composited)

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (prod iq3_s config (fused OFF) plus --control-vector Huihui-refusal-direction --cvec-mode project --cvec-dir default per-layer layers 1..47 (47 steered); engine 0.1.62 source arch 89 CUDA 13.5; unlocked RTX 4080)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 28 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 69.6 (65.3-73.7) n=3 | 0.67 (0.61-0.70) n=3 | 2.07 (1.75-2.85) n=3 | 5.4 (4.4-7.2) n=3 | 0/3 |
| prompt-4k | 1026.9 (1021.7-1029.7) n=3 | 66.6 (63.6-73.1) n=3 | 4.02 (4.01-4.04) n=3 | 7.65 (6.59-10.40) n=3 | 11.1 (11.0-12.0) n=3 | 2/3 |
| prompt-16k | 2006.1 (1999.9-2007.8) n=3 | 65.5 (63.9-68.8) n=3 | 8.20 (8.20-8.23) n=3 | 15.14 (15.00-15.29) n=2 | 16.0 (15.6-16.2) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
