# Strata standard benchmark: NVIDIA GeForce RTX 4080

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261119T225536Z-0c26b232`, started 2026-11-19T22:55:36Z. **Status: nonconforming.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

This run is not a complete standard result:

- the Strata commit is unknown (not a git checkout and no COMMIT file)

## Hardware and software

- GPU: NVIDIA GeForce RTX 4080 (16.0 GiB dedicated)
- CPU: AMD Ryzen 9 5950X 16-Core Processor; RAM: 125.7 GiB
- Storage of the model files: not measured
- OS: Garuda Linux 7.3.5-zen1-1-zen; drivers: gpu-driver 620.73.04; runtime: cuda 13.5
- Strata commit: `not measured`; engine 0.1.62; release build
- Background work: KDE Plasma desktop session, browser with this agent GUI, discord, telegram (GPU-composited)

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (engine 0.1.62 local source build (arch 89, CUDA 13.5); prod tuning: --pool-workers 12 --pool-tasks 96 --expert-cache auto --vram-reserve-mib 1400 --mtp mtp/rt + cyrillic draft-vocab --kv int8 --kv-resident 32768 --prompt-cache 32 --conversation-cache-mib 24576 slots 2; env STRATA_KV_PREFETCH=1; vision CPU; GPU is factory-unlocked (gpu_unlock DKMS, 3135MHz max SM, 340W limit))
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 35 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 71.4 (59.1-71.9) n=3 | 0.70 (0.62-0.70) n=3 | 1.94 (1.83-2.05) n=2 | 5.3 (4.8-9.3) n=3 | 1/3 |
| prompt-4k | 1015.4 (1009.1-1021.7) n=3 | 62.8 (62.3-71.0) n=3 | 4.06 (4.04-4.09) n=3 | 8.68 (7.62-9.75) n=2 | 12.2 (11.2-12.3) n=3 | 3/3 |
| prompt-16k | 1992.6 (1991.0-1999.5) n=3 | 65.2 (59.6-66.7) n=3 | 8.26 (8.23-8.27) n=3 | 13.48 (12.21-14.75) n=2 | 16.1 (15.9-16.8) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-short-r1: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
