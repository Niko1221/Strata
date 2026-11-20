# Strata standard benchmark: NVIDIA GeForce RTX 5090

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261120T160527Z-c9d1b29e`, started 2026-11-20T16:05:27Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 5090 (31.8 GiB dedicated)
- CPU: Intel(R) Core(TM) Ultra 9 285K; RAM: 60.9 GiB
- Storage of the model files: nvme KINGSTON SNV3S4000G (nvme)
- OS: Ubuntu 26.04.1 LTS 7.0.0-43-generic; drivers: gpu-driver 595.99.02; runtime: cuda 13.2
- Strata commit: `58ee3227720629ff2e41d640165ebe89c880dfef`; engine 0.1.62; release build
- Background work: Hermes-agent idle

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-iq3_xxs; quantization UD-IQ3_XXS; revision d96201194f103c906994f283f504209dc8d2a10c; source revision not measured
- Files: `Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf` `9d5644b371ffac92...`; `Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00004.gguf` `aacaff3bdb9032c2...`; `Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00004.gguf` `ef8f14e790ab471b...`; `Qwen3.8-Flash-Next-UD-IQ3_XXS-00004-of-00004.gguf` `5de8b5dbf89409ac...`
- Track: **custom** (0.2.16/engine 0.1.62, UD-IQ3_XXS, 262K kv-int8 kv-resident 32768 mmap-experts DFlash2 conf-auto HC_Q8=1 TAIL_SKIP=0)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 45 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 123.1 (42.8-134.1) n=3 | 0.79 (0.78-0.91) n=3 | 1.30 (1.29-1.74) n=3 | 3.4 (2.7-9.8) n=3 | 0/3 |
| prompt-4k | 1331.6 (406.5-1455.4) n=3 | 138.4 (116.5-167.1) n=3 | 3.09 (2.83-10.08) n=3 | 6.04 (4.35-13.04) n=3 | 6.8 (5.9-14.5) n=3 | 3/3 |
| prompt-16k | 3663.1 (2649.3-4263.9) n=3 | 155.5 (155.4-161.9) n=3 | 4.49 (3.86-6.20) n=3 | 7.35 (6.96-7.75) n=2 | 7.8 (7.1-9.4) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 0 | 0 | 0 | 0 | 0 |

Sanity checks: **11 of 12**.

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
| T2 | fail | expected one structured tool call, got 0 |
| T3 | pass | the expected tool call |
| T4 | pass | the expected tool call |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
