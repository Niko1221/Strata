# Strata standard benchmark: NVIDIA GeForce RTX 5060 Ti

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261105T212205Z-71916a97`, started 2026-11-05T21:22:05Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 5060 Ti (15.9 GiB dedicated)
- CPU: AMD Ryzen 7 3700X 8-Core Processor; RAM: 63.9 GiB
- Storage of the model files: ssd Samsung SSD 970 EVO Plus 1TB (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 617.42; runtime: cuda 13.4
- Strata commit: `1cddbff5a96e73700d4caa6731a2b1accd1622f4`; engine 0.1.57; source build
- Background work: Strata desktop app running; this agent chat session served by the same engine on the same GPU

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-iq3_xxs; quantization IQ3_XXS (Unsloth UD, Qwen3.8-Flash-Next-UD-IQ3_XXS); revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf` `9d5644b371ffac92...`; `Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00004.gguf` `aacaff3bdb9032c2...`; `Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00004.gguf` `ef8f14e790ab471b...`; `Qwen3.8-Flash-Next-UD-IQ3_XXS-00004-of-00004.gguf` `5de8b5dbf89409ac...`
- Track: **default** (Strata 0.2.9 desktop app, engine 0.1.57 official Windows CUDA13 package, setup defaults; UI resource profile performance, context 262144, KV int8, vision on; drafting dflash2 (q4_0 drafter), spec 6, draft-conf 0.8)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 7 min 48 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 24.4 (22.8-27.2) n=3 | 2.42 (2.02-2.58) n=3 | 6.52 (4.48-6.59) n=3 | 16.2 (12.1-18.8) n=3 | 0/3 |
| prompt-4k | 593.4 (593.2-594.5) n=3 | 23.5 (23.2-25.8) n=3 | 6.97 (6.94-60.62) n=3 | 24.88 (22.47-74.12) n=3 | 28.9 (26.8-82.4) n=3 | 3/3 |
| prompt-16k | 1113.5 (1112.3-1115.6) n=3 | 25.1 (24.1-27.5) n=3 | 14.80 (14.75-14.80) n=3 | 27.46 (27.05-27.88) n=2 | 35.1 (33.4-36.0) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 4 | 8 | 0 | 0 |

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
| 1 | 142.703 | 1824 | 12.781792 | 0 |
| 2 | 98.688 | 2048 | 20.75227 | 0 |
| 3 | 99.484 | 1982 | 19.922802 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r2-c1: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c4: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 25 request(s) before this run.
