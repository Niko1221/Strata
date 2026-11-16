# Strata standard benchmark: NVIDIA GeForce RTX 5090

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261116T175425Z-296dd721`, started 2026-11-16T17:54:25Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 5090 (31.8 GiB dedicated)
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor; RAM: 189.7 GiB
- Storage of the model files: ssd Samsung SSD 990 EVO Plus 4TB (nvme)
- OS: Windows 10.0.26200; drivers: gpu-driver 621.24; runtime: cuda 13.5
- Strata commit: `3af562d541ec76c8da63d568248a919c8a54906b`; engine 0.1.60; release build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-q8_0; quantization Q8_0; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf` `2dabcbb53ca537a7...`; `Qwen3.8-Flash-Next-Q8_0-00002-of-00006.gguf` `494ca4ed3dbf97bc...`; `Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf` `34efd79a80a1ce54...`; `Qwen3.8-Flash-Next-Q8_0-00004-of-00006.gguf` `bfa634025fabbd26...`; `Qwen3.8-Flash-Next-Q8_0-00005-of-00006.gguf` `232a8f14cc0fa426...`; `Qwen3.8-Flash-Next-Q8_0-00006-of-00006.gguf` `538a93bca9180649...`; `draft_vocab.bin` `b1e1d3a7a9e4bf86...`; `expert-profile.bin` `8f59b4aa8873209d...`; `qwen3.8-flash-next-dflash2.q4_0.pack` `cda40d62d0eb8f37...`
- Track: **default** (release 0.2.14.1, Unsloth Q8_0, 128K, stock)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 1 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 54.9 (52.6-61.2) n=3 | 0.75 (0.61-0.78) n=3 | 2.16 (2.11-2.44) n=3 | 5.8 (5.5-6.2) n=3 | 0/3 |
| prompt-4k | 1863.0 (1860.9-1870.7) n=3 | 57.4 (57.2-59.4) n=3 | 2.23 (2.23-2.23) n=3 | 6.51 (6.31-6.70) n=2 | 11.1 (10.8-11.2) n=3 | 3/3 |
| prompt-16k | 6520.1 (6442.4-6596.2) n=3 | 57.8 (55.9-58.0) n=3 | 2.59 (2.55-2.59) n=3 | 10.14 (9.91-10.38) n=2 | 11.4 (11.4-11.7) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
