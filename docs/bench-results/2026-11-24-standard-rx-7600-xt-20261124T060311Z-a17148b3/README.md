# Strata standard benchmark: AMD Radeon RX 7600 XT

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261124T060311Z-a17148b3`, started 2026-11-24T06:03:11Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon RX 7600 XT (16.0 GiB dedicated)
- CPU: AMD Ryzen 7 5700X 8-Core Processor; RAM: 63.9 GiB
- Storage of the model files: hdd WDC WD80EFPX-68C4ZN0 (sata)
- OS: Windows 10.0.19044; drivers: gpu-driver 32.0.32029.2010; runtime: not measured
- Strata commit: `2e87c08db18aa4577c22fd623d6ba53252719d63`; engine 0.1.66; release build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-q2_0; quantization Q2_0; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` `69820c02ec7d0b45...`; `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `draft_vocab_cyrillic.bin` `25d7fd1670a2e0ec...`; `expert-profile.bin` `8f59b4aa8873209d...`; `qwen3.8-flash-next-dflash2.q4_0.pack` `cda40d62d0eb8f37...`
- Track: **custom** (Strata 0.2.20 HIP, saved profile: 64K context, prefill 256, mmap PLE, no PLE prefetch, 1024 MiB conversation cache; local benchmark bind and log.)
- Context 65536 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 21 min 37 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 9.5 (8.9-9.8) n=3 | 1.42 (1.36-1.62) n=3 | 23.58 (19.36-29.14) n=3 | 54.1 (53.6-58.7) n=3 | 2/3 |
| prompt-4k | 136.4 (127.4-138.9) n=3 | 11.6 (9.2-15.7) n=3 | 30.05 (29.52-32.17) n=3 | 47.52 (43.22-67.53) n=3 | 73.8 (62.6-87.6) n=3 | 2/3 |
| prompt-16k | 140.5 (140.5-142.4) n=3 | 10.3 (9.6-15.6) n=3 | 116.67 (115.14-116.69) n=3 | 148.69 (129.20-168.17) n=2 | 166.1 (147.9-169.8) n=3 | 3/3 |

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
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
