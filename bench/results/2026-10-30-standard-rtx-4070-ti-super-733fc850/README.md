# Strata standard benchmark: NVIDIA GeForce RTX 4070 Ti SUPER

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261030T143616Z-733fc850`, started 2026-10-30T14:36:16Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER (16.0 GiB dedicated)
- CPU: AMD Ryzen 9 9900X 12-Core Processor; RAM: 61.6 GiB
- Storage of the model files: ssd WD_BLACK SN850X 2000GB (nvme)
- OS: Windows 10.0.26200; drivers: gpu-driver 616.92; runtime: cuda 13.0, cuda-driver-supports 13.4
- Strata commit: `870aedd9e1ebe1c91f94da93b59af17729f6c5c0`; engine 0.1.54; source build
- Background work: desktop idle, no other GPU work

## Model and configuration

- Model: qwen3.8-flash-next-iq2_xs; quantization IQ2_XS; revision d4af3be5701549435e0770d625f79dd2def1358f; source revision de4b8e4d43b917e7706784d8bb445c9af86a3540
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` `92cee27ae5bbadcd...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (0.2.8 source install, engine 0.1.54 (prebuilt cuda13), abliterated IQ2_XS (alesha-pro GSQ-RCO), refusal control vector layers 4-44, 128K context, 1 slot)
- Context 128000 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 53 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 90.6 (86.5-102.0) n=3 | 0.54 (0.41-0.58) n=3 | 2.74 (1.08-4.40) n=2 | 6.2 (3.8-6.4) n=3 | 2/3 |
| prompt-4k | 1799.3 (1563.6-1871.0) n=3 | 82.3 (80.4-101.1) n=3 | 2.32 (2.23-2.64) n=3 | 4.51 (4.51-4.51) n=1 | 8.5 (7.3-9.0) n=3 | 3/3 |
| prompt-16k | 2422.1 (2408.9-2431.2) n=3 | 94.1 (83.4-94.4) n=3 | 6.81 (6.78-6.85) n=3 | 10.42 (9.70-11.14) n=2 | 12.2 (12.2-13.0) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 1 | 8 | 0 | 0 |
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
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-short-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
