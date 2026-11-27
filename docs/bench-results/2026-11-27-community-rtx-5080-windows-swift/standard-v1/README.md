# Strata standard benchmark: NVIDIA GeForce RTX 5080

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261127T190659Z-7d89f7f9`, started 2026-11-27T19:06:59Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 5080 (15.9 GiB dedicated)
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor; RAM: 61.4 GiB
- Storage of the model files: ssd Samsung SSD 9100 PRO 2TB (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 621.14; runtime: cuda 13.5
- Strata commit: `64524fb9fc82222ddce554c0643b64f056ec8366`; engine 0.1.62; release build
- Background work: not measured

## Model and configuration

- Model: swift-1.5-iq3_xxs; quantization IQ3_XXS; revision not measured; source revision not measured
- Files: `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `3bddaa667c750f63...`; `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `b0b15f782af71eb4...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (preset Swift 1.5 256K: kv q4_0, kv-resident 32768, conversation-cache-mib 4096, vram-reserve-mib 1536, vision cpu; 4x20k-token warm-up with unrelated text before the run)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 15 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 122.9 (121.5-136.8) n=3 | 0.39 (0.36-0.42) n=3 | 2.44 (0.80-3.28) n=3 | 4.1 (2.8-4.6) n=3 | 1/3 |
| prompt-4k | 3537.0 (3486.7-3567.5) n=3 | 131.6 (112.7-144.1) n=3 | 1.19 (1.19-1.20) n=3 | 3.27 (2.69-3.84) n=2 | 5.1 (4.7-5.7) n=3 | 3/3 |
| prompt-16k | 4137.0 (4118.4-4144.5) n=3 | 125.4 (124.2-132.1) n=3 | 4.00 (3.98-4.01) n=3 | 6.33 (5.99-6.94) n=3 | 8.1 (7.9-8.1) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 5 request(s) before this run.
