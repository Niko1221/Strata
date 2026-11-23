# Strata standard benchmark: NVIDIA GeForce RTX 3060

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261123T075641Z-6e8d71cd`, started 2026-11-23T07:56:41Z. **Status: nonconforming.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

This run is not a complete standard result:

- the Strata commit is unknown (not a git checkout and no COMMIT file)

## Hardware and software

- GPU: NVIDIA GeForce RTX 3060 (12.0 GiB dedicated)
- CPU: not measured; RAM: 53.6 GiB
- Storage of the model files: nvme KINGSTON OM8TAP41024K1-A00 (nvme)
- OS: Ubuntu 26.04.1 LTS 7.0.0-43-generic; drivers: gpu-driver 580.178.04; runtime: cuda 13.0
- Strata commit: `not measured`; engine 0.1.66; release build
- Background work: not measured

## Model and configuration

- Model: swift-1.5-iq3_xxs; quantization Swift-1.5 GSQ-RCO IQ3_XXS (ukisai); revision not measured; source revision not measured
- Files: `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `3bddaa667c750f63...`; `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `b0b15f782af71eb4...`
- Track: **custom** (0.2.20 / engine 0.1.66 compiled locally (CUDA 13.1, sm_86, aarch64); 131072 ctx; auto PCIe share over OCuLink x4)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 6 min 0 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 29.3 (27.3-29.6) n=3 | 2.72 (1.95-4.84) n=3 | 6.72 (5.06-7.65) n=3 | 17.4 (14.9-18.2) n=3 | 0/3 |
| prompt-4k | 550.0 (502.6-551.4) n=3 | 29.6 (29.3-30.2) n=3 | 7.51 (7.48-8.22) n=3 | 16.24 (12.81-21.80) n=3 | 24.9 (24.8-25.1) n=3 | 3/3 |
| prompt-16k | 825.9 (765.5-879.3) n=3 | 28.7 (27.5-29.0) n=3 | 19.92 (18.72-21.48) n=3 | 28.18 (25.19-31.16) n=2 | 37.7 (37.3-39.1) n=3 | 3/3 |

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
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 7 request(s) before this run.
