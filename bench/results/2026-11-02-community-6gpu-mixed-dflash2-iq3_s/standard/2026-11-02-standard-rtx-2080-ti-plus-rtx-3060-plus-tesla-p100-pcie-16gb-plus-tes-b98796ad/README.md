# Strata standard benchmark: NVIDIA GeForce RTX 2080 Ti, NVIDIA GeForce RTX 3060, Tesla P100-PCIE-16GB, Tesla P100-PCIE-16GB, Tesla P100-PCIE-16GB, NVIDIA GeForce RTX 3060

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261102T082431Z-b98796ad`, started 2026-11-02T08:24:31Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 2080 Ti (11.0 GiB dedicated); NVIDIA GeForce RTX 3060 (12.0 GiB dedicated); Tesla P100-PCIE-16GB (16.0 GiB dedicated); Tesla P100-PCIE-16GB (16.0 GiB dedicated); Tesla P100-PCIE-16GB (16.0 GiB dedicated); NVIDIA GeForce RTX 3060 (12.0 GiB dedicated)
- CPU: Intel(R) Xeon(R) W-2150B CPU @ 3.00GHz; RAM: 188.4 GiB
- Storage of the model files: not measured
- OS: <user> 24.04.5 LTS 6.8.0-147-generic; drivers: gpu-driver 580.178.04; runtime: cuda 13.0
- Strata commit: `870aedd9e1ebe1c91f94da93b59af17729f6c5c0`; engine 0.1.54; release build
- Background work: none

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (branch dflash2, DRAFTING=mtp, parallel 2, 16 GB conversation cache)
- Context 178176 tokens; server capacity 2 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 4 min 50 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 50.5 (48.3-52.9) n=3 | 1.82 (1.62-1.83) n=3 | 2.96 (2.77-3.68) n=3 | 10.9 (10.2-11.7) n=3 | 1/3 |
| prompt-4k | 428.0 (427.3-428.3) n=3 | 43.8 (41.2-44.1) n=3 | 9.60 (9.59-9.61) n=3 | 21.00 (21.00-21.00) n=1 | 21.3 (21.2-22.0) n=3 | 3/3 |
| prompt-16k | 857.9 (855.0-858.9) n=3 | 45.5 (42.2-48.7) n=3 | 19.15 (19.12-19.21) n=3 | 25.49 (25.25-25.73) n=2 | 30.4 (29.6-31.3) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
