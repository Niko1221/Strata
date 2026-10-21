# Strata standard benchmark: NVIDIA GeForce RTX 2080 Ti, NVIDIA GeForce RTX 2080 Ti

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261021T025035Z-09a9bf8e`, started 2026-10-21T02:50:35Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 2080 Ti (22.0 GiB dedicated); NVIDIA GeForce RTX 2080 Ti (22.0 GiB dedicated)
- CPU: Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz; RAM: 94.1 GiB
- Storage of the model files: not measured
- OS: Ubuntu 24.04.5 LTS 6.8.0-147-generic; drivers: gpu-driver 595.99.02; runtime: cuda 13.2
- Strata commit: `9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4`; engine 0.1.46.3; source build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs; quantization GSQ-RCO IQ3_XXS; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision de4b8e4d43b917e7706784d8bb445c9af86a3540
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `219ea929900dfa9e...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **default** (Strata 0.2.1.4 / engine 0.1.46.3 (local build, sm_75, CUDA 12.8); 2x RTX 2080 Ti 22GB, PCIe 3.0 x8; env SM75_PERF+BF16_TC+PA_V2T; layer-split 26; parallel 3; repeat run 2)
- Context 262144 tokens; server capacity 3 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 51 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s |
| --- | --- | --- | --- | --- | --- |
| short | 131.6 (129.7-137.1) n=3 | 85.2 (81.0-88.9) n=3 | 0.61 (0.58-0.62) n=3 | 1.29 (1.23-1.84) n=3 | 5.0 (4.4-5.8) n=3 |
| prompt-4k | 920.9 (915.5-924.9) n=3 | 76.4 (74.1-77.1) n=3 | 4.47 (4.45-4.49) n=3 | 9.35 (8.52-10.82) n=3 | 11.2 (11.1-11.4) n=3 |
| prompt-16k | 1590.4 (1590.2-1603.4) n=3 | 68.0 (67.9-72.2) n=3 | 10.34 (10.25-10.34) n=3 | not measured | 17.8 (17.4-17.9) n=3 |

Cells are median (min-max) n=samples.

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
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The runner could not tell whether the server was freshly loaded.
