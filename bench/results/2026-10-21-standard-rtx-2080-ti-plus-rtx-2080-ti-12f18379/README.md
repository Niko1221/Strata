# Strata standard benchmark: NVIDIA GeForce RTX 2080 Ti, NVIDIA GeForce RTX 2080 Ti

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261021T021121Z-12f18379`, started 2026-10-21T02:11:21Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 2080 Ti (22.0 GiB dedicated); NVIDIA GeForce RTX 2080 Ti (22.0 GiB dedicated)
- CPU: Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz; RAM: 94.1 GiB
- Storage of the model files: not measured
- OS: Ubuntu 24.04.5 LTS 6.8.0-147-generic; drivers: gpu-driver 595.99.02; runtime: cuda 13.2
- Strata commit: `9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4`; engine 0.1.46.3; source build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s-abl; quantization GSQ-RCO IQ3_S (SC117 abliterated); revision eec4e10a2c29b440a2ae85591fa11569ce752963; source revision de4b8e4d43b917e7706784d8bb445c9af86a3540
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf` `40f369eca301ec6e...`; `Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (Strata 0.2.1.4 / engine 0.1.46.3 (local build, sm_75, CUDA 12.8); 2x RTX 2080 Ti 22GB, PCIe 3.0 x8; env SM75_PERF+BF16_TC+PA_V2T; layer-split 26; parallel 3; ABLITERATED derivative (SC117), not the original weights)
- Context 262144 tokens; server capacity 3 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 54 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s |
| --- | --- | --- | --- | --- | --- |
| short | 86.4 (85.7-95.7) n=3 | 69.8 (68.4-73.4) n=3 | 0.91 (0.83-0.92) n=3 | 2.26 (1.95-2.62) n=3 | 5.2 (5.1-6.4) n=3 |
| prompt-4k | 746.4 (746.4-746.7) n=3 | 71.5 (68.7-77.5) n=3 | 5.51 (5.51-5.52) n=3 | 9.49 (8.39-10.68) n=3 | 12.7 (12.1-13.0) n=3 |
| prompt-16k | 1488.7 (1484.8-1495.2) n=3 | 68.3 (64.0-79.1) n=3 | 11.04 (10.99-11.07) n=3 | 14.61 (12.56-16.65) n=2 | 18.5 (17.5-19.0) n=3 |

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
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The runner could not tell whether the server was freshly loaded.
