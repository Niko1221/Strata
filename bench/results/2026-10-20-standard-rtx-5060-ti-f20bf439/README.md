# Strata standard benchmark: NVIDIA GeForce RTX 5060 Ti

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261020T111949Z-f20bf439`, started 2026-10-20T11:19:49Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 5060 Ti (15.9 GiB dedicated)
- CPU: AMD Ryzen 9 8945HS w/ Radeon 780M Graphics; RAM: 54.9 GiB
- Storage of the model files: hdd Virtual Disk (unknown)
- OS: Ubuntu 24.04.4 LTS 6.18.33.2-microsoft-standard-WSL2; drivers: gpu-driver 616.56; runtime: not measured
- Strata commit: `9ae93f7b1fe989d04a8e404d32cddaeeff10505f`; engine 0.1.46.1; source build
- Background work: WSL2 on Windows 11

## Model and configuration

- Model: swift-1.5-iq3_xxs; quantization IQ3_XXS; revision not measured; source revision not measured
- Files: `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `3bddaa667c750f63...`; `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `b0b15f782af71eb4...`
- Track: **default** (0.2.1 (engine 0.1.46.1), WSL2 + systemd, swift-1.5-iq3_xxs, --pcie-frac 0.20, STRATA_PREFILL_CPU_SHARE=0)
- Context 65536 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 4 min 2 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s |
| --- | --- | --- | --- | --- | --- |
| short | 42.7 (38.9-46.8) n=3 | 46.0 (25.7-51.8) n=3 | 1.83 (1.68-2.02) n=3 | 3.78 (3.28-5.12) n=3 | 8.3 (6.9-15.0) n=3 |
| prompt-4k | 702.7 (657.4-704.5) n=3 | 50.6 (33.7-51.2) n=3 | 5.85 (5.84-6.30) n=3 | 13.98 (11.57-20.20) n=3 | 16.0 (15.8-21.4) n=3 |
| prompt-16k | 1332.8 (1320.6-1347.2) n=3 | 46.8 (44.9-55.1) n=3 | 12.33 (12.20-12.47) n=3 | 18.81 (17.91-19.71) n=2 | 23.1 (21.6-23.8) n=3 |

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
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 42 request(s) before this run.
