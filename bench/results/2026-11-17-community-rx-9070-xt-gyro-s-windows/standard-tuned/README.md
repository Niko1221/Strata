# Strata standard benchmark: AMD Radeon RX 9070 XT

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261117T163750Z-e933b70d`, started 2026-11-17T16:37:50Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon RX 9070 XT (15.9 GiB dedicated)
- CPU: AMD Ryzen 5 7600X3D 6-Core Processor; RAM: 31.4 GiB
- Storage of the model files: ssd SKHynix_HFS512GDE9X084N (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 32.0.32015.16001; runtime: not measured
- Strata commit: `3af562d541ec76c8da63d568248a919c8a54906b`; engine 0.1.60; release build
- Background work: Claude desktop app; Google Chrome

## Model and configuration

- Model: qwen3.8-flash-next-gyro-gyro-s; quantization not measured; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf` `c6ea1b08ca312f99...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (release 0.2.14 (engine 0.1.60 official Windows HIP rx9070 build); settings tuned by hand from the app's defaults, see the community report)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 21 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 83.7 (76.9-84.4) n=3 | 0.77 (0.69-0.89) n=3 | 1.81 (1.56-2.80) n=3 | 4.9 (3.9-4.9) n=3 | 0/3 |
| prompt-4k | 1276.7 (1175.6-1278.5) n=3 | 81.1 (79.5-85.9) n=3 | 3.23 (3.23-3.52) n=3 | 6.47 (4.62-6.69) n=3 | 9.7 (8.2-9.8) n=3 | 2/3 |
| prompt-16k | 1608.3 (1602.5-1612.1) n=3 | 81.2 (78.6-87.9) n=3 | 10.25 (10.19-10.28) n=3 | 13.67 (12.75-14.95) n=3 | 16.0 (15.0-16.8) n=3 | 2/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 5 | 4 | 0 | 0 |
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
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
