# Strata standard benchmark: AMD Radeon(TM) 8060S Graphics

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261103T120237Z-8ca6387a`, started 2026-11-03T12:02:37Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon(TM) 8060S Graphics (96.0 GiB dedicated)
- CPU: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S; RAM: 31.6 GiB
- Storage of the model files: ssd Sandisk PC SN5100S SDFQTSM-1T00-1002 (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 32.0.22032.6002; runtime: not measured
- Strata commit: `c5ab5d4003161333044f6f16f5dacd40b7be6d9a`; engine 0.1.56; release build
- Background work: local AI service paused; desktop session only

## Model and configuration

- Model: qwen3.8-flash-next-q2_0; quantization Q2_0; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` `69820c02ec7d0b45...`; `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **default** (release 0.2.10 (engine 0.1.56), setup defaults: strata install --family qwen --model Q2_0 --context 131072 --vision no)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 4 min 7 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 48.2 (47.3-49.2) n=3 | 0.84 (0.83-0.88) n=3 | 2.66 (2.05-3.62) n=3 | 11.1 (6.8-11.4) n=3 | 1/3 |
| prompt-4k | 1021.4 (1014.9-1034.6) n=3 | 43.1 (39.8-47.0) n=3 | 4.08 (4.00-4.09) n=3 | 12.73 (9.75-14.25) n=3 | 15.9 (14.9-16.8) n=3 | 3/3 |
| prompt-16k | 1061.2 (1056.7-1062.4) n=3 | 42.2 (39.0-45.1) n=3 | 15.52 (15.48-15.58) n=3 | 23.72 (21.78-26.53) n=3 | 27.6 (26.8-28.7) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 2 | 7 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 0 | 0 | 0 | 0 | 0 |

Sanity checks: **11 of 12**.

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
| T2 | fail | expected one structured tool call, got 0 |
| T3 | pass | the expected tool call |
| T4 | pass | the expected tool call |

## Failures, exclusions and limitations

- No request failed.
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
