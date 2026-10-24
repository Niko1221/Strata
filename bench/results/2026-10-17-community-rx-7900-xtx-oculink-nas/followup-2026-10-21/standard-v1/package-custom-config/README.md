# Strata standard benchmark: AMD Radeon

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261020T230416Z-abc7ad85`, started 2026-10-20T23:04:16Z. **Status: nonconforming.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

This run is not a complete standard result:

- the Strata commit is unknown (not a git checkout and no COMMIT file)

## Hardware and software

- GPU: AMD Radeon
- CPU: Intel(R) Core(TM) Ultra 7 255H; RAM: 62.3 GiB
- Storage of the model files: not measured
- OS: Ubuntu 24.04.5 LTS 6.18.15; drivers: not measured; runtime: not measured
- Strata commit: `not measured`; engine 0.1.46; release build
- Background work: llama-swap proxy (idle)

## Model and configuration

- Model: b021; quantization IQ3_XXS; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `219ea929900dfa9e...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (prebuilt hip-rx7900 0.1.46 + control 0.1.46.2; served by serve/server.py with a hand-written run config: --pool-workers 9 --vram-reserve-mib 832 --spec 4 --kv int8 --max-context 262144, hipBLASLt table gfx1100)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 25 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s |
| --- | --- | --- | --- | --- | --- |
| short | 51.7 (47.4-67.0) n=3 | 69.0 (57.9-70.7) n=3 | 1.51 (1.17-1.65) n=3 | 4.17 (2.29-9.47) n=3 | 8.6 (7.2-10.5) n=3 |
| prompt-4k | 743.9 (735.3-752.9) n=3 | 63.5 (60.5-64.1) n=3 | 5.53 (5.46-5.59) n=3 | 11.17 (11.15-13.12) n=3 | 13.6 (13.4-14.0) n=3 |
| prompt-16k | 1236.7 (1230.9-1238.2) n=3 | 62.5 (61.0-73.1) n=3 | 13.28 (13.27-13.35) n=3 | 19.39 (16.14-20.16) n=3 | 21.5 (20.2-21.7) n=3 |

Cells are median (min-max) n=samples.

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
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 1 request(s) before this run.
