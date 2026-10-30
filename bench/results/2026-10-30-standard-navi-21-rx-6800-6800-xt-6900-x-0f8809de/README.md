# Strata standard benchmark: Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261030T154543Z-0f8809de`, started 2026-10-30T15:45:43Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]
- CPU: 13th Gen Intel(R) Core(TM) i3-13100T; RAM: 78.5 GiB
- Storage of the model files: ssd QEMU HARDDISK (unknown)
- OS: <host> 24.04.4 LTS 6.8.0-124-generic; drivers: gpu-driver 6.8.0-124-generic; runtime: rocm 6.3.0-39
- Strata commit: `36f6663c25e60abdfc626abe48e0e8fa3a9fe762`; engine 0.1.54; release build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (RX 6900 XT gfx1030, released rx6000 package, PVE VM with GPU passthrough; IQ3_S from ModelScope; 256K ctx, vision cpu, kv k8v4, pool-cores 1,3,5)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 54 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 41.7 (40.6-43.3) n=3 | 1.05 (0.99-1.21) n=3 | 3.10 (2.40-3.49) n=3 | 8.8 (8.5-10.8) n=3 | 0/3 |
| prompt-4k | 908.5 (895.6-913.3) n=3 | 39.7 (37.0-43.4) n=3 | 4.55 (4.52-4.62) n=3 | 12.61 (8.96-16.25) n=2 | 17.4 (16.4-18.4) n=3 | 3/3 |
| prompt-16k | 1128.9 (1123.2-1131.1) n=3 | 40.2 (40.0-44.3) n=3 | 14.56 (14.54-14.64) n=3 | 25.03 (24.78-25.29) n=2 | 27.3 (26.1-27.3) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 15 request(s) before this run.
