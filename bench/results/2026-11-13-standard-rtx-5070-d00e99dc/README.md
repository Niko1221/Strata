# Strata standard benchmark: NVIDIA GeForce RTX 5070

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261113T083505Z-d00e99dc`, started 2026-11-13T08:35:05Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 5070 (11.9 GiB dedicated)
- CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics; RAM: 30.8 GiB
- Storage of the model files: ssd CT1000E100SSD8 (nvme)
- OS: Windows 10.0.26200; drivers: gpu-driver 617.42; runtime: cuda 13.4
- Strata commit: `d2f096475817d4666c364c820b0ca830f751cfc5`; engine 0.1.59; release build
- Background work: Windows 11 desktop driven by the integrated Radeon 780M (the RTX 5070 reports no display); Claude desktop app open and idle; LM Studio background service running with no model loaded

## Model and configuration

- Model: qwen3.8-flash-next-gyro-gyro-s; quantization Gyro-S TQ1_0 (agentionai rotor quant); revision 1ada08422e404aca51ab9df665f089f34c419c6b; source revision not measured
- Files: `Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf` `c6ea1b08ca312f99...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **default** (Second sample. Strata 0.2.13 Terminal (Control 0.1.59) and engine 0.1.59 windows-x86_64-cuda-cuda13 from the signed feed. Setup: strata install -- --family gyro --model Gyro-S --context 65536 --vision no --drafter mtp. Not setup defaults: 64K context, pictures off, MTP drafter (setup's auto would pick DFlash2).)
- Context 65536 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 34 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 42.8 (41.6-47.2) n=3 | 0.91 (0.89-1.05) n=3 | 3.70 (3.25-4.02) n=3 | 9.0 (8.7-9.1) n=3 | 0/3 |
| prompt-4k | 805.1 (783.2-817.4) n=3 | 42.8 (41.7-43.8) n=3 | 5.16 (5.06-5.28) n=3 | 9.44 (8.89-13.08) n=3 | 15.3 (15.0-17.5) n=3 | 1/3 |
| prompt-16k | 1326.5 (1278.6-1335.7) n=3 | 43.0 (41.1-44.4) n=3 | 12.41 (12.33-12.89) n=3 | 16.77 (16.75-16.80) n=2 | 23.1 (22.4-25.3) n=3 | 1/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 7 | 2 | 0 | 0 |
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
- The server had answered no request before this run (a freshly loaded engine).
