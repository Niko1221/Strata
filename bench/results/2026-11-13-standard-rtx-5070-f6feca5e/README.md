# Strata standard benchmark: NVIDIA GeForce RTX 5070

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261113T153217Z-f6feca5e`, started 2026-11-13T15:32:17Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 5070 (11.9 GiB dedicated)
- CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics; RAM: 30.8 GiB
- Storage of the model files: ssd CT1000E100SSD8 (nvme)
- OS: Windows 10.0.26200; drivers: gpu-driver 617.42; runtime: cuda 13.4
- Strata commit: `d2f096475817d4666c364c820b0ca830f751cfc5`; engine 0.1.59; release build
- Background work: Windows 11 desktop driven by the integrated Radeon 780M (the RTX 5070 reports no display); Claude desktop app open and idle; LM Studio background service running with no model loaded; a 5-second nvidia-smi power logger running

## Model and configuration

- Model: qwen3.8-flash-next-gyro-gyro-s; quantization Gyro-S TQ1_0 (agentionai rotor quant); revision 1ada08422e404aca51ab9df665f089f34c419c6b; source revision not measured
- Files: `Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf` `c6ea1b08ca312f99...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (Drafter comparison, arm mtp-a. Strata 0.2.13 Terminal (Control 0.1.59), engine 0.1.59 windows-x86_64-cuda-cuda13. Config from strata install -- --family gyro --model Gyro-S --context 65536 --vision no --drafter mtp, plus env STRATA_RESIDENT_HEADROOM_GIB=12 and drafting: mtp (the drafter chosen by name). Windows page file 32-48 GB.)
- Context 65536 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 24 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 50.9 (44.5-51.9) n=3 | 1.26 (1.03-1.30) n=3 | 3.45 (3.00-4.20) n=3 | 8.2 (7.8-9.1) n=3 | 0/3 |
| prompt-4k | 916.1 (243.3-928.7) n=3 | 54.9 (44.3-58.5) n=3 | 4.50 (4.45-16.88) n=3 | 7.41 (6.16-27.61) n=3 | 13.8 (10.8-28.4) n=3 | 2/3 |
| prompt-16k | 1464.1 (1032.4-1468.8) n=3 | 53.5 (50.2-60.1) n=3 | 11.25 (11.19-15.92) n=3 | 16.38 (13.50-19.28) n=3 | 20.8 (18.0-23.1) n=3 | 1/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 6 | 3 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 10 | 2 | 0 | 0 |

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

Concurrency rounds (four requests released together):

| Round | Wall s | Output tokens | Aggregate tok/s | Failed |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 34.672 | 1565 | 45.137287 | 0 |
| 2 | 37.157 | 1669 | 44.917512 | 0 |
| 3 | 35.516 | 1745 | 49.132785 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r2-c4: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
