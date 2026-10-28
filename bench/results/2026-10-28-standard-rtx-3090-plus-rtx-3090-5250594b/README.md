# Strata standard benchmark: NVIDIA GeForce RTX 3090, NVIDIA GeForce RTX 3090

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261028T134720Z-5250594b`, started 2026-10-28T13:47:20Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3090 (24.0 GiB dedicated); NVIDIA GeForce RTX 3090 (24.0 GiB dedicated)
- CPU: Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz; RAM: 247.5 GiB
- Storage of the model files: not measured
- OS: Ubuntu 26.10 7.3.0-9-generic; drivers: gpu-driver 580.178.04; runtime: cuda 13.0
- Strata commit: `a42cc3d7df70d1940de82cd0882f03e1f5dfc741`; engine 0.1.52; release build
- Background work: headless server; production server for a coding agent, idle during the run

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `t630-expert-profile.bin` `02536638857760cd...`
- Track: **custom** (Strata 0.2.7 official Linux release, engine 0.1.52. Tuned: both cards, layer_split 25 + --trim-stage-weights, --prefill 16384, --spec 4 --spec-min-p 0.7, --vram-reserve-mib 1400, --adapt-async 1, conversation cache 40 GiB/16 slots, STRATA_PREFILL_CPU_SHARE=auto with MAX 4096, numactl --interleave=all, 250 W per card, model GGUFs in /dev/shm. See bench/results/2026-10-28-community-t630-2x3090-pcie3/README.md.)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first <key>mpty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 5 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 128.8 (126.7-130.4) n=3 | 0.53 (0.52-0.55) n=3 | 1.33 (1.00-1.66) n=3 | 3.2 (2.0-3.6) n=3 | 0/3 |
| prompt-4k | 1126.7 (1099.9-1126.9) n=3 | 114.2 (110.4-118.5) n=3 | 3.66 (3.66-3.74) n=3 | 6.04 (6.04-6.04) n=1 | 8.2 (8.0-8.3) n=3 | 3/3 |
| prompt-16k | 1610.4 (1538.5-1611.8) n=3 | 115.4 (109.2-135.9) n=3 | 10.24 (10.20-10.68) n=3 | 14.05 (11.53-14.70) n=3 | 14.7 (14.0-15.4) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 58 request(s) before this run.
