# Strata standard benchmark: AMD Radeon RX 6900 XT

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261115T013907Z-e407aa13`, started 2026-11-15T01:39:07Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon RX 6900 XT
- CPU: AMD Ryzen 5 5600X 6-Core Processor; RAM: 121.5 GiB
- Storage of the model files: nvme PNY CS2140 2TB SSD (nvme)
- OS: Ubuntu 26.04.1 LTS 7.0.0-43-generic; drivers: gpu-driver 7.0.0-43-generic; runtime: rocm 10.0.0
- Strata commit: `d2f096475817d4666c364c820b0ca830f751cfc5`; engine 0.1.57; source build
- Background work: not measured

## Model and configuration

- Model: strata-split-256k; quantization IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision 72bb0be6
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (one RX 6900 XT (gpu 1 of 2, the other card idle); this box's production build strata-0211fz-72bb0be6: v0.2.11 + local patch set (on-demand dual data-parallel prompt read from #5002 + follow-ups, int8 hc copies for the persistent hc read (#5475), decode fusions, FP16 hc/attn_qkv without widening, dense GEMV/expert tables for gfx1030; plus PROMPT_ATTN_RDNA2 #3381, PREFILL_CPU_SHARE_BATCH #3382, SELECT_SIMT=2 #3073, PW_WITH_BATCH #3095); production env, --spec 3 --spec-min-p 0.6 --pipeline-windows 2 --prefill-pipe 1600 --mtp-q4 head, kv int8, conversation cache 0, vision off, parallel 1; client tree v0.2.13; Ryzen 5 5600X, 128 GB DDR4-3200, RX 6900 XT 16 GB at 245 W and -81 mV, PCIe 4.0 x8 per card, Ubuntu 26.04 kernel 7.0.0-43, ROCm 10.0)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 5 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 51.6 (51.2-57.7) n=3 | 0.65 (0.58-0.67) n=3 | 2.22 (1.64-2.55) n=3 | 6.5 (5.6-7.7) n=3 | 0/3 |
| prompt-4k | 1074.0 (1064.8-1074.7) n=3 | 47.9 (47.6-49.9) n=3 | 3.86 (3.84-3.88) n=3 | 13.70 (13.17-14.23) n=2 | 14.5 (14.1-14.6) n=3 | 3/3 |
| prompt-16k | 1473.0 (1472.8-1473.0) n=3 | 52.7 (50.0-55.7) n=3 | 11.16 (11.16-11.17) n=3 | 15.60 (15.60-15.60) n=1 | 20.9 (20.3-21.4) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 5 | 7 | 0 | 0 |

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
| 1 | 42.315646 | 1960 | 46.318565 | 0 |
| 2 | 39.344243 | 1912 | 48.59669 | 0 |
| 3 | 35.205942 | 1785 | 50.701668 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c2: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r2-c4: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
