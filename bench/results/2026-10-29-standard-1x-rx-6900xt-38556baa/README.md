# Strata standard benchmark: AMD Radeon RX 6900 XT

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261029T101606Z-38556baa`, started 2026-10-29T10:16:06Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon RX 6900 XT
- CPU: AMD Ryzen 5 5600X 6-Core Processor; RAM: 121.4 GiB
- Storage of the model files: nvme PNY CS2140 2TB SSD (nvme)
- OS: Ubuntu 26.04.1 LTS 7.0.0-38-generic; drivers: gpu-driver 7.0.0-38-generic; runtime: rocm 10.0.0
- Strata commit: `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d`; engine 0.1.52; source build
- Background work: not measured

## Model and configuration

- Model: strata-split-256k; quantization IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision 9ca7fd21
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (one card (GPU 1), 131072 context, P-single plus STRATA_KVA=1 (0.2.7 single-card approximate late-layer KV on cold prompts >= 4096): this box's production build strata-027-9ca7fd21: v0.2.7 merged into the local patch set (PROMPT_ATTN_RDNA2 #3381, PREFILL_CPU_SHARE_BATCH #3382, SELECT_SIMT=2 #3073, PW_WITH_BATCH #3095), production env (the upstream switches above plus STRATA_SELECT_SIMT=2, STRATA_PREFILL_CPU_SHARE_BATCH=1, STRATA_PROMPT_ATTN_RDNA2=1, rocBLAS 5.6.0 table), --spec 3 --spec-min-p 0.7 --pipeline-windows 2 --prefill-pipe 3072, parallel 1 (no batch slots), kv int8, conversation cache 0, vision off, run-config sampling temperature 0 / presence_penalty 0; cards at 245 W and -81 mV, Ryzen 5 5600X, 128 GB DDR4-3200, PCIe 4.0 x8 per card, Ubuntu 26.04 kernel 7.0.0-38, ROCm 10.0)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 25 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 50.3 (49.4-50.9) n=3 | 0.84 (0.79-0.84) n=3 | 2.15 (2.09-2.58) n=3 | 8.0 (6.3-9.0) n=3 | 0/3 |
| prompt-4k | 1026.1 (1015.5-1029.1) n=3 | 43.6 (42.0-46.8) n=3 | 4.03 (4.02-4.06) n=3 | 11.36 (11.36-11.36) n=1 | 15.8 (14.9-16.2) n=3 | 3/3 |
| prompt-16k | 1344.5 (1343.8-1349.1) n=3 | 45.5 (43.8-48.0) n=3 | 12.23 (12.20-12.24) n=3 | 17.13 (17.13-17.13) n=1 | 23.5 (22.9-23.8) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 1 | 11 | 0 | 0 |

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
| 1 | 43.749674 | 1908 | 43.611754 | 0 |
| 2 | 47.204451 | 2048 | 43.385739 | 0 |
| 3 | 48.302777 | 2048 | 42.399219 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c3: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r2-c3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
