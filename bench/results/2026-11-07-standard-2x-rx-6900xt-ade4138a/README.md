# Strata standard benchmark: AMD Radeon RX 6900 XT

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261107T031700Z-ade4138a`, started 2026-11-07T03:17:00Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon RX 6900 XT
- CPU: AMD Ryzen 5 5600X 6-Core Processor; RAM: 121.5 GiB
- Storage of the model files: nvme PNY CS2140 2TB SSD (nvme)
- OS: Ubuntu 26.04.1 LTS 7.0.0-42-generic; drivers: gpu-driver 7.0.0-42-generic; runtime: rocm 10.0.0
- Strata commit: `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d`; engine 0.1.57; source build
- Background work: not measured

## Model and configuration

- Model: strata-split-256k; quantization IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision 4783921b99612c1b1deac3442f7909820e2432e0
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (two RX 6900 XT, layer split 27 + on-demand DP read (>=12K tokens); this box's waitlist-0211 build (E708): v0.2.11 + local patch set (on-demand dual data-parallel prompt read from PR #5002 + follow-ups, MMQ wave-uniform activation reads for RDNA2, FP16 hyper-connection/attn_qkv without widening, int8 copies for the persistent hc read on Q8_0's grid (STRATA_HC_PERSIST_Q8=2), dense GEMV and expert-layout tables for gfx1030, fused SwiGLU+quantize; plus PROMPT_ATTN_RDNA2 #3381, PREFILL_CPU_SHARE_BATCH #3382, SELECT_SIMT=2 #3073, PW_WITH_BATCH #3095), production env, --spec 3 --spec-min-p 0.6 --pipeline-windows 2 --prefill-pipe 1600 --mtp-q4 head, kv int8, conversation cache 0, vision off, parallel 1, run-config sampling temperature 0 / presence_penalty 0; cards at 245 W and -81 mV, Ryzen 5 5600X, 128 GB DDR4-3200, PCIe 4.0 x8 per card, Ubuntu 26.04 kernel 7.0.0-42, ROCm 10.0)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 5 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 84.7 (79.5-91.7) n=3 | 0.49 (0.46-0.51) n=3 | 2.23 (1.54-3.85) n=3 | 6.0 (3.2-6.5) n=3 | 2/3 |
| prompt-4k | 1234.2 (1232.3-1237.1) n=3 | 80.7 (80.0-84.3) n=3 | 3.34 (3.33-3.34) n=3 | 8.15 (7.94-8.36) n=2 | 9.7 (9.4-9.7) n=3 | 3/3 |
| prompt-16k | 2123.8 (2119.4-2132.9) n=3 | 85.5 (80.4-86.9) n=3 | 7.74 (7.71-7.76) n=3 | 11.35 (11.02-11.68) n=2 | 13.7 (13.6-14.1) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 1 | 8 | 0 | 0 |
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
| 1 | 21.933992 | 1853 | 84.480746 | 0 |
| 2 | 24.434904 | 2015 | 82.464003 | 0 |
| 3 | 22.707651 | 1881 | 82.835517 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
