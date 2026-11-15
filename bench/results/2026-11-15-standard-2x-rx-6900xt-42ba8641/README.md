# Strata standard benchmark: AMD Radeon RX 6900 XT

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261115T014826Z-42ba8641`, started 2026-11-15T01:48:26Z. **Status: complete.**

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
- Track: **custom** (two RX 6900 XT, layer split 27 + on-demand DP prompt read; this box's production build strata-0211fz-72bb0be6: v0.2.11 + local patch set (on-demand dual data-parallel prompt read from #5002 + follow-ups, int8 hc copies for the persistent hc read (#5475), decode fusions, FP16 hc/attn_qkv without widening, dense GEMV/expert tables for gfx1030; plus PROMPT_ATTN_RDNA2 #3381, PREFILL_CPU_SHARE_BATCH #3382, SELECT_SIMT=2 #3073, PW_WITH_BATCH #3095); production env, --spec 3 --spec-min-p 0.6 --pipeline-windows 2 --prefill-pipe 1600 --mtp-q4 head, kv int8, conversation cache 0, vision off, parallel 1; client tree v0.2.13; Ryzen 5 5600X, 128 GB DDR4-3200, RX 6900 XT 16 GB at 245 W and -81 mV, PCIe 4.0 x8 per card, Ubuntu 26.04 kernel 7.0.0-43, ROCm 10.0)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 57 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 91.3 (81.4-99.0) n=3 | 0.47 (0.45-0.47) n=3 | 1.29 (1.16-2.12) n=3 | 4.7 (3.8-4.8) n=3 | 0/3 |
| prompt-4k | 1275.5 (1274.2-1280.0) n=3 | 92.4 (81.4-93.8) n=3 | 3.23 (3.22-3.23) n=3 | 6.20 (5.90-9.11) n=3 | 8.7 (8.7-9.5) n=3 | 2/3 |
| prompt-16k | 2176.5 (2174.4-2189.9) n=3 | 87.9 (80.9-96.8) n=3 | 7.56 (7.51-7.56) n=3 | 11.35 (10.72-12.82) n=3 | 13.4 (12.8-13.8) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 4 | 5 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 4 | 8 | 0 | 0 |

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
| 1 | 24.123898 | 1998 | 82.822436 | 0 |
| 2 | 22.035293 | 1896 | 86.043784 | 0 |
| 3 | 23.88146 | 1949 | 81.611426 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
