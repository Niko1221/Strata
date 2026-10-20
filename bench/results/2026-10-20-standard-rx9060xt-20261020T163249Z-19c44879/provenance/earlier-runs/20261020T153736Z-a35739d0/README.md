# Strata standard benchmark: AMD Radeon

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261020T153736Z-a35739d0`, started 2026-10-20T15:37:36Z. **Status: nonconforming.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

This run is not a complete standard result:

- the Strata commit is unknown (not a git checkout and no COMMIT file)

## Hardware and software

- GPU: AMD Radeon
- CPU: 12th Gen Intel(R) Core(TM) i7-12700H; RAM: 62.6 GiB
- Storage of the model files: not measured
- OS: Ubuntu 24.04.4 LTS 6.18.18.c1126-trim; drivers: not measured; runtime: not measured
- Strata commit: `not measured`; engine 0.1.46.1; release build
- Background work: household voice assistant (open-xiaoai) on the same engine: prompt-prefix warm-ups about every 20 min; Qwen3-ASR on the Intel iGPU, March7 TTS on CPU (idle unless spoken to)

## Model and configuration

- Model: qwen3.8-flash-next-iq2_xs; quantization IQ2_XS; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` `92cee27ae5bbadcd...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (Strata 0.2.1 (upstream 9ae93f7b) + 9 local commits (branch amd/nas-sync-021 @ 1130de3c: Q4_0-KV WMMA prompt attention STRATA_HIP_WMMA_Q4, conversation disk tier, elastic CPU image encoder, /v1/models fields, retuned gfx1200 hipBLASLt table, NAS Docker build); Docker on fnOS (Debian 12), ROCm 7.2.3 userspace; run config: IQ2_XS native pack, --kv q4_0 --max-context 524288 --batch 2 --batch-mtp --prefill 8192 --spec 4 --spec-min-p 0.5 --resident-experts --pool-workers 13 --vram-reserve-mib 896 --vision (CPU encoder, elastic); env STRATA_ROUTE_TAIL_SKIP=7 STRATA_MTP_SIDE_STREAM=1 STRATA_PF_FUSED=1 STRATA_HIP_WMMA_Q4=1 STRATA_PREFILL_RING=384 STRATA_SELECT_WMMA=1 STRATA_HIP_QUEUE_PRIME=0 GPU_MAX_HW_QUEUES=2 STRATA_POOL_SPIN_US=0 STRATA_HIPBLASLT_TUNING=<retuned 162-row table>)
- Context 524288 tokens; server capacity 2 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 31 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s |
| --- | --- | --- | --- | --- | --- |
| short | 57.1 (20.0-67.5) n=3 | 47.5 (42.9-48.5) n=3 | 1.39 (1.18-3.88) n=3 | 5.75 (4.83-6.47) n=3 | 10.6 (8.2-10.8) n=3 |
| prompt-4k | 1220.3 (1214.0-1227.0) n=3 | 43.8 (43.3-50.9) n=3 | 3.39 (3.37-3.42) n=3 | 11.98 (9.40-14.45) n=3 | 15.0 (13.4-15.2) n=3 |
| prompt-16k | 1501.3 (1409.4-1512.9) n=3 | 42.1 (40.1-42.9) n=3 | 10.96 (10.88-11.67) n=3 | 22.46 (22.21-22.72) n=2 | 23.6 (23.1-23.6) n=3 |

Cells are median (min-max) n=samples.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 2 | 10 | 0 | 0 |

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
| R4 | fail | the JSON value differs from the expected value |
| T1 | pass | the expected tool call |
| T2 | pass | the expected tool call |
| T3 | pass | the expected tool call |
| T4 | pass | the expected tool call |

Concurrency rounds (four requests released together):

| Round | Wall s | Output tokens | Aggregate tok/s | Failed |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 42.858745 | 2048 | 47.78488 | 0 |
| 2 | 37.970429 | 1816 | 47.826692 | 0 |
| 3 | 42.690816 | 2048 | 47.972847 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c1: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r1-c2: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r1-c4: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r2-c2: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c1: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c2: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c3: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c4: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The runner could not tell whether the server was freshly loaded.
