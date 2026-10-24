# Strata standard benchmark: AMD Radeon

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261020T162259Z-8f6f45b7`, started 2026-10-20T16:22:59Z. **Status: nonconforming.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

This run is not a complete standard result:

- unexpected reuse of the user payload in performance-prompt-16k-r1
- unexpected reuse of the user payload in performance-prompt-16k-r2
- unexpected reuse of the user payload in performance-prompt-16k-r3

## Hardware and software

- GPU: AMD Radeon
- CPU: 12th Gen Intel(R) Core(TM) i7-12700H; RAM: 62.6 GiB
- Storage of the model files: not measured
- OS: Ubuntu 24.04.4 LTS 6.18.18.c1126-trim; drivers: not measured; runtime: not measured
- Strata commit: `690e4f6f6343b9b999b4b8badca4cc47ae262679`; engine 0.1.46.1; release build
- Background work: household voice assistant (open-xiaoai) on the same engine: prompt-prefix warm-ups about every 20 min; Qwen3-ASR on the Intel iGPU, March7 TTS on CPU (idle unless spoken to)

## Model and configuration

- Model: qwen3.8-flash-next-iq2_xs; quantization IQ2_XS; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` `92cee27ae5bbadcd...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (Strata 0.2.1 (upstream 9ae93f7b) + 10 commits (github.com/shenmintao/Strata branch nas/gfx1200-021 @ 690e4f6f: Q4_0-KV WMMA prompt attention STRATA_HIP_WMMA_Q4, conversation disk tier, elastic CPU image encoder, /v1/models fields, retuned gfx1200 hipBLASLt table, NAS Docker build); Docker on fnOS (Debian 12), ROCm 7.2.3 userspace; run config: IQ2_XS native pack, --kv q4_0 --max-context 524288 --batch 2 --batch-mtp --prefill 8192 --spec 4 --spec-min-p 0.5 --resident-experts --pool-workers 13 --vram-reserve-mib 896 --vision (CPU encoder, elastic); env GPU_MAX_HW_QUEUES=2 HSA_ENABLE_SDMA=1 STRATA_ATTN_LANECELL=1 STRATA_DMA_BATCH=0 STRATA_HC_Q8=1 STRATA_HIPBLASLT_TUNING=<the retuned 162-row gfx1200 table> STRATA_HIP_QUEUE_PRIME=0 STRATA_HIP_WMMA_Q4=1 STRATA_MTP_SIDE_STREAM=1 STRATA_PF_FUSED=1 STRATA_POOL_SPIN_US=0 STRATA_PREFILL_RING=384 STRATA_RESIDENT_HEADROOM_GIB=5 STRATA_ROUTE_TAIL_SKIP=7 STRATA_SELECT_WMMA=1)
- Context 524288 tokens; server capacity 2 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 38 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s |
| --- | --- | --- | --- | --- | --- |
| short | 75.1 (73.5-77.4) n=3 | 46.9 (42.8-47.4) n=3 | 1.06 (1.03-1.08) n=3 | 3.68 (2.87-7.90) n=3 | 8.3 (7.3-12.2) n=3 |
| prompt-4k | 1219.1 (843.3-1239.9) n=3 | 41.8 (40.5-42.5) n=3 | 3.40 (3.34-4.89) n=3 | not measured | 16.0 (15.4-17.1) n=3 |
| prompt-16k | not measured | not measured | not measured | not measured | not measured |

Cells are median (min-max) n=samples.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 3 | 9 | 0 | 0 |

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
| 1 | 41.53456 | 2048 | 49.308335 | 0 |
| 2 | 37.642154 | 1847 | 49.06733 | 0 |
| 3 | 38.386797 | 1851 | 48.219704 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c3: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r2-c4: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r1: unexpected cache reuse: 16382 cached tokens exceed the 55-token shared template prefix
- Excluded from speed summaries: performance-prompt-16k-r2: unexpected cache reuse: 16382 cached tokens exceed the 55-token shared template prefix
- Excluded from speed summaries: performance-prompt-16k-r3: unexpected cache reuse: 16382 cached tokens exceed the 55-token shared template prefix
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 1 request(s) before this run.
