# Strata standard benchmark: AMD Radeon

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261020T163249Z-19c44879`, started 2026-10-20T16:32:49Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

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

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 36 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s |
| --- | --- | --- | --- | --- | --- |
| short | 69.1 (54.7-75.5) n=3 | 45.9 (45.4-47.0) n=3 | 1.16 (1.06-1.45) n=3 | 7.66 (5.93-7.82) n=3 | 11.6 (11.1-12.3) n=3 |
| prompt-4k | 1219.5 (1213.2-1234.2) n=3 | 41.6 (40.9-46.2) n=3 | 3.39 (3.36-3.41) n=3 | 10.45 (10.45-10.45) n=1 | 15.7 (14.4-15.8) n=3 |
| prompt-16k | 1507.8 (1506.2-1508.1) n=3 | 42.5 (41.9-49.0) n=3 | 10.93 (10.92-10.93) n=3 | 15.97 (15.97-15.97) n=1 | 22.9 (21.3-23.1) n=3 |

Cells are median (min-max) n=samples.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 2 | 7 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 2 | 10 | 0 | 0 |

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
| 1 | 42.519855 | 2048 | 48.165733 | 0 |
| 2 | 39.340754 | 1996 | 50.736191 | 0 |
| 3 | 40.941439 | 1973 | 48.190783 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c1: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r1-c4: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r2-c1: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c2: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c4: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The runner could not tell whether the server was freshly loaded.

## Submitter's notes (added by hand; everything above is the runner's)

**Hardware the runner could not read.** GPU: AMD Radeon RX 9060 XT 16 GB (gfx1200), on a PCIe 4.0 x8 link (the
board's root port). CPU i7-12700H (6P + 8E, no AVX-512, AVX-VNNI), 64 GB DDR4-3200 dual channel. The OS line above is
the container's (Ubuntu 24.04, ROCm 7.2.3 userspace, hipBLASLt 1.2.2); the host is fnOS (Debian 12 based, kernel 6.18).
Model files on a ZFS pool on the NAS; the PLE shard on a 16K-record dataset.

**What ran.** `strata-nas` Docker image built from
[shenmintao/Strata `nas/gfx1200-021` @ 690e4f6f](https://github.com/shenmintao/Strata/tree/nas/gfx1200-021) (the image
carries `/opt/strata/COMMIT`): upstream 0.2.1 (`9ae93f7b`) plus ten commits - an opt-in Q4_0-KV WMMA prompt attention
(`STRATA_HIP_WMMA_Q4`), a conversation-cache disk tier (`--conversation-disk-dir`), an elastic CPU image encoder,
extra `/v1/models` fields, a gfx1200 hipBLASLt table retuned with T 4096/6144/8192 buckets, and the NAS Docker build.
The full run config (LAN address replaced) is [provenance/run-config.json](provenance/run-config.json); the engine's
log of this run is [engine-log.txt](engine-log.txt).  The engine also serves a household voice assistant, which re-warms its
5K-token system prompt about every 20 minutes; the runner was started after its warm-up following the fresh start.

**Other runs of this configuration (kept, not selected away):**

| run | engine | status | 4K prompt | 16K prompt | sanity |
|---|---|---|---:|---:|---:|
| `20261020T153736Z-a35739d0` | same engine source (1130de3c, no COMMIT file in the image) | nonconforming: commit unknown | 1220.3 | 1501.3 | 11/12 (R4 answered `{"north": -124, "south": -875}`) |
| `20261020T162259Z-8f6f45b7` | this image | nonconforming: the 16K prompts were read back from the disk tier the first run had filled | 1219.1 | (reused) | 12/12 |
| **`20261020T163249Z-19c44879` (this one)** | this image, the earlier runs' session files removed first | **complete** | **1219.5** | **1507.8** | **12/12** |

Their `results.json` and `README.md` are in [provenance/earlier-runs](provenance/earlier-runs).
