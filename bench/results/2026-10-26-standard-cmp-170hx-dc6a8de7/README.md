# Strata standard benchmark: NVIDIA CMP 170HX

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261026T040148Z-dc6a8de7`, started 2026-10-26T04:01:48Z. **Status: nonconforming.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

This run is not a complete standard result:

- the Strata commit is unknown (not a git checkout and no COMMIT file)

## Hardware and software

- GPU: NVIDIA CMP 170HX (40.0 GiB dedicated)
- CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz; RAM: 247.5 GiB
- Storage of the model files: nvme ZHITAI TiPlus7100s 2TB (nvme)
- OS: Ubuntu 26.04 LTS 7.0.0-29-generic; drivers: gpu-driver 610.43.03; runtime: cuda 13.3
- Strata commit: `not measured`; engine 0.1.50; release build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision de4b8e4d43b917e7706784d8bb445c9af86a3540
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (Strata 0.1.50 CUDA engine. Dual-engine 2-card group A (PCI 02:00 + 82:00) serving :8080; twin group B idle on :8081 (no client). 8 sm_80 bit-exact switches: SIDE_MIXER=2 ATTN_DAILY=1 GR_UP75=1 GU_SWIGLU=1 QSA_ROWS=1 B6_MMVQ_ROWS=1 CPU_IQ_ALL_EXACT1=1 DEQUANT_WIDE=1. --prompt-cache 24 --prompt-cache-every 4096 --prompt-cache-spread. --kv q4_0 --kv-resident 20480. --pipeline-windows 2. --mtp --spec 4. --max-context 1048576. parallel 4 slots.)
- Context 1048576 tokens; server capacity 4 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 59 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 90.7 (89.3-100.0) n=3 | 0.98 (0.70-1.03) n=3 | 2.25 (2.04-2.40) n=3 | 4.5 (4.0-5.0) n=3 | 0/3 |
| prompt-4k | 1465.8 (1461.4-1467.4) n=3 | 78.2 (77.4-82.0) n=3 | 2.87 (2.86-2.87) n=3 | 8.61 (8.30-8.91) n=2 | 9.4 (9.1-9.5) n=3 | 3/3 |
| prompt-16k | 2445.2 (2431.8-2446.6) n=3 | 77.0 (76.7-79.4) n=3 | 6.79 (6.78-6.83) n=3 | 12.55 (12.28-12.83) n=2 | 13.5 (13.2-13.5) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

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
| 1 | 18.195271 | 2025 | 111.292654 | 0 |
| 2 | 17.750308 | 1956 | 110.195271 | 0 |
| 3 | 17.910912 | 2010 | 112.222091 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c4: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r2-c3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 2 request(s) before this run.
