# Strata standard benchmark: AMD Radeon RX 6900 XT

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261029T103240Z-b696e06d`, started 2026-10-29T10:32:40Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon RX 6900 XT
- CPU: AMD Ryzen 5 5600X 6-Core Processor; RAM: 121.4 GiB
- Storage of the model files: nvme PNY CS2140 2TB SSD (nvme)
- OS: Ubuntu 26.04.1 LTS 7.0.0-38-generic; drivers: gpu-driver 7.0.0-38-generic; runtime: rocm 10.0.0
- Strata commit: `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d`; engine 0.1.52; source build
- Background work: not measured

## Model and configuration

- Model: iq3s-split; quantization IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision 5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (two cards, layer split auto, --pipeline-windows 2: the 0.2.7 rx6000 engine package (0.1.52) unchanged, with the upstream opt-in switches: STRATA_ROCBLAS_TUNING (tools/hip; the package's rocBLAS is 5.4.0 and the shipped gfx1030 table is 5.6.0, so the default kernels run), STRATA_IO_THREADS=128, STRATA_SH_STREAM=1, STRATA_HIP_PROMPT_F16=1, STRATA_HIP_ADAPT_KERNEL_COPY=1, STRATA_PREFILL_CPU_SHARE=auto, STRATA_HC_PERSIST=1, STRATA_PIPED_CONCURRENT=1, STRATA_PREFILL_STREAM_MIN=2048; --spec 3 --spec-min-p 0.7; run-config sampling temperature 0 / presence_penalty 0; 131072 context, kv int8, kv-resident 32768, no vision; cards at 245 W and -81 mV, Ryzen 5 5600X, 128 GB DDR4-3200, PCIe 4.0 x8 per card, Ubuntu 26.04 kernel 7.0.0-38, ROCm 10.0)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 52 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 73.8 (71.5-75.7) n=3 | 0.89 (0.79-0.90) n=3 | 2.54 (2.29-3.32) n=3 | 6.2 (5.2-6.6) n=3 | 0/3 |
| prompt-4k | 828.5 (828.4-842.6) n=3 | 66.0 (65.1-75.4) n=3 | 4.97 (4.88-4.97) n=3 | 10.58 (9.20-11.96) n=2 | 12.6 (11.7-12.8) n=3 | 3/3 |
| prompt-16k | 1355.9 (1353.3-1357.7) n=3 | 74.0 (64.5-75.1) n=3 | 12.12 (12.10-12.14) n=3 | 16.31 (16.07-16.56) n=2 | 19.0 (18.9-20.0) n=3 | 3/3 |

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
| 1 | 30.738559 | 1943 | 63.21051 | 0 |
| 2 | 28.119756 | 1895 | 67.390343 | 0 |
| 3 | 30.498283 | 2048 | 67.151321 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
