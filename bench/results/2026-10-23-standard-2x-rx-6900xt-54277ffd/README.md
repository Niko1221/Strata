# Strata standard benchmark: AMD Radeon, AMD Radeon

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261023T031335Z-54277ffd`, started 2026-10-23T03:13:35Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon; AMD Radeon
- CPU: AMD Ryzen 5 5600X 6-Core Processor; RAM: 121.5 GiB
- Storage of the model files: nvme PNY CS2140 2TB SSD (nvme)
- OS: Ubuntu 26.04.1 LTS 7.0.0-38-generic; drivers: not measured; runtime: not measured
- Strata commit: `34cd85340c42a551df70416813bddbaf602985c2`; engine 0.1.47; source build
- Background work: not measured

## Model and configuration

- Model: iq3s-split; quantization IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision de4b8e4d43b917e7706784d8bb445c9af86a3540
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (two cards, layer split auto 0-23/24-47, --pipeline-windows 2: the 0.2.2 rx6000 engine package (0.1.47) unchanged, with the upstream opt-in switches this box uses in production: STRATA_ROCBLAS_TUNING (tools/hip table), STRATA_IO_THREADS=128, STRATA_SH_STREAM=1, STRATA_HIP_PROMPT_F16=1, STRATA_HIP_ADAPT_KERNEL_COPY=1, STRATA_PREFILL_CPU_SHARE=auto, STRATA_HC_PERSIST=1, STRATA_PIPED_CONCURRENT=1, STRATA_PREFILL_STREAM_MIN=2048; --spec 3 --spec-min-p 0.7; run-config sampling temperature 0 / presence_penalty 0; 131072 context, kv int8, kv-resident 32768, no vision; cards at 245 W and -81 mV)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 48 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 76.2 (75.4-82.5) n=3 | 0.67 (0.67-0.68) n=3 | 1.89 (1.11-2.42) n=3 | 4.5 (4.2-6.0) n=3 | 0/3 |
| prompt-4k | 849.1 (846.3-851.7) n=3 | 66.4 (63.7-70.7) n=3 | 4.85 (4.83-4.86) n=3 | 9.85 (9.85-9.85) n=1 | 12.5 (12.1-12.9) n=3 | 3/3 |
| prompt-16k | 1369.1 (1361.7-1373.2) n=3 | 67.7 (65.7-73.8) n=3 | 12.00 (11.97-12.07) n=3 | 18.00 (16.67-18.32) n=3 | 19.6 (18.9-19.7) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
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
| 1 | 33.929303 | 2048 | 60.360804 | 0 |
| 2 | 26.022343 | 1814 | 69.709326 | 0 |
| 3 | 29.814331 | 2014 | 67.551407 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c1: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r1-c3: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r1-c4: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
