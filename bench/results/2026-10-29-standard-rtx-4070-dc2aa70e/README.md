# Strata standard benchmark: NVIDIA GeForce RTX 4070

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261029T090659Z-dc2aa70e`, started 2026-10-29T09:06:59Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 4070 (12.0 GiB dedicated)
- CPU: AMD Ryzen 9 5950X 16-Core Processor; RAM: 125.7 GiB
- Storage of the model files: nvme CT1000T500SSD5 (nvme)
- OS: Ubuntu 24.04.5 LTS 7.0.0-38-generic; drivers: gpu-driver 595.91.07; runtime: cuda 13.2
- Strata commit: `4b9e05a2a806a3fe24f7b2560d98626c37dade35`; engine 0.1.51; source build
- Background work: headless Linux box, nothing else on the GPUs; the server is the box's normal service, idle during the run

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (v0.2.6 CUDA source build (engine 0.1.51), 2x RTX 4070 layer split 25 (--layer-split 25), STRATA_STAGE_TRIM=1 STRATA_PREFILL_CPU_SHARE=auto STRATA_PREFILL_CPU_SHARE_MAX=3072 STRATA_SPLIT_MTP_BATCH=1 STRATA_ADAPT_LAG=2 STRATA_ROUTE_TAIL_SKIP=0; --spec 3 --spec-min-p 0.70 --pcie-frac 0 --kv int8 --kv-resident 32768 --max-context 131072; conversation cache 32 GiB)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 4 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 86.4 (82.5-93.2) n=3 | 0.82 (0.53-0.84) n=3 | 2.12 (1.26-2.41) n=3 | 4.6 (4.5-4.9) n=3 | 0/3 |
| prompt-4k | 1181.8 (1181.4-1182.5) n=3 | 81.2 (79.0-86.3) n=3 | 3.49 (3.48-3.97) n=3 | 7.29 (6.95-7.64) n=2 | 9.9 (9.8-9.9) n=3 | 3/3 |
| prompt-16k | 2420.9 (2415.4-2472.0) n=3 | 82.8 (78.4-83.0) n=3 | 6.81 (6.80-7.25) n=3 | 11.13 (10.52-12.72) n=3 | 13.3 (13.0-13.4) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 0 | 0 | 0 | 0 | 0 |

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

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
