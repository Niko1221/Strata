# Strata standard benchmark: NVIDIA GeForce RTX 3090, NVIDIA GeForce RTX 3090, NVIDIA GeForce RTX 3090, NVIDIA GeForce RTX 3090

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261108T164512Z-e32ea56b`, started 2026-11-08T16:45:12Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3090 (24.0 GiB dedicated); NVIDIA GeForce RTX 3090 (24.0 GiB dedicated); NVIDIA GeForce RTX 3090 (24.0 GiB dedicated); NVIDIA GeForce RTX 3090 (24.0 GiB dedicated)
- CPU: AMD Ryzen Threadripper 3990X 64-Core Processor; RAM: 251.5 GiB
- Storage of the model files: nvme CT1000P310SSD8 (nvme)
- OS: Ubuntu 24.04.5 LTS 7.0.0-38-generic; drivers: gpu-driver 580.196.07; runtime: cuda 13.0
- Strata commit: `146df3c468739147f520ac3816de806f71232571`; engine 0.1.58; source build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-iq4_xs; quantization UD-IQ4_XS; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` `5ce89370720f8bf9...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf` `577a38a2392b40ca...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf` `d4634e6d84f0ebb0...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (S: Strata-212@146df3c4, engine strata-212-stocktab-f83fd81691a5b053, CUDA 13.0 source build (STRATA_MMQ_TAB=ON), 4x RTX 3090 320 W, layer split over 4 cards; stock v0.2.12: the args/env stock setup writes for this box (128K, int8 KV, MTP, parallel 3 as strata launch sets for 24 GB), vision off, sampling.presence_penalty 0 (inert here: the suite sends 0); server env: [])
- Context 131072 tokens; server capacity 3 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 18 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 127.4 (125.5-137.1) n=3 | 0.49 (0.38-0.49) n=3 | 1.79 (1.29-1.96) n=3 | 3.0 (3.0-3.5) n=3 | 0/3 |
| prompt-4k | 1921.4 (1912.9-1995.8) n=3 | 128.3 (124.2-136.6) n=3 | 2.15 (2.07-2.16) n=3 | 4.95 (4.35-5.55) n=2 | 6.1 (5.9-6.2) n=3 | 3/3 |
| prompt-16k | 3219.1 (3211.1-3285.0) n=3 | 127.9 (127.4-143.7) n=3 | 5.12 (5.01-5.13) n=3 | 7.90 (6.71-8.04) n=3 | 9.1 (8.6-9.1) n=3 | 3/3 |

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
| 1 | 9.751499 | 1933 | 198.225934 | 0 |
| 2 | 9.743433 | 1968 | 201.982197 | 0 |
| 3 | 10.431865 | 2048 | 196.321559 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
