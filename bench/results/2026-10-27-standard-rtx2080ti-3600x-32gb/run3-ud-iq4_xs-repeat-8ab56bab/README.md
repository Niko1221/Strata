# Strata standard benchmark: NVIDIA GeForce RTX 2080 Ti

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261027T065910Z-8ab56bab`, started 2026-10-27T06:59:10Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 2080 Ti (11.0 GiB dedicated)
- CPU: AMD Ryzen 5 3600X 6-Core Processor; RAM: 31.9 GiB
- Storage of the model files: ssd Netac NVMe SSD 1TB (nvme)
- OS: Windows 10.0.26200; drivers: gpu-driver 621.01; runtime: cuda 13.5
- Strata commit: `562a0c2ed7169ff6373f893b162106d55a828297`; engine 0.1.50; release build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-iq4_xs; quantization UD-IQ4_XS; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` `5ce89370720f8bf9...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf` `577a38a2392b40ca...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf` `d4634e6d84f0ebb0...`
- Track: **custom** (C2: C1 + UNBUFFERED_LOAD=0 + DECODE_TIMING)
- Context 32768 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 45 min 33 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 3.0 (2.7-3.3) n=3 | 15.89 (14.41-16.42) n=3 | 49.80 (33.66-56.94) n=3 | 143.0 (141.9-166.6) n=3 | 0/3 |
| prompt-4k | 107.8 (106.9-108.0) n=3 | 2.9 (2.9-3.1) n=3 | 38.47 (38.33-38.97) n=3 | 104.56 (104.56-104.56) n=1 | 211.9 (202.0-213.5) n=3 | 3/3 |
| prompt-16k | 151.7 (151.3-152.6) n=3 | 3.2 (3.0-3.5) n=3 | 108.50 (107.86-108.80) n=3 | 222.28 (222.28-222.28) n=1 | 269.7 (253.7-280.2) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 1 request(s) before this run.
