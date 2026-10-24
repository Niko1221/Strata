# Strata standard benchmark: NVIDIA GeForce RTX 2080 Ti

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261024T062225Z-375ff2de`, started 2026-10-24T06:22:25Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 2080 Ti (22.0 GiB dedicated)
- CPU: Intel(R) Xeon(R) CPU E5-2696 v4 @ 2.20GHz; RAM: 94.3 GiB
- Storage of the model files: nvme WD Green SN350 1TB (nvme)
- OS: Ubuntu 24.04.5 LTS 7.0.0-38-generic; drivers: gpu-driver 580.178.04; runtime: cuda 13.0
- Strata commit: `06e51dfaef7003f01f23b85e63c800b10efe4066`; engine 0.1.49; source build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-iq4_xs; quantization unsloth/UD-IQ4_XS; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` `5ce89370720f8bf9...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf` `577a38a2392b40ca...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf` `d4634e6d84f0ebb0...`
- Track: **custom** (layer split gpu [0,1] + --batch 4 --batch-groups 2 + --prompt-cache-every 4096 + --resident-budget-gib 55; Unsloth UD-IQ4_XS; 180 W power limit; dual-socket Xeon E5-2696 v4)
- Context 131072 tokens; server capacity 4 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 52 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 46.9 (45.8-48.5) n=3 | 1.68 (1.12-1.68) n=3 | 3.40 (3.23-6.71) n=3 | 9.4 (5.9-11.3) n=3 | 0/3 |
| prompt-4k | 696.3 (664.8-697.6) n=3 | 47.1 (44.6-51.9) n=3 | 5.91 (5.91-6.20) n=3 | 13.63 (12.32-14.94) n=2 | 16.8 (15.8-17.7) n=3 | 3/3 |
| prompt-16k | 1113.5 (1113.4-1114.2) n=3 | 50.7 (48.6-52.3) n=3 | 14.77 (14.76-14.78) n=3 | 21.84 (20.50-23.42) n=3 | 24.8 (24.5-25.3) n=3 | 3/3 |

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
| 1 | 31.208511 | 1790 | 57.356149 | 0 |
| 2 | 28.436389 | 1774 | 62.384855 | 0 |
| 3 | 28.271586 | 1979 | 69.99961 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r2-c4: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c4: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 6 request(s) before this run.
