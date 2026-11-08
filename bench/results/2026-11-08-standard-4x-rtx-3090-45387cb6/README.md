# Strata standard benchmark: NVIDIA GeForce RTX 3090, NVIDIA GeForce RTX 3090, NVIDIA GeForce RTX 3090, NVIDIA GeForce RTX 3090

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261108T171117Z-45387cb6`, started 2026-11-08T17:11:17Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3090 (24.0 GiB dedicated); NVIDIA GeForce RTX 3090 (24.0 GiB dedicated); NVIDIA GeForce RTX 3090 (24.0 GiB dedicated); NVIDIA GeForce RTX 3090 (24.0 GiB dedicated)
- CPU: AMD Ryzen Threadripper 3990X 64-Core Processor; RAM: 251.5 GiB
- Storage of the model files: nvme CT1000P310SSD8 (nvme)
- OS: Ubuntu 24.04.5 LTS 7.0.0-38-generic; drivers: gpu-driver 580.196.07; runtime: cuda 13.0
- Strata commit: `a099d2953b6b70084d92da0f3ac7b015719a4d5e`; engine 0.1.58; source build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-iq4_xs; quantization UD-IQ4_XS; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` `5ce89370720f8bf9...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf` `577a38a2392b40ca...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf` `d4634e6d84f0ebb0...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (PT-r2: Strata-212p@a099d295, engine strata-212p-74702164d1e5e37d, CUDA 13.0 source build (STRATA_MMQ_TAB=ON), 4x RTX 3090 320 W, layer split over 4 cards; v0.2.12 + our series (havok2-htwo) + P9-P11 + upstream's sm_86 table, 16 slots, 262K, int8 KV, MTP, bit-changing defaults pinned to 0 (exact); server env: [STRATA_ADMIT_BY_NEW=1 STRATA_YIELD_SHORT=2048 STRATA_YIELDS_MAX=4 STRATA_ADMIT_SHORT_FIRST=8192 STRATA_PREFIX_DIAG=1])
- Context 262144 tokens; server capacity 16 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 0 min 54 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 155.7 (145.8-155.8) n=3 | 0.23 (0.14-0.23) n=3 | 0.79 (0.74-1.25) n=3 | 2.6 (2.2-2.9) n=3 | 0/3 |
| prompt-4k | 5030.7 (5024.6-5043.1) n=3 | 155.0 (145.7-163.9) n=3 | 0.83 (0.83-0.83) n=3 | 2.50 (1.96-2.85) n=3 | 4.1 (3.9-4.3) n=3 | 3/3 |
| prompt-16k | 6063.0 (6007.8-6087.1) n=3 | 125.2 (124.3-138.3) n=3 | 2.73 (2.72-2.75) n=3 | 5.94 (4.39-6.23) n=3 | 6.8 (6.3-6.8) n=3 | 2/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 4 | 5 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 6 | 6 | 0 | 0 |

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
| 1 | 6.936859 | 1963 | 282.981101 | 0 |
| 2 | 6.737848 | 2001 | 296.979095 | 0 |
| 3 | 6.614224 | 1829 | 276.525258 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c1: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
