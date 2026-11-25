# Strata standard benchmark: NVIDIA GeForce RTX 3090

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261125T221305Z-f86a98b8`, started 2026-11-25T22:13:05Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3090 (24.0 GiB dedicated)
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor; RAM: 63.6 GiB
- Storage of the model files: ssd Lexar SSD NQ790 2TB (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 617.42; runtime: cuda 13.4
- Strata commit: `2af30ccaa0729177c4725ee2073f72bf550e2f86`; engine 0.1.67; release build
- Background work: OpenCode chat session connected but idle during run

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-q2_k_xl; quantization UD-Q2_K_XL; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00004.gguf` `297e174dc185f2bf...`; `Qwen3.8-Flash-Next-UD-Q2_K_XL-00002-of-00004.gguf` `a4782511810bb61b...`; `Qwen3.8-Flash-Next-UD-Q2_K_XL-00003-of-00004.gguf` `4dffc4a1cea7ad48...`; `Qwen3.8-Flash-Next-UD-Q2_K_XL-00004-of-00004.gguf` `5de8b5dbf89409ac...`; `draft_vocab.bin` `b1e1d3a7a9e4bf86...`; `expert-profile.bin` `8f59b4aa8873209d...`; `qwen3.8-flash-next-dflash2.q4_0.pack` `cda40d62d0eb8f37...`
- Track: **custom** (post-tune engine 0.1.67: args +--adapt-async 1; env +STRATA_L3_PF=1, STRATA_PIN_BY_NEED=1, STRATA_SELECT_TWO_PASS=1, STRATA_PREFILL_ROUTED_STREAM=1; ROUTE_TAIL_SKIP=0 kept; large pages enabled 2026-11-26)
- Context 204800 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 3 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 34.9 (33.6-36.3) n=3 | 0.53 (0.52-0.59) n=3 | 2.47 (2.27-4.03) n=3 | 9.0 (7.8-11.6) n=3 | 0/3 |
| prompt-4k | 1937.0 (1191.8-1938.2) n=3 | 30.4 (25.5-39.9) n=3 | 2.16 (2.14-3.45) n=3 | 14.47 (12.61-15.73) n=3 | 20.3 (14.9-22.2) n=3 | 3/3 |
| prompt-16k | 2581.8 (2489.4-2599.1) n=3 | 39.7 (36.2-43.8) n=3 | 6.41 (6.34-6.64) n=3 | 15.27 (12.89-17.66) n=2 | 19.5 (18.0-20.5) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 14 request(s) before this run.
