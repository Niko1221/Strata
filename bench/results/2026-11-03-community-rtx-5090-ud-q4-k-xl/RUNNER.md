# Strata standard benchmark: NVIDIA GeForce RTX 5090

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261104T011658Z-129ef329`, started 2026-11-04T01:16:58Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 5090 (31.8 GiB dedicated)
- CPU: AMD Ryzen 9 5950X 16-Core Processor; RAM: 121.4 GiB
- Storage of the model files: not measured
- OS: Ubuntu 24.04.5 LTS 7.0.0-42-generic; drivers: gpu-driver 580.178.04; runtime: cuda 13.0
- Strata commit: `870aedd9e1ebe1c91f94da93b59af17729f6c5c0`; engine 0.1.54; release build
- Background work: shared production box; k8s pod; scheduled 02:15 CET re-run with crew idle (the 17:41 daytime run had operator agent traffic interleaving)

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-q4_k_xl; quantization UD-Q4_K_XL; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` `4448186216b3af4c...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf` `3f342f1c1580473f...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf` `56758f40269cad5c...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf` `753bda48b98ba4f1...`
- Track: **custom** (release 0.2.8 (build 0.1.54), docker image cuda13, resident-budget 55 GiB, parallel 1, kv int8 + kv-resident 32768, 262k ctx; quiet-night re-run)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 7 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 61.4 (56.2-63.7) n=3 | 0.55 (0.52-0.66) n=3 | 2.65 (2.41-3.00) n=3 | 6.6 (6.4-7.1) n=3 | 0/3 |
| prompt-4k | 1815.0 (1811.4-1817.0) n=3 | 66.4 (63.7-67.6) n=3 | 2.28 (2.28-2.28) n=3 | 6.06 (5.67-6.54) n=3 | 9.8 (9.8-10.3) n=3 | 2/3 |
| prompt-16k | 3116.2 (3110.8-3152.0) n=3 | 65.0 (61.3-72.4) n=3 | 5.29 (5.23-5.30) n=3 | not measured | 13.2 (12.3-13.6) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 4 | 5 | 0 | 0 |
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
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 925 request(s) before this run.
