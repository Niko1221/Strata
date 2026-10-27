# Strata standard benchmark: NVIDIA GeForce RTX 3080, NVIDIA TITAN RTX

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261027T004239Z-41033bda`, started 2026-10-27T00:42:39Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3080; NVIDIA TITAN RTX
- CPU: Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz; RAM: 125.7 GiB
- Storage of the model files: nvme SAMSUNG MZVL21T0HCLR-00B00 (nvme)
- OS: Ubuntu 24.04.5 LTS 6.8.0-147-generic; drivers: not measured; runtime: not measured
- Strata commit: `4b9e05a2a806a3fe24f7b2560d98626c37dade35`; engine 0.1.48; source build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-ud-q4kxl; quantization UD-Q4_K_XL; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` `4448186216b3af4c...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf` `6f3e21a5ae10c912...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf` `bb24c1d9c42c129a...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf` `3d6a9aee4ade5740...`
- Track: **custom** (q4kxl-swap-check)
- Context 262144 tokens; server capacity 2 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 6 min 32 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 38.4 (30.0-39.5) n=3 | 1.94 (1.87-1.97) n=3 | 9.59 (8.67-13.89) n=3 | 15.4 (15.0-22.5) n=3 | 1/3 |
| prompt-4k | 679.3 (516.3-691.1) n=3 | 37.5 (37.5-49.5) n=3 | 6.07 (5.99-7.97) n=3 | 14.49 (14.49-14.83) n=3 | 19.8 (16.7-21.3) n=3 | 2/3 |
| prompt-16k | 522.1 (512.4-522.4) n=3 | 35.8 (34.2-35.9) n=3 | 34.59 (31.44-39.78) n=3 | not measured | 49.1 (46.5-54.2) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 4 request(s) before this run.
