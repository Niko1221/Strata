# Strata standard benchmark: NVIDIA GeForce RTX 5090 Laptop GPU

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261120T195352Z-8eb18178`, started 2026-11-20T19:53:52Z. **Status: nonconforming.**

Measured by **jie3000**. This folder and
[`2026-11-20-standard-rtx-5090-laptop-gpu-ef253641`](../2026-11-20-standard-rtx-5090-laptop-gpu-ef253641/README.md)
are the same machine, model and configuration with a different engine (0.1.62
vs 0.1.63); see [NOTES.md](NOTES.md) for the A/B, the run configs and why the
runner recorded `nonconforming`.

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

This run is not a complete standard result:

- no model file could be hashed

## Hardware and software

- GPU: NVIDIA GeForce RTX 5090 Laptop GPU (23.9 GiB dedicated); PCIe link Gen 5 x16
- CPU: Intel(R) Core(TM) Ultra 9 275HX; RAM: 255.4 GiB
- Storage of the model files: not measured
- OS: Windows 10.0.26300; drivers: gpu-driver 617.42; runtime: cuda 13.4
- Strata commit: `12d491da4a6904eb49aaa6f2f63ec4b1a17582e3`; engine 0.1.62; release build
- Background work: not measured; GPU power limit 150 W (default) / 175 W (max)

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-q4_k_xl; quantization UD-Q4_K_XL; revision not measured; source revision not measured
- Files: not measured
- Track: **custom** (engine A/B 0.1.162 on Control 0.1.63, UD-Q4_K_XL user pack)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 23 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 31.0 (29.7-33.1) n=3 | 1.01 (0.78-1.06) n=3 | 3.53 (3.14-5.69) n=3 | 10.1 (9.5-15.7) n=3 | 0/3 |
| prompt-4k | 1641.7 (1617.4-1644.9) n=3 | 35.8 (33.9-40.3) n=3 | 2.56 (2.53-2.59) n=3 | 9.94 (9.05-12.91) n=3 | 16.9 (15.2-17.6) n=3 | 3/3 |
| prompt-16k | 2617.4 (2565.2-2755.9) n=3 | 36.8 (32.4-37.0) n=3 | 6.31 (6.00-6.44) n=3 | 15.28 (14.31-16.25) n=2 | 20.2 (19.8-22.2) n=3 | 3/3 |

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
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
