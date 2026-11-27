# Strata standard benchmark: NVIDIA GeForce RTX 4090 Laptop GPU

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261124T162803Z-12b2b815`, started 2026-11-24T16:28:03Z. **Status: nonconforming.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

This run is not a complete standard result:

- no model file could be hashed
- performance-prompt-4k-r1 measured 3423 prompt tokens, outside 2% of its target
- performance-prompt-16k-r1 measured 13775 prompt tokens, outside 2% of its target
- performance-prompt-4k-r2 measured 3423 prompt tokens, outside 2% of its target
- performance-prompt-16k-r2 measured 13775 prompt tokens, outside 2% of its target
- performance-prompt-4k-r3 measured 3423 prompt tokens, outside 2% of its target
- performance-prompt-16k-r3 measured 13775 prompt tokens, outside 2% of its target
- sanity-R1 measured 3421 prompt tokens, outside 2% of its target
- sanity-R2 measured 3422 prompt tokens, outside 2% of its target
- sanity-R3 measured 13773 prompt tokens, outside 2% of its target
- sanity-R4 measured 13777 prompt tokens, outside 2% of its target

## Hardware and software

- GPU: NVIDIA GeForce RTX 4090 Laptop GPU (16.0 GiB dedicated)
- CPU: 13th Gen Intel(R) Core(TM) i9-13980HX; RAM: 62.5 GiB
- Storage of the model files: not measured
- OS: <host> 7.2.15-1-omarchy; drivers: gpu-driver 615.78.08; runtime: cuda 13.4
- Strata commit: `d46f5f268f8ed797e83a5c63a69bffbd7a581cf6`; engine strata-inkling-0.1.67; release build
- Background work: CPU governor powersave; GPU power cap 175 W, no other GPU workload

## Model and configuration

- Model: inkling-2-mini-sq-15; quantization SQ-15; revision not measured; source revision not measured
- Files: not measured
- Track: **custom** (SQ-15 128K preset, automatic prefill chunk 512, DSpark, context 131072. Not setup defaults.)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 39 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 85.2 (77.0-86.4) n=3 | 0.14 (0.14-0.14) n=3 | 3.50 (3.43-3.62) n=3 | 6.1 (5.9-6.6) n=3 | 3/3 |
| prompt-4k | 2195.4 (2043.7-2202.4) n=3 | 75.9 (74.7-76.7) n=3 | 1.58 (1.58-1.63) n=3 | 5.44 (5.21-5.64) n=3 | 8.2 (8.1-8.3) n=3 | 2/3 |
| prompt-16k | 1942.3 (1939.6-1943.2) n=3 | 20.2 (20.0-20.6) n=3 | 6.85 (6.85-6.87) n=3 | 19.56 (19.51-19.68) n=3 | 31.7 (29.8-32.1) n=3 | 2/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 2 | 7 | 0 | 0 |
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
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
