# Strata standard benchmark: AMD Radeon RX 9070 XT

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261123T010654Z-8698dd43`, started 2026-11-23T01:06:54Z. **Status: nonconforming.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

This run is not a complete standard result:

- the Strata commit is unknown (not a git checkout and no COMMIT file)
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

- GPU: AMD Radeon RX 9070 XT
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor; RAM: 91.9 GiB
- Storage of the model files: not measured
- OS: Pop!_OS 24.04 LTS 7.2.15-76070215-generic; drivers: gpu-driver 7.2.15-76070215-generic; runtime: rocm 7.2.4
- Strata commit: `not measured`; engine strata-inkling-0.1.66; release build
- Background work: desktop compositor on the same card; no other inference; work VM off

## Model and configuration

- Model: inkling-2-mini-sq-10; quantization not measured; revision not measured; source revision not measured
- Files: not measured
- Track: **custom** (Strata 0.2.20 source checkout; run config written by hand or by strata launch, see settings)
- Context 32768 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 14 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 62.0 (59.7-62.7) n=3 | 0.09 (0.09-0.09) n=3 | 0.09 (0.09-0.09) n=3 | 3.3 (2.4-3.4) n=3 | 0/3 |
| prompt-4k | 2376.4 (2367.1-2387.7) n=3 | 56.7 (54.1-59.4) n=3 | 1.45 (1.44-1.45) n=3 | 1.45 (1.44-1.45) n=3 | 4.8 (4.8-5.3) n=3 | 0/3 |
| prompt-16k | 2274.5 (2271.2-2280.4) n=3 | 44.1 (41.9-45.5) n=3 | 6.07 (6.06-6.08) n=3 | 6.07 (6.06-6.08) n=3 | 9.6 (8.2-9.8) n=3 | 0/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 9 | 0 | 0 | 0 |
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
