# Strata standard benchmark: AMD Radeon RX 9070 XT

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261123T011024Z-1e074ab8`, started 2026-11-23T01:10:24Z. **Status: nonconforming.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

This run is not a complete standard result:

- the Strata commit is unknown (not a git checkout and no COMMIT file)
- no model file could be hashed
- performance-prompt-4k-r1 measured 2795 prompt tokens, outside 2% of its target
- performance-prompt-16k-r1 measured 11206 prompt tokens, outside 2% of its target
- performance-prompt-4k-r2 measured 2795 prompt tokens, outside 2% of its target
- performance-prompt-16k-r2 measured 11206 prompt tokens, outside 2% of its target
- performance-prompt-4k-r3 measured 2795 prompt tokens, outside 2% of its target
- performance-prompt-16k-r3 measured 11206 prompt tokens, outside 2% of its target
- sanity-R1 measured 2789 prompt tokens, outside 2% of its target
- sanity-R2 measured 2790 prompt tokens, outside 2% of its target
- sanity-R3 measured 11200 prompt tokens, outside 2% of its target
- sanity-R4 measured 11207 prompt tokens, outside 2% of its target

## Hardware and software

- GPU: AMD Radeon RX 9070 XT
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor; RAM: 91.9 GiB
- Storage of the model files: not measured
- OS: Pop!_OS 24.04 LTS 7.2.15-76070215-generic; drivers: gpu-driver 7.2.15-76070215-generic; runtime: rocm 7.2.4
- Strata commit: `not measured`; engine hy4-serve-0; release build
- Background work: desktop compositor on the same card; no other inference; work VM off

## Model and configuration

- Model: hy4-flash-sq-64; quantization not measured; revision not measured; source revision not measured
- Files: not measured
- Track: **custom** (Strata 0.2.20 source checkout; run config written by hand or by strata launch, see settings)
- Context 32768 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 7 min 45 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 22.0 (11.9-28.3) n=3 | 0.53 (0.51-1.16) n=3 | 24.00 (13.15-34.85) n=2 | 23.8 (18.6-44.1) n=3 | 3/3 |
| prompt-4k | 411.7 (392.3-419.4) n=3 | 20.4 (15.1-26.1) n=3 | 6.79 (6.67-7.13) n=3 | 25.50 (19.97-27.95) n=3 | 31.9 (26.3-40.8) n=3 | 2/3 |
| prompt-16k | 502.0 (431.8-503.3) n=3 | 21.0 (11.0-22.2) n=3 | 22.34 (22.28-25.96) n=3 | 43.69 (43.69-43.69) n=1 | 49.0 (46.6-69.1) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 1 | 8 | 0 | 0 |
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
- Excluded from speed summaries: performance-short-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
