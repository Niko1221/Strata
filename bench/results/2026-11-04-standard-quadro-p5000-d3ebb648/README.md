# Strata standard benchmark: Quadro P5000

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261104T222951Z-d3ebb648`, started 2026-11-04T22:29:51Z. **Status: nonconforming.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

This run is not a complete standard result:

- no model file could be hashed

## Hardware and software

- GPU: Quadro P5000 (16.0 GiB dedicated)
- CPU: Intel(R) Xeon(R) CPU           X5650  @ 2.67GHz; RAM: 110.0 GiB
- Storage of the model files: not measured
- OS: Ubuntu 24.04.5 LTS 6.8.0-94-generic; drivers: gpu-driver 580.178.04; runtime: cuda 13.0
- Strata commit: `974074c549754029882c1cea1ce0e1e7674d7ba2`; engine 0.1.56; release build
- Background work: SearXNG idle on the same host; ComfyUI stopped

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision not measured; source revision ffa80d7
- Files: not measured
- Track: **custom** (Same as the IQ3_S default run of this machine + pw11; source build for a CPU without AVX (STRATA_ISA_FLOOR=none) + CUDA 12.9 sm_61)
- Context 32768 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 14 min 38 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 11.2 (11.1-11.9) n=3 | 2.65 (2.34-2.80) n=3 | 9.14 (8.42-10.52) n=3 | 30.8 (26.8-40.3) n=3 | 0/3 |
| prompt-4k | 277.6 (218.5-279.3) n=3 | 11.1 (10.0-11.3) n=3 | 14.84 (14.74-18.82) n=3 | not measured | 61.0 (59.9-70.0) n=3 | 3/3 |
| prompt-16k | 277.1 (269.2-277.2) n=3 | 12.3 (11.1-12.7) n=3 | 59.26 (59.24-60.98) n=3 | 92.71 (82.03-104.82) n=3 | 100.7 (99.5-106.9) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
