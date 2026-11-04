# Strata standard benchmark: AMD Radeon(TM) 8060S Graphics

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261103T093522Z-224ddae3`, started 2026-11-03T09:35:22Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon(TM) 8060S Graphics (96.0 GiB dedicated)
- CPU: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S; RAM: 31.6 GiB
- Storage of the model files: ssd Sandisk PC SN5100S SDFQTSM-1T00-1002 (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 32.0.22032.6002; runtime: not measured
- Strata commit: `c5ab5d4003161333044f6f16f5dacd40b7be6d9a`; engine 0.1.56; release build
- Background work: local AI service paused; desktop session only

## Model and configuration

- Model: qwen3.6-35b-a3b-ud-q4_k_xl; quantization UD-Q4_K_XL; revision not measured; source revision not measured
- Files: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` `55983c5a75a1ab96...`
- Track: **default** (release 0.2.10 (engine 0.1.56), setup defaults: strata install --family qwen36 --model UD-Q4_K_XL --context 131072 --vision no --parallel 1)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 6 min 11 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 53.3 (52.8-53.9) n=3 | 0.20 (0.20-0.20) n=3 | not measured | 9.8 (9.7-9.9) n=3 | 3/3 |
| prompt-4k | 730.6 (728.7-732.4) n=3 | 46.1 (46.1-46.2) n=3 | 5.58 (5.56-5.58) n=3 | not measured | 16.7 (16.7-16.7) n=3 | 3/3 |
| prompt-16k | 589.9 (588.4-590.8) n=3 | 29.9 (29.9-29.9) n=3 | 27.73 (27.69-27.81) n=3 | not measured | 44.9 (44.8-44.9) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 0 | 9 | 0 | 0 |
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
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-short-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-short-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-short-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
