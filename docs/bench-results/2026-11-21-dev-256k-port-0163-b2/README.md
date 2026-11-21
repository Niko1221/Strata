# Strata standard benchmark: Intel(R) Arc(TM) Pro B70 Graphics

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261121T175255Z-d9338e1d`, started 2026-11-21T17:52:55Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: Intel(R) Arc(TM) Pro B70 Graphics (32.0 GiB dedicated)
- CPU: Intel(R) Core(TM) i9-9900 CPU @ 3.10GHz; RAM: 63.8 GiB
- Storage of the model files: ssd Samsung SSD 860 EVO 250GB (raid)
- OS: Windows 10.0.26300; drivers: gpu-driver 32.0.101.8976; runtime: not measured
- Strata commit: `e0d84f2c6cc8234cfbb396a7d56727394b0681d1`; engine 0.1.63; source build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-coder-iq1_m; quantization IQ1_M; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf` `e11083ba855e7666...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (no notes given by the operator)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 33 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 82.8 (78.3-83.4) n=3 | 0.64 (0.57-0.64) n=3 | 1.99 (1.89-2.45) n=3 | 5.5 (4.9-6.0) n=3 | 0/3 |
| prompt-4k | 1214.5 (880.1-1218.9) n=3 | 74.0 (71.4-85.7) n=3 | 3.41 (3.40-4.69) n=3 | 7.93 (5.81-10.79) n=3 | 10.3 (9.4-11.8) n=3 | 3/3 |
| prompt-16k | 1406.9 (1406.0-1410.7) n=3 | 77.0 (68.8-77.4) n=3 | 11.70 (11.66-11.71) n=3 | 16.09 (15.93-16.24) n=2 | 18.3 (18.3-19.1) n=3 | 3/3 |

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
- The server had answered 22 request(s) before this run.
