# Strata standard benchmark: unknown GPU

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261030T120215Z-54c71f0f`, started 2026-10-30T12:02:15Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: not measured
- CPU: Intel(R) Core(TM) i9-9900 CPU @ 3.10GHz; RAM: 63.8 GiB
- Storage of the model files: ssd Samsung SSD 860 EVO 250GB (raid)
- OS: Windows 10.0.26200; drivers: not measured; runtime: not measured
- Strata commit: `57c075c9223ceb414efc5ced4b9257a471864aa2`; engine 0.1.54; source build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-coder-iq1_m; quantization IQ1_M; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf` `e11083ba855e7666...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (no notes given by the operator)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 26 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 76.1 (72.6-77.2) n=3 | 0.61 (0.60-0.61) n=3 | 2.02 (1.77-2.13) n=3 | 5.0 (4.5-5.4) n=3 | 0/3 |
| prompt-4k | 1018.9 (845.3-1025.3) n=3 | 70.6 (59.9-70.9) n=3 | 4.06 (4.04-4.89) n=3 | 8.93 (7.75-12.17) n=3 | 12.1 (11.3-12.6) n=3 | 3/3 |
| prompt-16k | 1071.1 (1067.4-1074.3) n=3 | 65.5 (60.3-72.0) n=3 | 15.35 (15.30-15.41) n=3 | 20.40 (19.52-21.28) n=2 | 23.1 (22.5-23.8) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 22 request(s) before this run.
