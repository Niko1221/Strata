# Strata standard benchmark: Apple M3 Max

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261108T073052Z-5b747d5c`, started 2026-11-08T07:30:52Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: Apple M3 Max (128.0 GiB shared)
- CPU: Apple M3 Max; RAM: 128.0 GiB
- Storage of the model files: not measured
- OS: macOS 27.2; drivers: not measured; runtime: metal 27.2
- Strata commit: `146df3c468739147f520ac3816de806f71232571`; engine 0.1.58-metal; release build
- Background work: Strata Desktop 0.2.12 host process, no model loaded; Ollama server idle, cloud model only; macOS desktop session with idle browser and editors

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization ISTA-DASLab Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **default** (Strata Desktop 0.2.12, app-generated IQ3_S config, tuning profile IQ3_S 128K Full speed (standard settings), Metal engine 0.1.58, no env overrides)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 6 min 33 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 55.4 (37.7-57.1) n=3 | 0.64 (0.52-0.71) n=3 | 2.62 (1.83-2.87) n=3 | 7.7 (6.7-8.1) n=3 | 0/3 |
| prompt-4k | 481.5 (391.8-527.0) n=3 | 51.7 (29.6-53.7) n=3 | 8.52 (7.79-10.48) n=3 | 17.80 (12.57-23.03) n=2 | 18.0 (17.7-27.8) n=3 | 3/3 |
| prompt-16k | 479.3 (326.7-487.6) n=3 | 36.8 (26.4-50.3) n=3 | 34.22 (33.64-50.19) n=3 | not measured | 47.5 (44.4-69.5) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
- The optional concurrency stage was not run.
- The Apple M3 Max is not among the Macs measured in [MAC.md](https://github.com/Niko1221/Strata/blob/main/docs/MAC.md) (M5 Max, M4 mini, M1 Max); this run is offered as draft-protocol pilot data for a system the docs do not cover.
- The engine selected the older prompt kernels on this chip (engine log: `the older expert / attention kernels (no M5 layout)`). Separate non-standard testing found no measurable change with `STRATA_METAL_PROMPT_EXPERTS=v3`, `STRATA_METAL_PROMPT_CHUNK=4096` or `STRATA_METAL_F16=1`; those runs are not part of this report.
- `provenance/engine-log-excerpt.txt` was attached manually (the runner in `--url` mode does not capture the engine log); it covers this run only, from the model load to the last suite request.
