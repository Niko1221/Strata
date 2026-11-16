# Strata standard benchmark: NVIDIA GeForce RTX 4060 Ti, NVIDIA GeForce RTX 4060 Ti

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261116T101335Z-adea42ae`, started 2026-11-16T10:13:35Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 4060 Ti; NVIDIA GeForce RTX 4060 Ti
- CPU: Intel(R) Xeon(R) W-2145 CPU @ 3.70GHz; RAM: 110.0 GiB
- Storage of the model files: hdd Virtual Disk (unknown)
- OS: Ubuntu 26.04.1 LTS 6.18.58.1-microsoft-standard-WSL2; drivers: not measured; runtime: not measured
- Strata commit: `628f74955a56fcba092495b800dd02af75373a0f`; engine 0.1.59; release build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-q2_0; quantization Q2_0; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` `69820c02ec7d0b45...`; `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **default** (re-run of #5817 baseline (2026-11-11, app 0.2.13.1 / engine 0.1.59) on the same box; warm server; same protocol and sampling)
- Context 262144 tokens; server capacity 2 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 21 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 69.1 (65.5-71.6) n=3 | 1.25 (1.23-1.37) n=3 | 2.40 (2.39-2.41) n=3 | 6.3 (5.7-6.6) n=3 | 0/3 |
| prompt-4k | 692.7 (674.5-698.8) n=3 | 62.0 (57.2-65.2) n=3 | 5.94 (5.89-6.10) n=3 | 11.50 (11.34-13.86) n=3 | 14.1 (13.9-14.9) n=3 | 3/3 |
| prompt-16k | 1233.2 (1226.8-1238.5) n=3 | 57.2 (55.2-65.7) n=3 | 13.32 (13.27-13.40) n=3 | 20.68 (19.73-21.64) n=2 | 22.2 (21.2-22.6) n=3 | 3/3 |

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
- The server had answered 45 request(s) before this run.
