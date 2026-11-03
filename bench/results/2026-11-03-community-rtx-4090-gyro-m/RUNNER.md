# Strata standard benchmark: NVIDIA GeForce RTX 4090

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261103T164154Z-4a85149c`, started 2026-11-03T16:41:54Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 4090 (24.0 GiB dedicated)
- CPU: Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz; RAM: 61.6 GiB
- Storage of the model files: not measured
- OS: Ubuntu 24.04.5 LTS 7.0.0-42-generic; drivers: gpu-driver 580.178.04; runtime: cuda 13.0
- Strata commit: `870aedd9e1ebe1c91f94da93b59af17729f6c5c0`; engine 0.1.54; release build
- Background work: shared production box; k8s pod; light crew/Open-WebUI agent traffic

## Model and configuration

- Model: qwen3.8-flash-next-gyro-gyro-m; quantization not measured; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf` `7aa36d15131d2551...`
- Track: **default** (release 0.2.8 (build 0.1.54), official docker image cuda13, gyro preset, batch 3, 262k ctx)
- Context 262144 tokens; server capacity 3 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 13 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 52.6 (45.6-59.4) n=3 | 0.77 (0.53-0.81) n=3 | 2.36 (1.74-3.20) n=3 | 6.6 (6.0-7.8) n=3 | 0/3 |
| prompt-4k | 1754.5 (1751.5-1810.9) n=3 | 52.7 (50.0-58.1) n=3 | 2.36 (2.29-2.37) n=3 | 5.91 (5.75-6.07) n=2 | 11.2 (10.4-12.5) n=3 | 2/3 |
| prompt-16k | 4476.3 (4442.6-4597.5) n=3 | 57.5 (55.3-57.9) n=3 | 3.71 (3.62-3.74) n=3 | 9.37 (8.98-9.52) n=3 | 12.6 (12.6-12.9) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 4 | 5 | 0 | 0 |
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
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 6 request(s) before this run.
