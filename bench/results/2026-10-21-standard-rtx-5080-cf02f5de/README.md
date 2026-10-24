# Strata standard benchmark: NVIDIA GeForce RTX 5080

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261021T213810Z-cf02f5de`, started 2026-10-21T21:38:10Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 5080 (15.9 GiB dedicated)
- CPU: AMD Ryzen 7 7800X3D 8-Core Processor; RAM: 94.9 GiB
- Storage of the model files: ssd INTEL SSDPED1D480GA (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 617.42; runtime: not measured
- Strata commit: `e6fff4ad12908de74952abe53b2bade45a68aa0f`; engine 0.1.47; release build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-q2_0; quantization Q2_0; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` `69820c02ec7d0b45...`; `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (no notes given by the operator)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 14 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s |
| --- | --- | --- | --- | --- | --- |
| short | 135.1 (134.3-161.0) n=3 | 141.0 (137.5-146.9) n=3 | 0.59 (0.50-0.59) n=3 | 1.06 (1.06-1.38) n=3 | 3.2 (3.1-4.0) n=3 |
| prompt-4k | 3313.4 (3305.1-3318.2) n=3 | 130.0 (127.5-130.5) n=3 | 1.25 (1.25-1.26) n=3 | 3.50 (3.47-3.84) n=3 | 5.2 (5.2-5.3) n=3 |
| prompt-16k | 4247.4 (4163.0-4251.8) n=3 | 128.8 (118.9-130.3) n=3 | 3.89 (3.88-3.97) n=3 | 6.88 (5.70-8.05) n=2 | 7.8 (7.1-8.2) n=3 |

Cells are median (min-max) n=samples.

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
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 30 request(s) before this run.
