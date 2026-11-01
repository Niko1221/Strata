# Strata standard benchmark: NVIDIA GeForce GTX 1070 Ti

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261101T124316Z-22feeead`, started 2026-11-01T12:43:16Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce GTX 1070 Ti (8.0 GiB dedicated)
- CPU: AMD Ryzen 9 5900XT 16-Core Processor; RAM: 125.7 GiB
- Storage of the model files: nvme Lexar SSD NM790 1TB (nvme)
- OS: Ubuntu 24.04.4 LTS 7.0.0-34-generic; drivers: gpu-driver 580.178.04; runtime: cuda 13.0
- Strata commit: `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`; engine 0.1.55; source build
- Background work: none: dedicated headless box, strata.service stopped

## Model and configuration

- Model: swift-1.5-iq2_xs; quantization IQ2_XS; revision not measured; source revision not measured
- Files: `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` `ca3b302d3d1eb6ed...`; `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf` `e4996a6de18fd49c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **default** (v0.2.9 tag, source build (Pascal needs the CUDA 12 engine), setup-generated config)
- Context 32768 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 9 min 50 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 34.0 (34.0-34.9) n=3 | 1.32 (1.18-1.34) n=3 | 3.90 (2.92-4.58) n=3 | 11.4 (11.1-11.6) n=3 | 0/3 |
| prompt-4k | 252.2 (250.6-252.2) n=3 | 32.2 (30.2-35.0) n=3 | 16.28 (16.27-16.38) n=3 | 24.43 (21.09-27.78) n=2 | 32.1 (30.7-33.2) n=3 | 2/3 |
| prompt-16k | 245.2 (245.1-246.5) n=3 | 31.8 (30.7-32.1) n=3 | 66.88 (66.54-66.91) n=3 | 77.10 (77.10-77.10) n=1 | 82.8 (82.6-83.5) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
