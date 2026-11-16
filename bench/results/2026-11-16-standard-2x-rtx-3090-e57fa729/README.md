# Strata standard benchmark: NVIDIA GeForce RTX 3090, NVIDIA GeForce RTX 3090

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261116T221648Z-e57fa729`, started 2026-11-16T22:16:48Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3090 (24.0 GiB dedicated); NVIDIA GeForce RTX 3090 (24.0 GiB dedicated)
- CPU: AMD Ryzen 9 3950X 16-Core Processor; RAM: 121.4 GiB
- Storage of the model files: nvme Samsung SSD 970 PRO 1TB (nvme)
- OS: Ubuntu 26.04.1 LTS 7.0.0-43-generic; drivers: gpu-driver 595.99.02; runtime: cuda 13.2
- Strata commit: `df7cd10802be2a4f5f9dfcf46fa6a4589f8f2917`; engine 0.1.61; source build
- Background work: none (headless desktop session, no other GPU work)

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s-dual; quantization agentionai Gyro-M TQ2_0; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf` `7aa36d15131d2551...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (Strata v0.2.15 + 3 carried patches (#1937 conversations to disk, #1099 P2P decode hand-off, table-only PLE shard); Gyro-M on 2x RTX 3090 NVLink, layer split 24/24, every expert resident)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 5 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 169.5 (158.1-170.6) n=3 | 0.38 (0.25-0.38) n=3 | 1.00 (0.87-1.05) n=3 | 2.6 (2.1-2.6) n=3 | 0/3 |
| prompt-4k | 2692.2 (2678.5-2704.1) n=3 | 163.2 (139.1-165.4) n=3 | 1.54 (1.53-1.54) n=3 | 3.55 (2.99-4.89) n=3 | 4.7 (4.6-5.2) n=3 | 3/3 |
| prompt-16k | 3882.7 (3871.9-3887.4) n=3 | 145.0 (136.0-157.9) n=3 | 4.25 (4.24-4.26) n=3 | 6.69 (6.47-6.90) n=2 | 7.8 (7.5-8.0) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 7 | 5 | 0 | 0 |

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

Concurrency rounds (four requests released together):

| Round | Wall s | Output tokens | Aggregate tok/s | Failed |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10.791718 | 1626 | 150.671098 | 0 |
| 2 | 11.443298 | 1812 | 158.345959 | 0 |
| 3 | 13.887526 | 2000 | 144.014132 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
