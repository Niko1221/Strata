# Strata standard benchmark: AMD Radeon RX 6800 XT

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261116T104014Z-12060451`, started 2026-11-16T10:40:14Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon RX 6800 XT
- CPU: AMD Ryzen 5 5500; RAM: 62.6 GiB
- Storage of the model files: nvme CT1000E100SSD8 (nvme)
- OS: Ubuntu 24.04.4 LTS 6.8.0-149-generic; drivers: gpu-driver 6.8.0-149-generic; runtime: rocm 6.4.3-128
- Strata commit: `baf16363b4403ef0ed4f538f5d5ed11c43fb9535`; engine 0.1.60.1; source build
- Background work: headless Ubuntu server, no desktop; image model service stopped; no other GPU process

## Model and configuration

- Model: qwen3.8-flash-next; quantization ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (1x RX 6800 XT 16 GB on a CPU PCIe 3.0 x8 slot (Ryzen 5 5500 is Gen3-only), same options as the two-card run, STRATA_HIP_PROMPT_F16=1, 5 pool workers, 128K context)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 6 min 31 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 36.3 (33.5-39.1) n=3 | 2.63 (2.57-2.67) n=3 | 4.51 (4.45-6.83) n=3 | 13.0 (11.3-13.6) n=3 | 0/3 |
| prompt-4k | 394.8 (394.7-395.1) n=3 | 32.4 (30.8-32.6) n=3 | 10.42 (10.42-10.42) n=3 | 21.87 (21.87-21.87) n=1 | 26.2 (26.1-27.0) n=3 | 3/3 |
| prompt-16k | 613.6 (607.2-616.2) n=3 | 32.2 (31.1-32.4) n=3 | 26.76 (26.65-27.06) n=3 | not measured | 42.8 (42.6-43.1) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 2 | 10 | 0 | 0 |

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
| 1 | 71.930349 | 2024 | 28.138331 | 0 |
| 2 | 70.280082 | 2048 | 29.140547 | 0 |
| 3 | 71.745323 | 2048 | 28.545415 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r3-c1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
