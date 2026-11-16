# Strata standard benchmark: AMD Radeon RX 6800 XT, AMD Radeon RX 6800 XT

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261116T103038Z-87126c1d`, started 2026-11-16T10:30:38Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon RX 6800 XT; AMD Radeon RX 6800 XT
- CPU: AMD Ryzen 5 5500; RAM: 62.6 GiB
- Storage of the model files: nvme CT1000E100SSD8 (nvme)
- OS: Ubuntu 24.04.4 LTS 6.8.0-149-generic; drivers: gpu-driver 6.8.0-149-generic; runtime: rocm 6.4.3-128
- Strata commit: `baf16363b4403ef0ed4f538f5d5ed11c43fb9535`; engine 0.1.60.1; source build
- Background work: headless Ubuntu server, no desktop; image model service stopped; no other GPU process

## Model and configuration

- Model: qwen3.8-flash-next; quantization ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (2x RX 6800 XT 16 GB on the two CPU PCIe 3.0 x8 slots (Ryzen 5 5500 is Gen3-only), layer split auto, setup-official options --spec-min-p 0.5 --kv int8 --kv-resident 32768, STRATA_HIP_PROMPT_F16=1, 5 pool workers, 128K context)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 4 min 42 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 53.3 (51.8-54.3) n=3 | 1.58 (1.43-1.62) n=3 | 3.43 (3.13-3.82) n=3 | 7.8 (6.7-8.7) n=3 | 0/3 |
| prompt-4k | 416.0 (399.0-416.5) n=3 | 46.1 (44.0-48.9) n=3 | 9.87 (9.86-10.30) n=3 | 17.68 (17.68-17.68) n=1 | 20.9 (20.3-21.9) n=3 | 3/3 |
| prompt-16k | 806.3 (805.2-807.6) n=3 | 50.7 (46.2-55.0) n=3 | 20.36 (20.33-20.39) n=3 | 25.83 (24.91-26.75) n=2 | 30.4 (29.6-31.5) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 5 | 7 | 0 | 0 |

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
| 1 | 41.04997 | 1757 | 42.801493 | 0 |
| 2 | 40.462168 | 1868 | 46.166582 | 0 |
| 3 | 44.995846 | 1957 | 43.492904 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r3-c4: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
