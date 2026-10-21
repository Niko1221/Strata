# Strata standard benchmark: AMD Radeon

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261021T181920Z-c7a2be23`, started 2026-10-21T18:19:20Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon
- CPU: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S; RAM: 124.4 GiB
- Storage of the model files: nvme Lexar SSD NQ790 2TB (nvme)
- OS: Ubuntu 26.04.1 LTS 7.0.0-38-generic; drivers: not measured; runtime: not measured
- Strata commit: `011c9e600a93bbc9fd451c563501b5831c32bbca`; engine 0.1.47; source build
- Background work: headless Ubuntu server (kernel 7.0.0-38), no desktop; the Halogen and Paiton containers stopped; nothing else on either GPU; soft RLIMIT_NOFILE raised to the hard limit (524288)

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (Strata 0.2.2 (011c9e60) source build. Radeon AI PRO R9700 (gfx1201, OCuLink PCIe 4.0 x4, 7.2 GB/s measured) alone, the Strix Halo 8060S hidden (ROCR_VISIBLE_DEVICES=0); the whole expert arena in the Strix Halo's RAM (no --mmap-experts, cudaHostRegister PORTABLE), --expert-cache auto, --prefill 32768, --spec 4 --spec-min-p 0.5, MTP, --kv int8, --max-context 262144, --vision (CPU encoder); env STRATA_PF_FUSED=1 STRATA_PF_GEMM=1 STRATA_PA_FAST=1 STRATA_HIP_WMMA=1 STRATA_SELECT_WMMA=1; parallel 4 (--batch 4, --batch-mtp), --kv-resident 32768)
- Context 262144 tokens; server capacity 4 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 59 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 70.2 (66.9-78.8) n=3 | 1.22 (1.15-1.25) n=3 | 3.35 (1.97-4.32) n=3 | 7.5 (7.5-8.5) n=3 | 1/3 |
| prompt-4k | 689.7 (651.0-691.9) n=3 | 65.4 (62.3-65.8) n=3 | 5.96 (5.94-6.36) n=3 | 13.23 (12.63-13.82) n=2 | 14.1 (13.8-14.1) n=3 | 3/3 |
| prompt-16k | 1829.0 (1825.7-2058.6) n=3 | 60.9 (60.3-69.9) n=3 | 8.99 (7.99-9.00) n=3 | 15.05 (13.85-16.25) n=2 | 16.5 (16.3-17.4) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 2 | 7 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 6 | 6 | 0 | 0 |

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
| 1 | 25.883177 | 1915 | 73.986281 | 0 |
| 2 | 24.194874 | 1952 | 80.678246 | 0 |
| 3 | 23.975851 | 1846 | 76.994139 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
