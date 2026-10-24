# Strata standard benchmark: AMD Radeon

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261021T182459Z-74ba69f0`, started 2026-10-21T18:24:59Z. **Status: complete.**

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
- Track: **custom** (Strata 0.2.2 (011c9e60) source build. Radeon AI PRO R9700 (gfx1201, OCuLink PCIe 4.0 x4, 7.2 GB/s measured) alone, the Strix Halo 8060S hidden (ROCR_VISIBLE_DEVICES=0); the whole expert arena in the Strix Halo's RAM (no --mmap-experts, cudaHostRegister PORTABLE), --expert-cache auto, --prefill 32768, --spec 4 --spec-min-p 0.5, MTP, --kv int8, --max-context 262144, --vision (CPU encoder); env STRATA_PF_FUSED=1 STRATA_PF_GEMM=1 STRATA_PA_FAST=1 STRATA_HIP_WMMA=1 STRATA_SELECT_WMMA=1; one request at a time (no --batch), --kv-grow (whole K/V in VRAM, grows), STRATA_PREFILL_CPU_SHARE=auto)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 20 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 82.9 (81.2-85.0) n=3 | 0.56 (0.48-0.59) n=3 | 1.44 (1.35-4.33) n=3 | 5.9 (4.4-6.9) n=3 | 1/3 |
| prompt-4k | 965.4 (917.8-992.9) n=3 | 78.1 (66.5-83.7) n=3 | 4.26 (4.14-4.48) n=3 | 8.09 (7.58-8.60) n=2 | 10.8 (10.6-11.8) n=3 | 3/3 |
| prompt-16k | 2166.8 (2004.9-2267.2) n=3 | 74.7 (68.4-85.8) n=3 | 7.59 (7.25-8.20) n=3 | 10.78 (9.94-11.62) n=2 | 14.1 (13.5-15.7) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 2 | 7 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 3 | 9 | 0 | 0 |

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
| 1 | 26.496568 | 1904 | 71.858363 | 0 |
| 2 | 27.189631 | 2030 | 74.660815 | 0 |
| 3 | 25.08586 | 1887 | 75.221659 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
