# Strata standard benchmark: AMD Radeon AI PRO R9700, AMD Radeon RX 6800 XT

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261108T062036Z-09cdc36f`, started 2026-11-08T06:20:36Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon AI PRO R9700 (31.9 GiB dedicated); AMD Radeon RX 6800 XT (16.0 GiB dedicated)
- CPU: AMD Ryzen 7 7700X 8-Core Processor; RAM: 47.5 GiB
- Storage of the model files: ssd WD_BLACK SN850X 4000GB (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 32.0.32015.7029; runtime: not measured
- Strata commit: `588b9d341155688fbe23bbcfc81b0bf2b096fed1`; engine 0.1.58; source build
- Background work: none

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-iq4_xs-v31; quantization iq4_xs; revision 814db05; source revision not measured
- Files: `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf` `93ff23a08e5cd750...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00004.gguf` `13d5dc8b93d039e4...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00004.gguf` `f6b9b5934908a204...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00004-of-00004.gguf` `5de8b5dbf89409ac...`; `expert-profile.bin` `a29187309336d2ce...`
- Track: **custom** (engine 0.1.58 HIP self-built multi-arch gfx1030+gfx1201 from source 588b9d34 (includes PR 5562, fused-layout shrink fix); vram-reserve 1792/3400; prefill auto:16384; resident 18 GiB pageable; kv k8v4; MTP spec 4; STRATA_SSD_GBPS=3.5; STRATA_PF_FUSED=1; STRATA_PF_FUSED_Q50=1)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 4 min 16 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 56.6 (56.4-59.5) n=3 | 1.93 (1.93-1.95) n=3 | 3.28 (3.07-3.40) n=3 | 8.0 (7.2-9.2) n=3 | 0/3 |
| prompt-4k | 535.1 (531.3-536.5) n=3 | 60.7 (59.8-63.0) n=3 | 7.69 (7.68-7.73) n=3 | 12.61 (11.66-12.84) n=3 | 16.1 (15.8-16.2) n=3 | 3/3 |
| prompt-16k | 864.6 (861.0-866.4) n=3 | 63.0 (57.7-64.3) n=3 | 19.01 (18.96-19.07) n=3 | 24.01 (23.45-24.56) n=2 | 27.1 (27.0-27.8) n=3 | 3/3 |

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
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
