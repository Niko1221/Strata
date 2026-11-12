# Strata standard benchmark: AMD Radeon AI PRO R9700, AMD Radeon RX 6800 XT

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261108T062709Z-921655c2`, started 2026-11-08T06:27:09Z. **Status: complete.**

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

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 4 min 19 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 56.3 (55.7-60.8) n=3 | 2.01 (1.90-2.03) n=3 | 3.20 (3.09-3.26) n=3 | 7.4 (7.1-8.2) n=3 | 0/3 |
| prompt-4k | 533.1 (532.8-539.7) n=3 | 57.3 (57.2-58.9) n=3 | 7.72 (7.63-7.72) n=3 | 14.47 (14.35-14.60) n=2 | 16.5 (16.4-16.7) n=3 | 3/3 |
| prompt-16k | 866.9 (863.8-867.1) n=3 | 58.9 (57.4-63.6) n=3 | 18.95 (18.94-19.01) n=3 | 21.66 (21.66-21.66) n=1 | 27.6 (27.0-27.8) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
