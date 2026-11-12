# Strata standard benchmark: Orin (nvgpu)

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261112T095550Z-79146be9`, started 2026-11-12T09:55:50Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: Orin (nvgpu)
- CPU: ARMv8 Processor rev 1 (v8l); RAM: 61.4 GiB
- Storage of the model files: nvme Samsung SSD 970 EVO 500GB (nvme)
- OS: <host> 22.04.5 LTS 5.15.185-tegra; drivers: gpu-driver 540.5.0; runtime: cuda 12.6
- Strata commit: `13eedf552417a6a791a7d13c6597e24300ee3a09`; engine 0.1.59.1; release build
- Background work: none

## Model and configuration

- Model: Qwen3.8-Flash-Next-IQ3_XXS; quantization IQ3_XXS; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `219ea929900dfa9e...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (Custom production profile on Jetson AGX Orin 64GB (nvgpu, CUDA 12.6, 12-core armv8.2-a+dotprod+fp16+i8mm): IQ3_XXS quant + custom pack + expert profile; prefill auto:16384, prompt-cache 4 + tail, conversation-cache 2048MiB (min-free 1024MiB), expert-cache 4000, kv int8, vram-reserve 4096MiB, pool-workers 3, MTP drafter t=4 min-p 0.5, ple-io mmap, max-context 131072; engine 0.1.59.1 built from upstream main 13eedf55 with the vendored strata-ggml)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 7 min 6 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 22.2 (20.2-24.0) n=3 | 1.94 (1.86-1.98) n=3 | 6.70 (4.41-9.41) n=3 | 16.5 (15.2-22.5) n=3 | 0/3 |
| prompt-4k | 517.2 (511.9-521.6) n=3 | 22.1 (21.1-22.3) n=3 | 7.98 (7.92-8.07) n=3 | 18.73 (17.61-22.94) n=3 | 31.2 (30.9-32.2) n=3 | 3/3 |
| prompt-16k | 668.4 (664.7-668.8) n=3 | 20.3 (19.9-21.8) n=3 | 24.59 (24.58-24.73) n=3 | 43.79 (39.43-48.15) n=2 | 49.8 (48.1-50.3) n=3 | 3/3 |

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
| 1 | 106.83587 | 2048 | 19.169592 | 0 |
| 2 | 105.009924 | 2048 | 19.502919 | 0 |
| 3 | 90.248381 | 1807 | 20.02252 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c3: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r2-c3: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r2-c4: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
