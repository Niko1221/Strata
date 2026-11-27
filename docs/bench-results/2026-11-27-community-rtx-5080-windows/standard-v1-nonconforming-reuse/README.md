# Strata standard benchmark: NVIDIA GeForce RTX 5080

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261127T173818Z-665630ef`, started 2026-11-27T17:38:18Z. **Status: nonconforming.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

This run is not a complete standard result:

- unexpected reuse of the user payload in performance-prompt-4k-r1
- unexpected reuse of the user payload in performance-prompt-16k-r1
- unexpected reuse of the user payload in performance-prompt-4k-r2
- unexpected reuse of the user payload in performance-prompt-16k-r2
- unexpected reuse of the user payload in performance-prompt-4k-r3
- unexpected reuse of the user payload in performance-prompt-16k-r3

## Hardware and software

- GPU: NVIDIA GeForce RTX 5080 (15.9 GiB dedicated)
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor; RAM: 61.4 GiB
- Storage of the model files: ssd Samsung SSD 9100 PRO 2TB (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 621.14; runtime: cuda 13.5
- Strata commit: `64524fb9fc82222ddce554c0643b64f056ec8366`; engine 0.1.62; release build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs; quantization IQ3_XXS; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `219ea929900dfa9e...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `draft_vocab.bin` `b1e1d3a7a9e4bf86...`; `expert-profile.bin` `8f59b4aa8873209d...`; `qwen3.8-flash-next-dflash2.q4_0.pack` `cda40d62d0eb8f37...`
- Track: **custom** (preset Qwen 256K: kv q4_0, kv-resident 32768, conversation-cache-mib 4096, vram-reserve-mib 1536, draft dflash2 spec 6 conf 0.7, vision cpu)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 0 min 53 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 136.5 (117.5-136.6) n=3 | 0.34 (0.34-0.36) n=3 | 0.98 (0.84-1.08) n=3 | 3.4 (3.1-4.1) n=3 | 1/3 |
| prompt-4k | not measured | not measured | not measured | not measured | not measured | 3/3 |
| prompt-16k | not measured | not measured | not measured | not measured | not measured | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 2 | 7 | 0 | 0 |
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
- Excluded from speed summaries: performance-prompt-16k-r1: unexpected cache reuse: 16382 cached tokens exceed the 55-token shared template prefix
- Excluded from speed summaries: performance-prompt-16k-r2: unexpected cache reuse: 16382 cached tokens exceed the 55-token shared template prefix
- Excluded from speed summaries: performance-prompt-16k-r3: unexpected cache reuse: 16382 cached tokens exceed the 55-token shared template prefix
- Excluded from speed summaries: performance-prompt-4k-r1: unexpected cache reuse: 4088 cached tokens exceed the 54-token shared template prefix
- Excluded from speed summaries: performance-prompt-4k-r2: unexpected cache reuse: 4088 cached tokens exceed the 54-token shared template prefix
- Excluded from speed summaries: performance-prompt-4k-r3: unexpected cache reuse: 4088 cached tokens exceed the 54-token shared template prefix
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 23 request(s) before this run.
