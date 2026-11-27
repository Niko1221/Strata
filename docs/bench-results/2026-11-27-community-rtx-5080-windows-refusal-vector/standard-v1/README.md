# Strata standard benchmark: NVIDIA GeForce RTX 5080

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261127T192650Z-6bf0b409`, started 2026-11-27T19:26:50Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

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
- Track: **custom** (single-variable A/B against the base IQ3_XXS preset: identical pack, kv q4_0, kv-resident 32768, spec 6, dflash2 selq4, 262144 context; the only addition is the control vector distributed as experimental-speed-projection (documented by its authors as a refusal-direction projection, layers 4-44, cvec-mode project, per-layer); 4x20k-token warm-up with unrelated text before the run)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 26 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 118.3 (112.3-118.4) n=3 | 0.39 (0.34-0.42) n=3 | 2.03 (1.62-2.69) n=3 | 4.7 (3.6-4.9) n=3 | 2/3 |
| prompt-4k | 2757.9 (2755.3-2768.5) n=3 | 104.3 (97.4-115.5) n=3 | 1.51 (1.49-1.52) n=3 | 5.04 (3.55-6.53) n=2 | 6.4 (5.9-6.8) n=3 | 3/3 |
| prompt-16k | 3703.9 (3681.6-3708.0) n=3 | 102.2 (98.5-106.0) n=3 | 4.47 (4.47-4.49) n=3 | not measured | 9.5 (9.3-9.7) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 1 | 8 | 0 | 0 |
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
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 5 request(s) before this run.
