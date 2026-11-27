# Strata standard benchmark: 2x MI25 (gfx900)

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261125T082815Z-84454468`, started 2026-11-25T08:28:15Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl). The benchmark started the engine for this run. Its startup and per-request lines are in [engine.log](engine.log). The effective arguments are in [config.json](config.json).

The runner's hardware probe recorded the display card. `results.json` still says Quadro P2000, driver 580.178.04, and CUDA 13.0. Those are the NVIDIA display GPU. The model ran on the two AMD cards below. `engine.log` names GPU 0 as AMD Radeon Pro WX 9100 and both HIP devices as gfx900.

## Hardware and software

- Inference GPUs: two flashed AMD Instinct MI25 / Radeon Pro WX 9100, gfx900, 16368 MiB each (sysfs `mem_info_vram_total` 17163091968 bytes). PCI 0000:0a:00.0 and 0000:07:00.0. Each endpoint link measured at 8.0 GT/s x16. Power cap 170 W on each. Layer split 27 (layers 0–26 on HIP 0, 27–47 on HIP 1). The Quadro P2000 (5 GiB, 75 W) is the display card only.
- CPU: Intel(R) Xeon(R) CPU E5-2690 v3 @ 2.60GHz; RAM: 125.7 GiB
- Storage of the model files: nvme PCIe SSD (nvme)
- OS: Linux Mint 22.3 7.0.0-44-generic; the probe's driver and runtime lines are the NVIDIA display stack, not the HIP engine. Engine 0.1.59.1 is a gfx900 source build (`STRATA_HIP_GFX900=ON`).
- Strata commit: `94517b9ab9e7984fc9cdc24df8f33114009ad010`; engine 0.1.59.1; source build
- Background work: Cinnamon on X11, composited by the Quadro P2000 display GPU; ollama serve is running and is not holding the MI25s

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (jonathanmajh/Strata gfx900-mi25 source build, engine 0.1.59.1, not an upstream release. Dual flashed MI25 layer-split 27, MTP spec 4, int8 KV, ple-io ram, conversation cache 0, vision projector loaded but this suite is text. Runtime: STRATA_EXPERIMENTAL_GFX900=1, STRATA_STAGE_TRIM=1, STRATA_SPLIT_OVERLAP=1, STRATA_HIP_PROMPT_F16=1, STRATA_GFX906_ATTN_QUERY_SWIZZLE=1, STRATA_GFX906_ATTN_REDUCE12=1, STRATA_PREFILL_CPU_SHARE=auto, STRATA_ARENA_MMAP=0. Expert cache prefilled from the startup profile. The benchmark starts the engine for this run.)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 8 min 59 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 32.0 (30.0-32.0) n=3 | 2.29 (2.27-2.29) n=3 | 5.97 (4.50-11.30) n=3 | 18.2 (17.8-19.0) n=3 | 1/3 |
| prompt-4k | 240.7 (239.8-242.9) n=3 | 29.8 (28.3-33.0) n=3 | 17.06 (16.90-17.12) n=3 | 24.90 (23.66-26.15) n=2 | 34.0 (32.6-35.1) n=3 | 3/3 |
| prompt-16k | 355.3 (354.8-356.7) n=3 | 27.8 (27.3-29.3) n=3 | 46.19 (46.01-46.24) n=3 | 56.85 (56.85-56.85) n=1 | 64.6 (63.4-64.9) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
