# Strata standard benchmark: 2x MI25 (gfx900)

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261125T070417Z-5f930e87`, started 2026-11-25T07:04:17Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

The runner's hardware probe recorded the display card. `results.json` still says Quadro P2000, driver 580.178.04, and CUDA 13.0. Those are the NVIDIA display GPU. The model ran on the two AMD cards below. Their `product_name` sysfs files are empty, which is why the probe kept the Quadro.

## Hardware and software

- Inference GPUs: two flashed AMD Instinct MI25 / Radeon Pro WX 9100, gfx900, 16368 MiB each (sysfs `mem_info_vram_total` 17163091968 bytes). PCI 0000:0a:00.0 and 0000:07:00.0. Each endpoint link measured at 8.0 GT/s x16. Power cap 170 W on each. Layer split 27. The Quadro P2000 (5 GiB, 75 W) is the display card only.
- During this run the hotter card reached 69°C junction and 61°C HBM. The engine process did not swap.
- CPU: Intel(R) Xeon(R) CPU E5-2690 v3 @ 2.60GHz; RAM: 125.7 GiB
- Storage of the model files: nvme PCIe SSD (nvme)
- OS: Linux Mint 22.3 7.0.0-44-generic; the probe's driver and runtime lines are the NVIDIA display stack, not the HIP engine. Engine 0.1.59.1 is a gfx900 source build (`STRATA_HIP_GFX900=ON`).
- Strata commit: `94517b9ab9e7984fc9cdc24df8f33114009ad010`; engine 0.1.59.1; source build
- Background work: Cinnamon on X11, composited by the Quadro P2000 display GPU; ollama serve is running and is not holding the MI25s

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (jonathanmajh/Strata gfx900-mi25 source build, engine 0.1.59.1, not an upstream release. Dual flashed MI25 layer-split 27, MTP spec 4, int8 KV, ple-io ram, conversation cache 0, vision projector loaded but this suite is text. Runtime: STRATA_EXPERIMENTAL_GFX900=1, STRATA_STAGE_TRIM=1, STRATA_SPLIT_OVERLAP=1, STRATA_HIP_PROMPT_F16=1, STRATA_GFX906_ATTN_QUERY_SWIZZLE=1, STRATA_GFX906_ATTN_REDUCE12=1, STRATA_PREFILL_CPU_SHARE=auto, STRATA_ARENA_MMAP=0. Expert cache prefilled from the startup profile. The server had answered 0 requests since load.)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 8 min 47 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 30.8 (29.9-33.2) n=3 | 2.29 (2.28-2.30) n=3 | 5.36 (5.29-9.04) n=3 | 16.0 (12.1-16.7) n=3 | 0/3 |
| prompt-4k | 240.3 (240.1-242.4) n=3 | 28.3 (27.7-29.8) n=3 | 17.08 (16.94-17.10) n=3 | 25.93 (25.93-25.93) n=1 | 35.2 (34.1-35.6) n=3 | 3/3 |
| prompt-16k | 355.6 (354.7-357.0) n=3 | 29.7 (29.7-30.8) n=3 | 46.15 (45.97-46.27) n=3 | 57.03 (55.00-58.44) n=3 | 63.2 (62.8-63.4) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
