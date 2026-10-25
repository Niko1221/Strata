# Strata standard benchmark: Tesla P100-PCIE-16GB

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261025T125922Z-4e220a2a`, started 2026-10-25T12:59:22Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: Tesla P100-PCIE-16GB (16.0 GiB dedicated)
- CPU: AMD Ryzen 5 5600G with Radeon Graphics; RAM: 93.7 GiB
- Storage of the model files: nvme PC SN730 NVMe WDC 512GB (nvme)
- OS: Linux Mint 22.3 6.8.0-147-generic; drivers: gpu-driver 580.196.07; runtime: cuda 13.0
  - Clarification, added by hand (this file is not one of the hashed artifacts): the `runtime: cuda 13.0` above is the CUDA API level the R580 driver reports as available, not the toolkit these measurements ran on. The engine binary links `libcudart.so.12` from `/usr/local/cuda-12.9` (CUDA SDK 12.9.2), and the build's `BUILD.json` records `toolkit: 12`, `archs: [60]`. Pascal sm_60 is not a CUDA 13 target, which is why 12.9 is pinned here.
- Strata commit: `7dd5cf18e68c639a8773e096c5c79f971ea2a9f5`; engine 0.1.49.2; source build
  - Build options (null in `software.build_options` because the server does not report them for a source build): CUDA 12.9.2, architecture sm_60 only, vision companion built for CPU
- Background work: Tesla P100 16 GB sola, Ryzen 5 5600G (6 cores), 93.7 GiB RAM, NVMe nvme0n1; motor levantado por el runner; sin otro trafico durante el benchmark

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-q4_k_xl; quantization UD-Q4_K_XL; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` `4448186216b3af4c...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf` `3f342f1c1580473f...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf` `56758f40269cad5c...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf` `753bda48b98ba4f1...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (build desde fuente v0.2.4.2-6-g7dd5cf18 (motor 0.1.49.2, sm_60); args restaurados a mano a la config del 18-oct: --pcie-frac 0.20 --spec-min-p 0.70 --vram-reserve-mib 700 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97; sin --ple-io ni --resident-budget-gib)
- Context 256000 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 9 min 7 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 34.2 (31.6-34.3) n=3 | 1.52 (1.36-1.54) n=3 | 5.16 (4.49-5.88) n=3 | 13.3 (12.2-13.8) n=3 | 0/3 |
| prompt-4k | 265.6 (259.0-276.6) n=3 | 30.7 (29.7-31.3) n=3 | 15.48 (14.89-15.85) n=3 | 28.21 (28.21-28.21) n=1 | 32.5 (31.2-32.7) n=3 | 3/3 |
| prompt-16k | 299.1 (291.5-311.4) n=3 | 30.2 (29.6-31.2) n=3 | 54.84 (52.69-56.28) n=3 | 69.96 (69.96-69.96) n=1 | 71.2 (69.6-73.6) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
