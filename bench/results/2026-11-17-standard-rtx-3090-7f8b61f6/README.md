# Strata standard benchmark: NVIDIA GeForce RTX 3090

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261117T213134Z-7f8b61f6`, started 2026-11-17T21:31:34Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3090 (24.0 GiB dedicated)
- CPU: AMD Ryzen 5 7600 6-Core Processor; RAM: 61.9 GiB
- Storage of the model files: nvme CT1000T500SSD8 (nvme)
- OS: CachyOS 7.3.4-1-cachyos; drivers: gpu-driver 620.70.06; runtime: cuda 13.5
- Strata commit: `c77572806996349b49995e4f98d41b5294d36a21`; engine 0.1.61; source build
- Background work: minimal self-made desktop session (about 50 MB of RAM)

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-iq4_xs-v31; quantization UD-IQ4_XS; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf` `93ff23a08e5cd750...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00004.gguf` `13d5dc8b93d039e4...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00004.gguf` `f6b9b5934908a204...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00004-of-00004.gguf` `5de8b5dbf89409ac...`; `draft_vocab.bin` `b1e1d3a7a9e4bf86...`; `expert-profile.bin` `8f59b4aa8873209d...`; `qwen3.8-flash-next-dflash2.q4_0.pack` `cda40d62d0eb8f37...`
- Track: **custom** (CachyOS Linux, kernel 7.3.4-1-cachyos. Changed from setup's config: --resident-budget-gib 52 (setup wrote 40), --aux-cpus auto (added), env STRATA_IQ4_W4A8=1, STRATA_ENGINE_NICE=0, STRATA_ROUTE_PRIOR=0 (the default, set explicitly). System: 64 GB DDR5-6000 (2 x 32 GB, as configured), RTX 3090 power limit 280 W (lowered from the 350 W default), gpu-unlock 2.0.0 loaded (third-party kernel module that lifts the GeForce half-rate cap on FP16/BF16/TF32 tensor maths), /sys/kernel/mm/lru_gen/min_ttl_ms=0 (CachyOS default 100; 0 stops early OOM kills of the pinned engine), swap off, systemd-oomd off, vm.swappiness=150, a minimal self-made desktop (about 50 MB of RAM), desktop on the Radeon iGPU (RTX 3090 drives no display).)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 39 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 100.0 (87.7-105.9) n=3 | 0.37 (0.33-0.37) n=3 | 1.13 (1.10-1.93) n=3 | 4.2 (3.5-4.3) n=3 | 0/3 |
| prompt-4k | 1857.1 (1853.2-1862.2) n=3 | 96.0 (83.3-101.6) n=3 | 2.22 (2.22-2.23) n=3 | 5.64 (3.91-7.36) n=3 | 7.5 (7.2-8.4) n=3 | 3/3 |
| prompt-16k | 3326.7 (3323.0-3329.4) n=3 | 87.7 (85.5-92.6) n=3 | 4.95 (4.95-4.95) n=3 | 9.28 (8.32-9.97) n=3 | 10.8 (10.5-10.9) n=3 | 3/3 |

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
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
