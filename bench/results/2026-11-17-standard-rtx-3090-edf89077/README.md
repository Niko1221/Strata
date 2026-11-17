# Strata standard benchmark: NVIDIA GeForce RTX 3090

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261117T210535Z-edf89077`, started 2026-11-17T21:05:35Z. **Status: complete.**

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
- Track: **custom** (CachyOS Linux, kernel 7.3.4-1-cachyos. Changed from setup's config: --resident-budget-gib 52 (setup wrote 40), --aux-cpus auto (added), env STRATA_IQ4_W4A8=1, STRATA_ENGINE_NICE=0, STRATA_ROUTE_PRIOR=0 (the default, set explicitly). System: 64 GB DDR5-6000 (2 x 32 GB, as configured), RTX 3090 power limit 280 W (lowered from the 350 W default), gpu-unlock 2.0.0 loaded (third-party kernel module that lifts the GeForce half-rate cap on FP16/BF16/TF32 tensor maths; added after the run), /sys/kernel/mm/lru_gen/min_ttl_ms=0 (CachyOS default 100; 0 stops early OOM kills of the pinned engine), swap off, systemd-oomd off, vm.swappiness=150, a minimal self-made desktop (about 50 MB of RAM), desktop on the Radeon iGPU (RTX 3090 drives no display).)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 59 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 89.4 (85.4-89.6) n=3 | 0.47 (0.41-0.50) n=3 | 1.39 (1.26-1.48) n=3 | 3.6 (3.3-4.1) n=3 | 0/3 |
| prompt-4k | 1796.2 (1792.0-1811.5) n=3 | 89.6 (81.6-92.9) n=3 | 2.30 (2.28-2.30) n=3 | 5.61 (4.43-7.37) n=3 | 8.0 (7.8-8.5) n=3 | 3/3 |
| prompt-16k | 3260.4 (3242.1-3267.4) n=3 | 80.4 (79.8-85.6) n=3 | 5.08 (5.04-5.15) n=3 | 8.91 (8.91-8.91) n=1 | 11.4 (11.0-11.5) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 0 | 0 | 0 | 0 | 0 |

Sanity checks: **11 of 12**.

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
| T1 | fail | expected one structured tool call, got 0 |
| T2 | pass | the expected tool call |
| T3 | pass | the expected tool call |
| T4 | pass | the expected tool call |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).

## Notes added after the run

The measurements and outputs are unchanged. Two facts in this README and in `results.json` were corrected by hand after
the run; nothing else was edited.

- **Background work.** The run's settings listed a Claude Code terminal session as running. It was not: the PC was
  restarted before the run, the run took 21:05:35 to about 21:07 UTC, and the Claude Code session started at 21:08:07.
  It is removed from the background list. The run's settings also listed ananicy-cpp, a priority service CachyOS runs by
  default; it is removed as part of the stock OS, and none of its rules names Strata or Python.
- **gpu-unlock.** [gpu-unlock](https://github.com/Zagitta/gpu-unlock) 2.0.0 (packages `gpu-unlock-dkms` and
  `gpu-unlock-firmware`) was installed and its `gpu_unlock` kernel module was loaded from boot. It lifts the GeForce
  half-rate cap on FP16, BF16 and TF32 tensor maths with FP32 accumulation. It was not in the run's notes and is added to
  the track notes. Its effect on these numbers was not measured: there is no run on this PC without it.
- **T1 (the failed sanity check).** The request finished normally (HTTP 200, `finish_reason` stop, 1,577 tokens at
  91.4 tok/s; no engine error in `engine.log`). The model's reasoning said it had to call `lookup_record`, then guessed
  at a tool result that did not exist, and it answered with an invented `{"id":"A-17","status":"found"}` and no tool call
  (`outputs.jsonl`, `sanity-T1`). T2-T4, the other tool calls, passed. The case was run once at temperature 1.0, as the
  protocol says, so this run cannot tell a rare sampling slip from a weakness of this quantization.
- **A second run.** The same PC and settings were run again as `2026-11-17-standard-rtx-3090-7f8b61f6`, 26 minutes
  later. It was faster (short decode 100.0 against 89.4 tok/s, 16K prompt 3,327 against 3,260 tok/s) and passed 12 of
  12. Speculative draft acceptance was about the same (78% and 79% on the speed requests), so drafting does not
  explain the gap. One difference is known: this run followed a 16-second reboot, the second a cold start after about
  2 minutes powered off, and at the two boots the gpu-unlock module found different values in the GPU's tensor
  speed-selector registers, the ones it exists to change (`ss0=0x0 ss1=0x0` here, `ss0=0x35001117 ss1=0x7` there; its
  own writes were the same and succeeded both times). Whether the selectors ended at the same rate is not known. Whether that, temperature, or ordinary
  variation between engine starts caused the gap is not known: temperatures and clocks were not recorded, and there are
  only two runs.
