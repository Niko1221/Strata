# Strata standard benchmark: AMD Radeon AI PRO R9700

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261021T182914Z-103ddfcf`, started 2026-10-21T18:29:14Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon AI PRO R9700 (31.9 GiB dedicated)
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor; RAM: 61.7 GiB
- Storage of the model files: nvme WD_BLACK SN8100 4000GB (nvme)
- OS: CachyOS 7.3.0-rc6-2-cachyos-rc; drivers: not measured; runtime: rocm 10.1.0
- Strata commit: `011c9e600a93bbc9fd451c563501b5831c32bbca`; engine 0.1.47; source build
- Background work: KDE Plasma Wayland compositor on the iGPU 0000:7c:00.0 (monitor on card2 HDMI); the R9700 0000:03:00.0 has no display output connected; machine otherwise idle during the suite: no compiles, downloads or other GPU users; model and packs on a WD_BLACK SN8100 4 TB NVMe (btrfs); PCIe gen5 x8, path 4/4, power limit 330 W

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S (Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S, 2 shards, 54817524224 + 28800138432 bytes); revision not measured; source revision de4b8e4d43b917e7706784d8bb445c9af86a3540
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (Locally tuned single-GPU Linux/HIP setup: calibrated --pool-workers 10, --vram-reserve-mib 1024 (setup default 700), opt-in env STRATA_HIPBLASLT_TUNING (30-row gfx1201/ROCm 10.1 table), STRATA_HIP_WMMA=1, STRATA_PF_FUSED=1, STRATA_GFX12_CHUNK=1 (prefill chunks 16384), STRATA_ROUTE_TAIL_SKIP=1, --spec 4 --spec-min-p 0.5, --mtp rt, Huihui refusal-direction control vector (project, layers 4-44), and the runner itself patched locally with the amd-smi hardware probe (PR #3519); engine 0.1.47 release binary, not a source build.)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 34 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 90.4 (89.7-90.6) n=3 | 0.35 (0.33-0.53) n=3 | 1.31 (1.29-1.39) n=3 | 3.3 (3.0-4.0) n=3 | 0/3 |
| prompt-4k | 2412.8 (2407.1-2419.3) n=3 | 90.0 (88.4-93.4) n=3 | 1.72 (1.71-1.72) n=3 | 5.18 (4.32-5.22) n=3 | 7.4 (7.2-7.5) n=3 | 3/3 |
| prompt-16k | 3430.3 (3421.4-3444.0) n=3 | 86.4 (75.6-94.0) n=3 | 4.80 (4.78-4.81) n=3 | 7.58 (7.32-7.84) n=2 | 10.7 (10.2-11.6) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 3 | 9 | 0 | 0 |

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
| T1 | pass | the expected tool call |
| T2 | pass | the expected tool call |
| T3 | fail | expected one structured tool call, got 0 |
| T4 | pass | the expected tool call |

Concurrency rounds (four requests released together):

| Round | Wall s | Output tokens | Aggregate tok/s | Failed |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 25.822987 | 2048 | 79.309183 | 0 |
| 2 | 21.894333 | 1882 | 85.958316 | 0 |
| 3 | 22.781496 | 2004 | 87.966128 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c2: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r1-c4: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r2-c4: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).

### Operator notes added before submitting

- `software.build_kind` reads `source` because the runner infers it from the presence of a `.git` folder in the
  Strata checkout (`strata_cli/bench_facts.py:433`: `"source" if (ROOT / ".git").exists() else "release"`). The
  engine here is the **release binary that setup downloads** (Strata 0.2.2, engine 0.1.47), not a source build, so
  the `source build` label above is the tool's inference and is wrong for this run; the code reading is confirmed
  static analysis, the wrong label is a measured artifact of this report.
- Two values the runner left `null` that this machine does report, measured separately on the same machine and
  configuration at 20:18 from the server's `/metrics` (not part of `results.json`): GPU **power limit 330 W**
  (323 W instantaneous at 48 °C) and **PCIe gen5 x8 (max gen5, path 4/4)**.
- `/software/drivers` stays null: `amd-smi static --driver` answers `N/A` for the Radeon AI PRO R9700 on
  ROCm 10.1 (amdgpu from kernel 7.3.0-rc6-2-cachyos-rc). ROCm 10.1.0 is in `software.runtime`.
- `/model/revision` and `/model/tokenizer_revision` are null: the download metadata kept here (the `.done` files,
  which only say `whole (checked against its own tensor directory)`) records no repository revision. The
  source-model revision is pinned in `/model/source_revision`.
- The runner carried the local patch from PR #3519, the `amd-smi` GPU and storage probe. Without it
  `hardware.gpus[0].name` reads the server's vendor stub and the folder is named `unknown`; the names and
  `memory_bytes` in `hardware` come from that probe. No engine code was changed for this run.
