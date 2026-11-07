# Strata standard benchmark: AMD Radeon AI PRO R9700, AMD Radeon RX 6800 XT

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261107T085746Z-57fe40fd`, started 2026-11-07T08:57:46Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon AI PRO R9700 (31.9 GiB dedicated); AMD Radeon RX 6800 XT (16.0 GiB dedicated)
- CPU: AMD Ryzen 7 7700X 8-Core Processor; RAM: 47.5 GiB
- Storage of the model files: ssd WD_BLACK SN850X 4000GB (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 32.0.32015.7029; runtime: not measured
- Strata commit: `508227b52fc179c9a001b8c72cfe589359b8e39a`; engine 0.1.57; source build
- Background work: none

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-iq3_xxs; quantization iq3_xxs; revision 3f3ba9c; source revision not measured
- Files: `Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf` `9d5644b371ffac92...`; `Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00004.gguf` `aacaff3bdb9032c2...`; `Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00004.gguf` `ef8f14e790ab471b...`; `Qwen3.8-Flash-Next-UD-IQ3_XXS-00004-of-00004.gguf` `5de8b5dbf89409ac...`; `expert-profile.bin` `81f5dd7fe968a515...`
- Track: **custom** (engine 0.1.57 HIP self-built multi-arch gfx1030+gfx1201 from source 508227b5; vram-reserve 1792/3400; prefill auto:16384; resident 20 GiB pageable; kv int8; MTP drafter; STRATA_SSD_GBPS=3.5)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 5 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 76.9 (72.5-81.5) n=3 | 1.24 (1.19-1.28) n=3 | 2.34 (2.23-2.53) n=3 | 6.1 (5.8-6.7) n=3 | 0/3 |
| prompt-4k | 696.3 (695.1-696.5) n=3 | 68.9 (68.8-71.4) n=3 | 5.91 (5.90-5.91) n=3 | 11.68 (11.37-12.00) n=2 | 13.3 (13.1-13.3) n=3 | 3/3 |
| prompt-16k | 1314.9 (1311.9-1320.0) n=3 | 71.8 (70.1-74.9) n=3 | 12.51 (12.47-12.53) n=3 | 17.97 (17.80-18.97) n=3 | 19.7 (19.3-19.8) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).

## Provenance and measurement notes

**Engine.** A self-built multi-architecture HIP engine (0.1.57), not an official release asset: official Windows HIP releases carry one AMD family each (`hip-rdna2`, `hip-rdna3`, `hip-rx9070`), so no single official download has code for both gfx1030 and gfx1201. Built from `508227b5` with a clean engine source tree; `engine/strata.exe` sha256 `687c62c1…6eef8ec`. See `provenance.json`.

**PR #4883 changes no inference code** (setup.py, docs and tools only): it is the setup path that accepts a `--gpus` split across AMD families and a self-built multi-arch engine. `software.build_kind: "source"` is correct, and `software.build_options` names the build.

**No baseline on the unmodified release is possible.** The stock engine cannot serve this configuration, so these are first measurements of a R9700 + RX 6800 XT split, not a speed delta.

**Weights.** `unsloth/Qwen3.8-Flash-Next-GGUF`, `UD-IQ3_XXS`, revision `3f3ba9c` ("Better UD 3.1 quants"); tokenizer from the same repo and revision. Corroborated by shard byte sizes matching the Hugging Face listing, not by a hash of the remote file. The original Qwen weights revision was not recorded and is null.


**Configuration under test, not defaults.** `--prefill auto:16384` (the planner chose chunk 15616), `--resident-budget-gib 20` with `STRATA_RESIDENT_PIN=0`, `--vram-reserve-mib 1792` / `--vram-reserve-later-mib 3400`, `--kv int8`, `STRATA_SSD_GBPS=3.5`, and the environment list in `configuration.experimental_options` are operator choices. `STRATA_SSD_GBPS=3.5` replaces the planner's assumed 1.50 GB/s with the measured sequential read of the model volume (4.00 GiB in 1.23 s); it changed throughput by less than the run-to-run spread. The engine logs its own warning that chunks of 8192 tokens or more have crashed Windows HIP on an RX 7900 XTX (#1630); no crash occurred in these runs, on this hardware, in three starts.


**Truncation.** The protocol caps first-answer output at 512 tokens. In the performance phase 6 of 9 requests truncated mid-reasoning in each run, so `first_answer_s` and `total_s` carry output-length variance. In run 3 all three `prompt-16k` requests truncated before any answer text, so that run's `first_answer_s` summary is null (`missing`: `/summaries/…/median`) and its median is taken over two runs.

**Skipped stage.** The optional concurrency stage was not run: the server is `parallel: 1` with no `--batch`, so it cannot be executed for this configuration.

**Fresh engine per run.** Each run was started by the runner (`extensions.started_engine: true`) and the server had answered no request before it (`extensions.server_requests_before_run: 0`).
