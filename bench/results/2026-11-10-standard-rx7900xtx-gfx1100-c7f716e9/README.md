# Strata standard benchmark: AMD Radeon

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261110T023522Z-c7f716e9`, started 2026-11-10T02:35:22Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon
- CPU: AMD Ryzen 9 5900X 12-Core Processor; RAM: 62.7 GiB
- Storage of the model files: ssd Samsung SSD 870 EVO 1TB (sata)
- OS: Debian GNU/Linux 13 (trixie) 7.2.12+deb13-amd64; drivers: not measured; runtime: not measured
- Strata commit: `7df6538d6c886093df39d74b36ed0a3396bf9467`; engine 0.1.59; source build
- Background work: not measured

## Model and configuration

- Model: swift-1.5-iq3_s; quantization IQ3_S; revision not measured; source revision not measured
- Files: `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `46cb3996cd1a17fe...`; `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `70ce6fb48a94fe34...`
- Track: **custom** (no notes given by the operator)
- Context 262144 tokens; server capacity 2 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 4 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 76.8 (76.2-83.6) n=3 | 0.96 (0.86-1.00) n=3 | 2.41 (1.74-2.74) n=3 | 5.0 (4.8-5.5) n=3 | 0/3 |
| prompt-4k | 1309.3 (1274.2-1311.2) n=3 | 84.1 (81.0-86.9) n=3 | 3.17 (3.15-3.24) n=3 | 5.42 (5.30-6.94) n=3 | 9.3 (9.0-9.4) n=3 | 3/3 |
| prompt-16k | 2939.7 (2937.2-2943.9) n=3 | 73.9 (70.0-85.6) n=3 | 5.61 (5.60-5.61) n=3 | 10.27 (8.54-12.00) n=2 | 12.5 (11.6-12.9) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).

## Provenance added by the operator (sanitized)

The runner could not read the GPU model (the server was already running); this section fills it in. Paths are redacted.

- GPU: AMD Radeon RX 7900 XTX, 24 GiB (gfx1100), PCIe 4.0 x16; three active display outputs.
- CPU / RAM / OS: as in the header above.
- Model, launch arguments and tuning: `run-config-sanitized.json` here.
- Vision: off - the `vision` section is removed from the run config.
- Commit nuance: the official component's `BUILD.json` records source `f-0213-rc 712fec6b4297`, while the source
  build is tag `v0.2.13` = `d2f0964`. Comparing those two commits, only `packaging/make_model_release.py` and
  `tests/terminal/test_make_model_release.py` differ; the engine sources (`src/`, `include/`, `cmake/` and the
  ggml tree) are identical, and both binaries print `strata 0.1.59`.
- Sysctls on the box: `vm.min_free_kbytes=524288`, `vm.watermark_scale_factor=100`.
- Engine: the **official `0.1.59-linux-x86_64-hip-rx7900` component from the signed feed** (release v0.2.13). It is
  portable and loads its own HIP stack from the component directory (`RUNPATH=$ORIGIN`; `BUILD.json`:
  `portable: true`, `hipblaslt_version: 100200`); the system ROCm install is not used by it.
- Tuning environment: `STRATA_SELECT_WMMA=1`, `STRATA_SCORES_THREAD=0`, `GPU_MAX_HW_QUEUES=4`,
  `STRATA_PREFILL_AUTO_MAX=32768`, `STRATA_PREFILL_RING=192`, `STRATA_HIPBLASLT_TUNING` pointing at the 100200 table.
- This run and the ROCm 10.1 run in the sibling directory were made under identical conditions (same launch
  arguments, same tuning environment, vision off, same sysctls); only the engine binary differs.
