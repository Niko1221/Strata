# Strata standard benchmark: AMD Radeon

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261110T000221Z-d45cfc1e`, started 2026-11-10T00:02:21Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon
- CPU: AMD Ryzen 9 5900X 12-Core Processor; RAM: 62.7 GiB
- Storage of the model files: ssd Samsung SSD 870 EVO 1TB (sata)
- OS: Debian GNU/Linux 13 (trixie) 7.2.12+deb13-amd64; drivers: not measured; runtime: not measured
- Strata commit: `7df6538d6c886093df39d74b36ed0a3396bf9467` - the checkout the runner found, which is a third-party
  RDNA3 fork of this project (see the operator section below). Engine 0.1.59; the runner labels this a "source
  build", but the engine binary is the official signed component, not a local build.
- Background work: not measured

## Model and configuration

- Model: swift-1.5-iq3_s; quantization IQ3_S; revision not measured; source revision not measured
- Files: `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `46cb3996cd1a17fe...`; `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `70ce6fb48a94fe34...`
- Track: **custom** (no notes given by the operator)
- Context 262144 tokens; server capacity 2 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 59 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 79.5 (69.9-84.3) n=3 | 0.96 (0.86-0.96) n=3 | 2.26 (1.73-2.33) n=3 | 4.6 (4.2-5.7) n=3 | 0/3 |
| prompt-4k | 1306.3 (1298.4-1310.7) n=3 | 81.4 (72.2-85.1) n=3 | 3.16 (3.15-3.17) n=3 | 6.08 (5.95-6.22) n=2 | 9.2 (9.2-10.2) n=3 | 2/3 |
| prompt-16k | 2942.1 (2936.2-2954.8) n=3 | 79.6 (78.7-82.6) n=3 | 5.60 (5.58-5.61) n=3 | 9.06 (7.86-9.90) n=3 | 12.0 (10.1-12.1) n=3 | 2/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 5 | 4 | 0 | 0 |
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
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).

## Provenance added by the operator (sanitized)

The runner could not read the hardware model over the API (`GPU: AMD Radeon` above) and the server was already
running, so it could not read the engine options. This section fills both in; paths are redacted.

- GPU: AMD Radeon RX 7900 XTX, 24 GiB (gfx1100), PCIe 4.0 x16; three active display outputs.
- Runtime: the engine is the portable component and loads its own HIP stack (`RUNPATH=$ORIGIN`):
  `libamdhip64.so.7`, `libhsa-runtime64.so.1`, `librocblas.so.5` and `libhipblaslt.so.1` come from the component
  directory, and `BUILD.json` states `portable: true` with `hipblaslt_version: 100200` (the engine's own line reads
  `hipBLASLt tuning enabled (26 rows, gfx1100, version 100200)`). A system ROCm 10.1.0 is installed on the box but
  the engine does not load it; it serves the source build and the tools (hipcc, rocminfo).
- Engine: `0.1.59-linux-x86_64-hip-rx7900` from the signed feed (release v0.2.13, upstream commit d2f0964).
- Control repository: `7df6538d` (a third-party RDNA3 fork of this project); the engine binary is the official component.
- The runner's protocol ID (`strata-qwen38-thinking-v1-draft.1`) and schema version (1) are the same as in
  v0.2.13's `docs/BENCHMARK_PROTOCOL.md`, so this report matches the current draft protocol. Only the control and
  serving layer come from the fork checkout; the engine and the timings it reports are the official component's.
- Launch arguments (sanitized): see `run-config-sanitized.json`.
- Tuning environment that matters for prompt speed: `STRATA_SELECT_WMMA=1` (the block scorer on the matrix cores),
  `STRATA_SCORES_THREAD=0`, `STRATA_PREFILL_AUTO_MAX=32768`, `STRATA_PREFILL_RING=192`, `STRATA_QSA_ROWS=1`.
- Vision: off (the `vision` section was removed from the run config); no image request was made.
- Note: the engine's prompt chunk is the host-aware plan's pick (32768 on this start, capped at 4096 for some prompts).
