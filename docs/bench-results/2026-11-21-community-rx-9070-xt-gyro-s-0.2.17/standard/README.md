# Strata standard benchmark: AMD Radeon RX 9070 XT

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261121T044757Z-701bcc16`, started 2026-11-21T04:47:57Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon RX 9070 XT (15.9 GiB dedicated)
- CPU: AMD Ryzen 5 7600X3D 6-Core Processor; RAM: 31.4 GiB
- Storage of the model files: ssd SKHynix_HFS512GDE9X084N (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 32.0.32029.2010; runtime: not measured
- Strata commit: `12d491da4a6904eb49aaa6f2f63ec4b1a17582e3`; engine 0.1.63; release build
- Background work: Windows desktop only; GPU at stock clocks (no Adrenalin tuning); Strata desktop app and Chrome closed, Claude desktop app open; no other GPU work

## Model and configuration

- Model: qwen3.8-flash-next-gyro-gyro-s; quantization Q4_0; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf` `c6ea1b08ca312f99...`; `draft_vocab_latin.bin` `0b4d93fe2ef9b519...`; `perfil-expertos-bench.bin` `9d69dc44abbdabf2...`; `qwen3.8-flash-next-dflash2.q4_0.pack` `cda40d62d0eb8f37...`
- Track: **custom** (release 0.2.17, hand-tuned config (DFlash2, spec 6, pcie-frac 1.0, 8 GiB pinned, prompt-cache-spread))
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 56 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 79.8 (77.6-95.7) n=3 | 1.05 (0.89-1.11) n=3 | 2.28 (1.95-3.28) n=3 | 5.0 (4.6-6.2) n=3 | 0/3 |
| prompt-4k | 1830.2 (1781.3-1831.8) n=3 | 89.2 (71.5-93.0) n=3 | 2.28 (2.27-2.31) n=3 | 4.61 (4.28-9.41) n=3 | 7.2 (6.6-9.4) n=3 | 1/3 |
| prompt-16k | 2446.1 (2426.7-2447.9) n=3 | 91.5 (88.9-94.4) n=3 | 6.74 (6.72-6.78) n=3 | 9.58 (9.02-9.69) n=3 | 12.3 (11.9-12.5) n=3 | 2/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 6 | 3 | 0 | 0 |
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
