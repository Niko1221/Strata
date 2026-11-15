# Strata standard benchmark: 2x NVIDIA GeForce RTX 3090 (layer split)

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261115T051855Z-b33d7937`, started 2026-11-15T05:18:55Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3090 + NVIDIA GeForce RTX 3090, 48 GiB VRAM total; layer split across both cards (the runner reads the first card only, so `results.json` records one)
- CPU: AMD Ryzen 9 5900X 12-Core Processor; RAM: 31.9 GiB
- Storage of the model files: ssd Samsung SSD 980 PRO 1TB (nvme)
- OS: Windows 10.0.22631; drivers: gpu-driver 621.24; runtime: cuda 13.5
- Strata commit: `146df3c468739147f520ac3816de806f71232571`; engine 0.1.57; source build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-gyro-gyro-s; quantization not measured; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf` `c6ea1b08ca312f99...`
- Track: **custom** (no notes given by the operator)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json` (the install's own configuration, unchanged); chat template: `provenance/chat_template.jinja`
  Launched with: `--pack gyro-gyro-s --native Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf --ple-gguf Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --resident-experts --kv-resident 32768 --vision --vram-reserve-mib 700 --vram-reserve-later-mib 1194 --conversation-cache-mib 3072 --conversation-cache-slots 8 --layer-split auto` (cards: helper CUDA0, main CUDA1)

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 34 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 106.1 (106.1-114.5) n=3 | 0.43 (0.30-0.50) n=3 | 1.24 (0.99-1.31) n=3 | 3.8 (2.9-4.3) n=3 | 0/3 |
| prompt-4k | 1787.2 (1776.0-1907.4) n=3 | 110.4 (109.0-123.1) n=3 | 2.31 (2.16-2.32) n=3 | 3.88 (3.79-4.14) n=3 | 6.1 (5.0-6.3) n=3 | 1/3 |
| prompt-16k | 2577.1 (2553.4-2673.5) n=3 | 92.2 (85.8-100.3) n=3 | 6.40 (6.17-6.48) n=3 | 11.13 (7.94-11.94) n=3 | 12.0 (10.5-12.3) n=3 | 2/3 |

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
- The server had answered 32 request(s) before this run.

## Operator notes

- Two RTX 3090 on one Ryzen 9 5900X board: the helper card sits on a chipset slot (PCIe gen3 x4, ~3.4 GB/s), the main card CPU-direct (gen4 x16); the engine splits layers across them (`--layer-split auto`, main = CUDA1).
- The configuration is this install's own, tuned for the two-card split: per-card VRAM reserves (`700`/`1194` MiB) that boot without a `LOW` line on the main card, and a pinned conversation cache (`3072` MiB / `8` slots), which is why the track is custom.
- No other program used the GPU during the run; the measuring client was the only load, and requests ran one at a time. The 32 requests served before the run were the operator's agent session on the same server (a restarted engine; the shared template prefix was the only reused prompt text).
- A quick-suite report of the same model and machine before this tuning is at [2026-11-08-community-2x-rtx-3090-qwen3-8-flash-next-gyro-gyro-s](../2026-11-08-community-2x-rtx-3090-qwen3-8-flash-next-gyro-gyro-s/README.md): prompt 1737.7/2385.3 and decode 97.1 tok/s at fresh 4K/16K against this run's 1787.2/2577.1 and 106.1 — about +3% at 4K, +8% at 16K, +9% decode after the cache pin and reserves above.
