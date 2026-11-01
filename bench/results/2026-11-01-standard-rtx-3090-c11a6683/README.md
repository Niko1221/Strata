# Strata standard benchmark: NVIDIA GeForce RTX 3090

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261101T082826Z-c11a6683`, started 2026-11-01T08:28:26Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3090 (24.0 GiB dedicated)
- CPU: AMD Ryzen 7 7800X3D 8-Core Processor; RAM: 62.3 GiB
- Storage of the model files: not measured
- OS: CachyOS 7.3.1-1-cachyos; drivers: gpu-driver 615.78.08; runtime: cuda 13.4
- Strata commit: `562a0c2ed7169ff6373f893b162106d55a828297`; engine 0.1.50; release build
- Background work: KDE Plasma (kwin/plasmashell) held about 1.1 GiB VRAM the whole run; agent harness stopped with the runtime; RSS sampler running; no game, browser or other GPU work; zram swap of 62.3 GiB present, nothing paged

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision de4b8e4d43b917e7706784d8bb445c9af86a3540
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (custom saved profile launch-iq3_s-128k tuned round t1: CachyOS host, 131072 ctx, --kv k8v4 --spec-min-p 0.70 --pcie-frac 0.17, --expert-cache auto, --prefill auto, --spec 4, --mtp rt pack, --vram-reserve-mib 700, --batch 3, server --parallel 3; release binaries (terminal 0.1.49.2 build 0.2.4.2, control 0.1.51, engine 0.1.50 cuda13, vision engine 0.1.48); Tier 1: kv int8->k8v4, spec-min-p 0.5->0.70, pcie-frac default 0.55->0.17)
- Context 131072 tokens; server capacity 3 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 4 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 76.4 (74.9-79.3) n=3 | 0.63 (0.59-0.66) n=3 | 1.61 (1.48-2.17) n=3 | 4.8 (4.4-5.5) n=3 | 0/3 |
| prompt-4k | 2065.3 (2052.7-2100.7) n=3 | 68.2 (67.6-70.5) n=3 | 2.01 (1.97-2.02) n=3 | 8.84 (8.41-9.27) n=2 | 9.5 (9.3-9.6) n=3 | 3/3 |
| prompt-16k | 2590.1 (2584.0-2594.9) n=3 | 70.8 (70.7-73.6) n=3 | 6.36 (6.34-6.37) n=3 | 12.44 (11.54-13.35) n=2 | 13.6 (13.3-13.6) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 4 | 8 | 0 | 0 |

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

Concurrency rounds (four requests released together):

| Round | Wall s | Output tokens | Aggregate tok/s | Failed |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 23.907845 | 1952 | 81.64684 | 0 |
| 2 | 25.420201 | 1928 | 75.845191 | 0 |
| 3 | 24.92548 | 1947 | 78.112839 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 1 request(s) before this run.
