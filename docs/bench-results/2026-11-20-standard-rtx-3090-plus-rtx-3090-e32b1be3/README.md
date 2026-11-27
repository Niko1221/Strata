# Strata standard benchmark: NVIDIA GeForce RTX 3090, NVIDIA GeForce RTX 3090

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261120T073741Z-e32b1be3`, started 2026-11-20T07:37:41Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl). The engine's startup log is included as [engine-log.txt](engine-log.txt) - the runner wrote it as `engine.log`, renamed here only to pass the repository's `*.log` ignore rule (the `artifacts` entry in results.json still lists it as `engine.log` with its sha256; byte-identical).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3090 (24.0 GiB dedicated); NVIDIA GeForce RTX 3090 (24.0 GiB dedicated)
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor; RAM: 127.6 GiB
- Storage of the model files: ssd Samsung SSD 990 EVO Plus 4TB (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 616.92; runtime: cuda 13.4
- Strata commit: `64524fb9fc82222ddce554c0643b64f056ec8366`; engine 0.1.62; release build
- Background work: Windows 11 Hyper-V host; Ubuntu Server + Talos VMs running at near-idle load (not stopped for this run); ~95 GB RAM free; otherwise idle desktop

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `draft_vocab.bin` `b1e1d3a7a9e4bf86...`; `expert-profile.bin` `8f59b4aa8873209d...`; `qwen3.8-flash-next-dflash2.q4_0.pack` `cda40d62d0eb8f37...`
- Track: **custom** (custom track: machine-calibrated setup (expert-profile.bin + --prefill-pipe 1536); DFlash2 drafter (--draft dflash2, q4_0-selq4 pack + selector, --spec 6 --draft-conf 0.35, --draft-fallback none), serial drafter, server capacity 1 slot; experimental speed projection ON (refusal-direction control vector: --control-vector-scaled ...speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer); vision ON (mmproj BF16); 2x RTX 3090 layer_split auto; --kv int8 --kv-resident 32768; 200K context; engine 0.1.62-windows-x86_64-cuda-cuda13 from the component store, Control/CLI 0.2.16)
- Context 204800 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 39 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 118.5 (115.1-128.7) n=3 | 0.55 (0.48-0.55) n=3 | 1.36 (1.33-3.75) n=3 | 3.5 (2.8-5.0) n=3 | 1/3 |
| prompt-4k | 1371.0 (1365.3-1377.3) n=3 | 119.6 (108.2-146.9) n=3 | 3.02 (3.00-3.03) n=3 | 4.14 (4.14-4.14) n=1 | 7.3 (6.5-7.7) n=3 | 3/3 |
| prompt-16k | 2547.1 (2516.7-2561.6) n=3 | 128.9 (109.5-135.4) n=3 | 6.45 (6.44-6.55) n=3 | 9.08 (8.56-9.59) n=2 | 10.4 (10.2-11.2) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 2 | 7 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 1 | 11 | 0 | 0 |

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
| 1 | 16.329 | 1893 | 115.928716 | 0 |
| 2 | 20.11 | 2048 | 101.839881 | 0 |
| 3 | 19.375 | 2048 | 105.703226 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c3: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r2-c1: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r2-c3: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c1: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).

## Author's note: same-day comparison with the MTP run (2026-11-20-standard-rtx-3090-plus-rtx-3090-e2f62039)

This second standard run measures the machine's **current daily production config**: the 0.2.16 CLI switched drafting from MTP to the downloaded DFlash2 drafter (serial, server capacity 1 slot instead of 3). The experimental speed projection (refusal-direction control vector) was **ON in both runs** — this run's engine log shows `control vector mode = project, dir = per-layer, layers 4..44 (41 steered)`, same as the other run. The only uncontrolled difference besides drafting: the two Hyper-V VMs stayed running at near-idle load here (stopped for the MTP run).

Medians, this run (DFlash2) vs the MTP run:

| Case | DFlash2 (this run) | MTP (e2f62039) |
| --- | --- | --- |
| short decode tok/s | **118.5** | 81.8 |
| prompt-4k tok/s / TTFT | **1,371** / 3.02 s | 983 / 4.20 s |
| prompt-16k tok/s / TTFT | **2,547** / 6.45 s | 1,960 / 8.41 s |
| concurrency rounds (aggregate tok/s) | 115.9 / 101.8 / 105.7 (capacity 1) | 80.4 / 81.7 / 85.0 (capacity 3) |
| sanity | 12/12 | 12/12 |

The DFlash2 drafter is the clear winner for single-stream decode and prefill; the concurrency stage here shows a 1-slot server still aggregating more than a 3-slot MTP server on four released requests, which is what "serial drafter" means in practice. Both runs are complete with 0 failed requests; each report documents exactly what its engine loaded.
