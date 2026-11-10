# Strata standard benchmark: NVIDIA GeForce RTX 5080

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261110T121229Z-573d548c`, started 2026-11-10T12:12:29Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 5080 (15.9 GiB dedicated)
- CPU: AMD Ryzen 7 7800X3D 8-Core Processor; RAM: 127.4 GiB
- Storage of the model files: ssd Samsung SSD 980 PRO 2TB (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 621.01; runtime: cuda 13.5
- Strata commit: `628f74955a56fcba092495b800dd02af75373a0f`; engine 0.1.59; source build
- Background work: Windows 11 desktop: explorer, StartMenu, SearchHost, Edge WebView2, Firefox, Signal, Cryptomator, ArgusMonitor, ScanSnap; about 2.1 GiB VRAM in use before the run (nvidia-smi)

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `draft_vocab.bin` `b1e1d3a7a9e4bf86...`; `expert-profile.bin` `8f59b4aa8873209d...`; `qwen3.8-flash-next-dflash2.q4_0.pack` `cda40d62d0eb8f37...`
- Track: **custom** (Strata 0.2.13.1 (tag v0.2.13.1, commit 628f7495), engine 0.1.59 built from source (cmake+ninja, MSVC 19.44, CUDA 13.4, sm_120, BUILD.json source=local). Run config = setup defaults plus the calibration written by setup on 2026-10-08: --pool-workers 4 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97, --expert-profile data/expert-profile.bin, --expert-cache auto --prefill auto --kv int8 --max-context 262144. Drafter: DFlash2 from the run config default drafting=auto (--draft dflash2 --spec 6 --draft-conf 0.7, no MTP layer).)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 52 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 78.7 (73.6-80.6) n=3 | 0.73 (0.60-0.73) n=3 | 1.75 (1.55-1.91) n=3 | 5.5 (4.1-5.6) n=3 | 0/3 |
| prompt-4k | 2679.4 (2593.5-2706.3) n=3 | 72.6 (66.5-76.2) n=3 | 1.58 (1.56-1.64) n=3 | 6.83 (5.94-7.73) n=2 | 8.7 (8.3-9.3) n=3 | 3/3 |
| prompt-16k | 3553.9 (3547.4-3577.7) n=3 | 70.8 (67.5-72.4) n=3 | 4.66 (4.66-4.68) n=3 | 10.25 (10.25-10.25) n=1 | 11.9 (11.7-12.2) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
- `model_card_revision` is null: the draft protocol does not pin a model-card revision yet.
- `source_revision` is null: the runner was not given `--source-revision`. The quantization repository revision above is the one the files were downloaded at.
- The optional concurrency stage was not run (`optional_duration_s` is null): the server was started with a capacity of one slot, so a concurrent stage would measure queueing rather than throughput.
- The engine logged `130 MiB of VRAM free with everything loaded - LOW` and suggested `--vram-reserve-mib 1082`. The engine options were left unchanged so that this run stays comparable with the same configuration measured on 0.2.13. No request stalled and no request failed, but a machine with less free VRAM may need that reserve.
- Machine paths in `engine.log` and in `run.command` are replaced by `<strata-install>`, `<data-dir>` and `<checkout>`; the artifact hashes are computed over the redacted bytes.
