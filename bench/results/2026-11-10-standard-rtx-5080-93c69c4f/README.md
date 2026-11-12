# Strata standard benchmark: NVIDIA GeForce RTX 5080

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261110T175450Z-93c69c4f`, started 2026-11-10T17:54:50Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 5080 (15.9 GiB dedicated)
- CPU: AMD Ryzen 7 7800X3D 8-Core Processor; RAM: 127.4 GiB
- Storage of the model files: ssd Samsung SSD 980 PRO 2TB (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 621.01; runtime: cuda 13.5
- Strata commit: `628f74955a56fcba092495b800dd02af75373a0f`; engine 0.1.59; source build
- Background work: Windows 11 desktop: explorer, StartMenuExperienceHost, SearchHost, Edge WebView2, Firefox, Signal, Cryptomator, ArgusMonitor, ScanSnap, UniGetUI, Acronis TrueImage; about 1.4 GiB VRAM in use before the run (nvidia-smi)

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs; quantization IQ3_XXS; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `219ea929900dfa9e...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `draft_vocab.bin` `b1e1d3a7a9e4bf86...`; `expert-profile.bin` `8f59b4aa8873209d...`; `qwen3.8-flash-next-dflash2.q4_0.pack` `cda40d62d0eb8f37...`
- Track: **custom** (Strata 0.2.13.1 (tag v0.2.13.1, commit 628f7495), engine 0.1.59 built from source (cmake+ninja, MSVC 19.44, CUDA 13.4, sm_120, BUILD.json source=local). Run config = setup defaults plus the calibration written on 2026-10-31: --pool-workers 5 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97 --vram-reserve-mib 700, --expert-profile data/expert-profile.bin, --expert-cache auto --prefill auto --kv int8 --max-context 262144. No --pcie-frac: the engine refines the PCIe share itself on a single CUDA card (PcieAuto); an explicit --pcie-frac 1.00 measured 80.5 tok/s chat decode against 92.0 without it on this machine, so the flag was removed before this run. Drafter: DFlash2 from the run config default drafting=auto (--draft dflash2 --spec 6 --draft-conf 0.7, no MTP layer); the --spec 4, --spec-min-p 0.70 and --mtp entries in the config are replaced by that decision. --vision is on; none of the measured requests used an image.)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 49 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 89.6 (81.1-101.2) n=3 | 0.67 (0.48-0.71) n=3 | 2.79 (2.03-4.36) n=3 | 6.0 (4.7-7.0) n=3 | 1/3 |
| prompt-4k | 2522.7 (2241.3-2723.0) n=3 | 79.9 (78.3-97.6) n=3 | 1.67 (1.55-1.87) n=3 | 5.43 (4.10-7.22) n=3 | 8.1 (6.8-8.4) n=3 | 3/3 |
| prompt-16k | 2727.7 (2545.0-3701.9) n=3 | 79.1 (72.4-85.3) n=3 | 6.05 (4.46-6.49) n=3 | 11.87 (11.17-12.56) n=2 | 12.5 (10.4-13.5) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 2 | 7 | 0 | 0 |
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
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
- `model_card_revision` and `source_revision` are null: the draft protocol does not pin a model card yet, and `--source-revision` was not passed. The quantization repository revision named above is the one the files were downloaded at.
- The optional concurrency stage was not run. The server was started with a capacity of one slot, so the stage would measure queueing rather than throughput; `optional_duration_s` is null with that reason.
- The engine logged `166 MiB of VRAM free with everything loaded - LOW` and suggested `--vram-reserve-mib 1046`; for one request it suggested 1664 after seeing 102 MiB free. The options were left as the configuration has them (`--vram-reserve-mib 700`). All 21 required requests finished and nothing stalled, but on a card with less free VRAM the reserve may be needed. The expert cache used the rest of the card: 2491 slots, 85.8 % to 91.6 % decode hits per request.
- Absolute machine paths in `engine.log` and in `run.command` are replaced by `<checkout>`, `<data-dir>` and `<strata-install>`, and the artifact hashes in `results.json` are recomputed over the redacted bytes.
- Background load: a normal Windows desktop (about 1.4 GiB VRAM in use before the run). The runner cannot see other programs, so this is reported, not controlled.
