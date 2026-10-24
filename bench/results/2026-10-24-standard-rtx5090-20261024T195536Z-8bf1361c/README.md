# Strata standard benchmark: NVIDIA GeForce RTX 5090

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261024T195536Z-8bf1361c`, started 2026-10-24T19:55:36Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 5090 (31.8 GiB dedicated)
- CPU: AMD Ryzen 7 9800X3D 8-Core Processor; RAM: 95.3 GiB
- Storage of the model files: ssd SHPP41-2000GM (nvme)
- OS: Windows 10.0.22631; drivers: gpu-driver 616.56; runtime: cuda 13.4
- Strata commit: `dfa3f4025c9f1bc4af8cee201b20a2d223061813`; engine 0.1.48; release build
- Background work: Sunshine streaming

## Model and configuration

- Model: qwen3.8-flash-next-nvfp4-nvfp4-nvidia; quantization NVFP4; revision not measured; source revision not measured
- Files: `expert-profile.bin` `8f59b4aa8873209d...`; `ple-fp8.gguf` `37f89b8ce5a34259...`; `qwen-nvidia-nvfp4-dense.gguf` `0f795ca0cbb908e0...`; `token-embd-bf16.gguf` `386203af9070c7b4...`
- Track: **custom** (NVFP4 model pack; context 131072; KV int8; KV resident 32768; spec 4; resident budget 63 GiB; VRAM reserve 2048 MiB)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 1 min 48 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | not displayed (short-prompt overhead; see JSON) | 73.1 (55.1-83.3) n=3 | 0.77 (0.59-0.78) n=3 | 1.84 (1.33-2.12) n=3 | 5.9 (4.1-6.0) n=3 | 0/3 |
| prompt-4k | 2085.6 (634.2-2262.6) n=3 | 86.5 (72.1-92.3) n=3 | 2.00 (1.84-6.50) n=3 | 5.31 (3.52-12.44) n=3 | 7.8 (6.4-13.6) n=3 | 2/3 |
| prompt-16k | 3583.7 (3105.7-3914.2) n=3 | 83.5 (74.1-87.6) n=3 | 4.61 (4.24-5.31) n=3 | 10.67 (10.67-10.67) n=1 | 11.4 (10.1-11.5) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 4 | 5 | 0 | 0 |
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
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).

## Submission context and evidence review

This is pilot data for **`strata-qwen38-thinking-v1-draft.1`**, a draft protocol. It is not an approved benchmark, independently reproduced result, or official ranking. The runner's `complete` status means all required outcomes were recorded; it does not certify protocol activation or every model identity field.

The existing run started at **2026-10-24T19:55:36Z** (2026-10-25 03:55:36 Asia/Shanghai). Required stages took **108.422 seconds**, including warm-up and excluding model loading. Optional concurrency was not selected. The directory uses the original UTC date and the genuine `run.id`, `20261024T195536Z-8bf1361c`; no new run ID was generated. No benchmark was started or rerun to prepare this submission.

All 22 saved requests match the frozen fixture payloads. The saved result passes the installed protocol validator, including summary reconstruction. Raw usage and client timing fields agree with request records. Recomputing the sanity verdicts from the saved final answers and structured tool calls gives 12/12. These are offline evidence checks, not a new hardware run or an independent engine reproduction.

The quantization repository revision, source model revision, tokenizer revision and model-card revision are **not measured**. They remain null, with reasons in `results.json`. Build options are also unavailable. `model.custom_pack` records that no preparation was declared; the hashed `expert-profile.bin` entry does not establish how that profile was prepared. No revision or preparation history has been inferred. The runner cannot verify server repeat-stop and loop-recovery settings. These limits prevent claiming an activated standard or fully pinned comparison.

The original eight files were retained locally. In this submission copy, absolute model, profile and MTP directory prefixes in the engine log were replaced with placeholders. Filenames, engine options, timing lines and every request outcome were retained. The raw output file, configuration, template and original corpus provenance are byte-identical to the saved files. Artifact SHA-256 values in `results.json` refer to the final submitted bytes.

- [Raw requests, outputs, usage and client event offsets](outputs.jsonl)
- [Sanitized effective configuration](config.json)
- [Sanitized engine log](engine.log)
- [Effective chat template](provenance/chat_template.jinja)
- [Frozen corpus manifest](provenance/corpus-manifest.sha256)
- [Corpus build provenance](provenance/corpus-build.json)
- [Submission checks, redaction counts and original file hashes](provenance/submission-review.json)
