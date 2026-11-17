# Strata standard benchmark: NVIDIA GeForce RTX 3070

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261117T115800Z-7c6e2e69`, started 2026-11-17T11:58:00Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3070 (8.0 GiB dedicated)
- CPU: Intel(R) Core(TM) i9-10850K CPU @ 3.60GHz; RAM: 63.8 GiB
- Storage of the model files: ssd WDS100T3X0C-00SJG0 (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 610.60; runtime: cuda 13.3
- Strata commit: `c77572806996349b49995e4f98d41b5294d36a21`; engine 0.1.60; release build
- Background work: Codex desktop and benchmark orchestration; other Windows desktop processes were not closed or continuously monitored; no other inference requests intentionally submitted.

## Model and configuration

- Model: qwen3.8-flash-next-q2_0; quantization Q2_0; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` `69820c02ec7d0b45...`; `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `draft_vocab.bin` `b1e1d3a7a9e4bf86...`; `expert-profile.bin` `8f59b4aa8873209d...`; `qwen3.8-flash-next-dflash2.q4_0.pack` `cda40d62d0eb8f37...`
- Track: **custom** (Released Windows CUDA engine 0.1.60 and Control 0.1.61; Q2_0, 32K context, one slot, vision off, prefill 512, vram-reserve-mib 1600; launcher-selected DFlash2; no calibration override retained.)
- Context 32768 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 9 min 34 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 41.6 (39.4-42.1) n=3 | 1.08 (0.88-1.08) n=3 | 7.84 (2.70-9.66) n=3 | 13.2 (9.5-13.8) n=3 | 2/3 |
| prompt-4k | 251.0 (247.2-251.2) n=3 | 37.5 (36.9-38.7) n=3 | 16.36 (16.33-16.62) n=3 | 26.94 (24.72-29.16) n=2 | 30.2 (29.6-30.2) n=3 | 3/3 |
| prompt-16k | 251.3 (250.3-253.2) n=3 | 40.7 (38.8-40.9) n=3 | 65.27 (64.78-65.53) n=3 | 72.78 (71.94-73.48) n=3 | 77.7 (77.3-78.7) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 1 | 8 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 3 | 9 | 0 | 0 |

Sanity checks: **11 of 12**.

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
| T3 | fail | expected one structured tool call, got 0 |
| T4 | pass | the expected tool call |

Concurrency rounds (four requests released together):

| Round | Wall s | Output tokens | Aggregate tok/s | Failed |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 55.063 | 2048 | 37.19376 | 0 |
| 2 | 55.484 | 2048 | 36.911542 | 0 |
| 3 | 46.719 | 1699 | 36.366361 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r2-c1: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).


## Operator evidence review

T3 failed the functional check: its final answer was `3 km = 3000 m`, with no
structured `convert_units` call. The text conversion is correct, but the protocol
requires a tool call, so this remains a failure and was not retried. The runner's
"No request failed" refers to HTTP/stream execution, not the sanity verdicts.

This run supplements the [16 November custom calibration report](../2026-11-16-community-rtx-3070-8gb-qwen/README.md).
It uses the draft protocol, with thinking enabled and fixed sampling; the earlier
thinking-off requests remain a separate historical series and are not pooled here.
The same installed release is measured, with the launcher's effective drafting
arguments recorded in this run's `config.json` and `engine.log`. This is not a
paired speedup claim against the earlier measurements.

The optional stage released four requests together against the unchanged
**1-slot server**. It measures queueing under that capacity;
it does not establish four-way parallel decoding. No request was intentionally
sent by another inference client during this run. Codex and ordinary Windows
desktop background processes were not closed or continuously monitored.

The runner's result status is **complete**. All outcomes, output-cap
truncations, failures, missing metadata and exclusions remain in the evidence.
Protocol draft status remains visible; this is not an officially approved result.
Missing immutable model/source/tokenizer revisions are not filled by guessing.

Validation: `strata bench report` validates the result format. The accompanying
`provenance/validate_evidence.py` independently checks request counts, sampling,
timing-derived rates, summary medians/ranges, artifact hashes and all twelve sanity
verdicts directly from the saved final answers/tool calls. Independent score:
**11/12**, matching the runner. See `provenance/independent-validation.json`.
Task-specific data/workspace roots were additionally replaced with placeholders;
artifact hashes were refreshed after redaction, without changing measured numbers.
`SHA256SUMS.txt` covers the final submitted files.
