# Strata standard benchmark: Apple M2 Ultra

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261106T014011Z-1067121f`, started 2026-11-06T01:40:11Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: Apple M2 Ultra (128.0 GiB shared)
- CPU: Apple M2 Ultra; RAM: 128.0 GiB
- Storage of the model files: not measured
- OS: macOS 27.2; drivers: not measured; runtime: metal 27.2
- Strata commit: `3e0210663b07deafce51310ae622862c2fe05901`; engine 0.1.57-metal; release build
- Background work: Ollama 0.40.13: qwen3.8:27b-mlx loaded, idle; Home Assistant (Docker/OrbStack), idle

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `draft_vocab.bin` `b1e1d3a7a9e4bf86...`; `expert-profile.bin` `8f59b4aa8873209d...`; `qwen3.8-flash-next-dflash2.q4_0.pack` `cda40d62d0eb8f37...`
- Track: **custom** (Strata 0.2.11 release binary (Terminal install.sh), macOS 27.2, Mac Studio with Apple M2 Ultra (24-core CPU, 60-core GPU), 128 GB, internal SSD. Setup: strata launch --model-path (IQ3_S shards from ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, downloaded with hf) --context 32k --vision off --tune skip; DFlash2 drafter as set up by launch; profile created from the installed run config with port set to 8086. Background: Ollama 0.40.13 with qwen3.8:27b-mlx (27 GB) loaded but idle, Home Assistant in Docker, small house services.)
- Context 32768 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 4 min 7 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 69.6 (66.2-73.4) n=3 | 0.64 (0.59-0.66) n=3 | 2.29 (1.73-2.72) n=3 | 5.9 (5.7-6.5) n=3 | 0/3 |
| prompt-4k | 734.5 (682.3-736.0) n=3 | 57.7 (57.0-63.0) n=3 | 5.59 (5.58-6.02) n=3 | 11.04 (11.04-11.04) n=1 | 14.5 (13.7-15.0) n=3 | 3/3 |
| prompt-16k | 745.9 (744.3-746.0) n=3 | 59.2 (59.0-69.0) n=3 | 22.00 (21.99-22.04) n=3 | 29.53 (25.79-29.69) n=3 | 30.7 (29.4-30.7) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
