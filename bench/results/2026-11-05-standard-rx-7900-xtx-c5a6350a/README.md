# Strata standard benchmark: AMD Radeon RX 7900 XTX

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261105T161234Z-c5a6350a`, started 2026-11-05T16:12:34Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon RX 7900 XTX (24.0 GiB dedicated)
- CPU: AMD Ryzen 9 5900X 12-Core Processor; RAM: 63.9 GiB
- Storage of the model files: ssd KINGSTON SKC3000D2048G (nvme)
- OS: Windows 10.0.19044; drivers: gpu-driver 32.0.32015.7029; runtime: not measured
- Strata commit: `c3b3f5035506f3e966f1b4b7b188aeaabae92f1d`; engine 0.1.57; source build
- Background work: llama-swap proxy running with no model loaded

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs; quantization IQ3_XXS; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `219ea929900dfa9e...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `draft_vocab.bin` `b1e1d3a7a9e4bf86...`; `expert-profile.bin` `8f59b4aa8873209d...`; `qwen3.8-flash-next-dflash2.q4_0.pack` `cda40d62d0eb8f37...`
- Track: **default** (Strata 0.2.11 (engine 0.1.57), Windows HIP RDNA3 release engine; setup.py defaults for qwen IQ3_XXS: context 131072, KV int8, KV streaming (--kv-resident 32768), MTP --spec 4, pictures off, speed projection off)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 2 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 83.7 (74.5-89.9) n=3 | 0.99 (0.85-1.03) n=3 | 2.56 (1.53-3.51) n=3 | 6.1 (5.3-6.5) n=3 | 1/3 |
| prompt-4k | 975.4 (970.7-981.2) n=3 | 71.6 (69.5-72.9) n=3 | 4.24 (4.21-4.26) n=3 | 10.40 (10.40-10.40) n=1 | 11.4 (11.2-11.6) n=3 | 3/3 |
| prompt-16k | 1159.1 (1154.2-1161.0) n=3 | 75.8 (73.5-84.3) n=3 | 14.18 (14.17-14.24) n=3 | 19.00 (17.22-19.10) n=3 | 20.9 (20.3-21.1) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 2 | 7 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 2 | 10 | 0 | 0 |

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
| 1 | 31.465974 | 2048 | 65.086178 | 0 |
| 2 | 31.028806 | 2048 | 66.003184 | 0 |
| 3 | 28.062494 | 1934 | 68.917609 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).

## Operator notes

- Correction for this run: `configuration.track_notes` and `run.command` in `results.json` say "MTP --spec 4". That is wrong. `config.json` and the first line of `engine.log` show `--draft dflash2 --spec 6 --draft-conf 0.35`, the same drafting as the later runs (drafting left at Auto, which resolved to DFlash2). `results.json` was not changed after the run. The runs `61de0c1e` and `7c5e57d7` were made afterwards with corrected track notes; their command and configuration are otherwise the same (`config.json` is byte-identical in all four runs).
- Hardware not in the generated report: model files are on a KINGSTON SKC3000D2048G (NVMe, 2 TB). RAM is 4 x 16 GiB at 2666 MT/s. GPU driver: AMD 32.0.32015.7029 (2026-10-18). HIP runtime: `amdhip64_7.dll` 10.0.3686.0, shipped in the engine folder; this is why `/software/runtime` is null. OS: Windows 10 IoT Enterprise LTSC 2021 (build 19044). Power limit: not measured.
- Install: the folder is a shallow git clone of tag v0.2.11 (commit `c3b3f5035506f3e966f1b4b7b188aeaabae92f1d`) set up by `setup.py`, which downloaded the ready-made Windows HIP RDNA3 engine 0.1.57. That is why `build_kind` reads `source` although the engine binary is the release package. The configuration is the output of `setup.py --family qwen --model IQ3_XXS` with defaults: context 131072, KV int8, `--kv-resident 32768`, pictures off, speed projection off. There are no environment overrides except setup's own hipBLASLt tuning table.
- Model: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF at revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, the revision `setup.py` pins and checks by size and SHA-256. Source and tokenizer revisions are not given.
- Background: the llama-swap proxy was idle with no model loaded. Four runs were made on four fresh engine starts with the same configuration; all four are kept. In order (run id, engine start time on the first line of `engine.log`, local clock): `20261105T161234Z-c5a6350a` 17:10:59; `20261105T161847Z-d22b4ea8` 17:17:24; `20261105T163250Z-61de0c1e` 17:31:29; `20261105T163849Z-7c5e57d7` 17:37:26.
- Context for strata issue 3431 (KV streaming reading 0 blocks): before these runs, 40 fresh engine starts of the same configuration, plus 1 earlier test start, each sent a short question and then a fresh long prompt of about 26K tokens (start 1 of the 40 and the test start used about 40K tokens). 0 of the 41 starts logged `KV streaming: ... of 0 block reads`, 41 of 41 answered correctly, and every start logged `36 slices pinned (28 GiB)`. These starts ran through `serve\server.py` with the configuration arguments as setup wrote them; the engine command line of every start shows `--draft dflash2 --spec 6 --draft-conf 0.35`, the same drafting as the benchmark runs.
