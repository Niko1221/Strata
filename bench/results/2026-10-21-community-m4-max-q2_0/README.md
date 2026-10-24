# Strata standard benchmark: Apple M4 Max

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261021T082110Z-fd14aff4`, started 2026-10-21T08:21:10Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: Apple M4 Max (64.0 GiB shared)
- CPU: Apple M4 Max; RAM: 64.0 GiB
- Storage of the model files: not measured
- OS: macOS 27.0.1; drivers: not measured; runtime: metal 27.0.1
- Strata commit: `90767252f420e25223c739d604145ad15911d48b`; engine 0.1.46-metal; release build
- Background work: Mac Studio M4 Max, 64 GB unified memory, macOS 27.0.1 (26A434), AC power, no power limit set; no other GPU/CPU-heavy workloads running

## Model and configuration

- Model: qwen3.8-flash-next-q2_0; quantization Q2_0; revision ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF@ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` `69820c02ec7d0b45...`; `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **default** (Strata 0.2.1 release, engine 0.1.46 macos-arm64-metal. Setup defaults: model q2_0, context 32k, vision gpu, tune skip. Model files on an NVMe SSD; no other GPU or CPU-heavy workloads.)
- Context 32768 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 4 min 8 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s |
| --- | --- | --- | --- | --- | --- |
| short | 183.5 (134.9-203.6) n=3 | 72.4 (72.1-72.6) n=3 | 0.44 (0.40-0.59) n=3 | 1.19 (1.14-1.34) n=3 | 7.5 (7.3-7.6) n=3 |
| prompt-4k | 692.4 (647.1-693.3) n=3 | 69.2 (68.5-69.8) n=3 | 5.93 (5.92-6.35) n=3 | 8.21 (8.09-9.03) n=3 | 12.0 (11.7-13.3) n=3 |
| prompt-16k | 683.8 (682.8-684.5) n=3 | 66.0 (65.7-66.6) n=3 | 23.99 (23.96-24.02) n=3 | 28.65 (28.27-30.65) n=3 | 31.7 (31.7-31.7) n=3 |

Cells are median (min-max) n=samples.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 5 | 4 | 0 | 0 |
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
- The runner could not tell whether the server was freshly loaded.

## Provenance notes

Four things a reviewer should know that the generated report cannot express on its own.

**The Strata commit above was placed by hand.** The v0.2.1 release ships no `COMMIT` file, so the runner could not read the commit and marked the first attempt at this run `nonconforming` ("the Strata commit is unknown"). `90767252f420e25223c739d604145ad15911d48b` was written into the installed terminal package by the submitter; it is the commit that the annotated tag `v0.2.1` dereferences to, and it is not the tag object's own SHA. The engine binary is the unmodified 0.1.46 `macos-arm64-metal` release package, so `build_kind: release` is accurate. Reported upstream as [#3561](https://github.com/Niko1221/Strata/issues/3561).

**The tokenizer has no upstream revision.** `vocab.json`, `merges.txt` and `token_type.json` in the pack are generated from the Q2_0 GGUF by `tools/strata_tokenizer.py` (`setup.py` step 6), not copied from a Hugging Face repository. We checked this rather than assuming it: all of them differ byte for byte from `Qwen/Qwen3.8-Flash-Next` at `de4b8e4d`, and `token_type.json` does not exist upstream at all. `--tokenizer-revision` therefore records the derivation instead of naming a revision that does not exist.

**The three macOS nulls are platform gaps.** `hardware/storage`, `software/drivers` and `gpus[0].power_limit_w` are null because the runner identifies the drive and the GPU driver through Linux and Windows tooling that does not exist on Apple Silicon. The model files did sit on an internal NVMe SSD and no power limit was set. These are not measurements that were skipped; the tools to take them are not there. `protocol.model_card_revision` is null for every submitter while the protocol is a draft.

**Reproducibility.** Three complete runs on this machine, each from a freshly loaded engine, same release and same configuration:

| Run | short decode | 4K prompt | 4K decode | 16K prompt | 16K decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-10-20 | 72.9 | 694.2 | 69.6 | 693.2 | 66.9 |
| 2026-10-21 | 72.9 | 696.3 | 69.9 | 686.9 | 65.9 |
| this run | 72.4 | 692.4 | 69.2 | 683.8 | 66.0 |

Medians in tokens/s, n=3 each. Decode varies by under 1%; prompt throughput by under 2%.

## Why this machine is worth a row

`docs/MAC.md` lists measurements for an M5 Max at 128 GB, an M4 mini at 32 GB and an M1 Max at 34 GB, and says other chips are untested. An M4 Max at 64 GB is not among them, and it is the size at which that same table recommends Q2_0 as the largest pack that stays resident (it puts IQ3_XXS at 74 GB). `bench/results/` contains no Apple Silicon row at all yet, so this fills the gap between the M4 mini (32 GB) and the M5 Max (128 GB) figures in that table.
