# Strata standard benchmark: NVIDIA GeForce RTX 3090

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261020T211303Z-ca1ea1e8`, started 2026-10-20T21:13:03Z. **Status: nonconforming.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

This run is not a complete standard result:

- the Strata commit is unknown (not a git checkout and no COMMIT file)

## Hardware and software

- GPU: NVIDIA GeForce RTX 3090 (24.0 GiB dedicated)
- CPU: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz; RAM: 123.4 GiB
- Storage of the model files: nvme Micron_7300_MTFDHBA960TDF (nvme)
- OS: Ubuntu 24.04.4 LTS 7.0.0-38-generic; drivers: gpu-driver 580.173.02; runtime: cuda 13.0
- Strata commit: `not measured`; engine 0.1.46; release build
- Background work: not measured

## Model and configuration

- Model: swift-1.5-q6_k; quantization Q6_K; revision not measured; source revision not measured
- Files: `Swift-1.5-Qwen3.8-Flash-Next-Q6_K-00001-of-00005.gguf` `da4cedfee7e5298c...`; `Swift-1.5-Qwen3.8-Flash-Next-Q6_K-00002-of-00005.gguf` `8aeb4c8e478809a9...`; `Swift-1.5-Qwen3.8-Flash-Next-Q6_K-00003-of-00005.gguf` `c93bf8b3d818fa22...`; `Swift-1.5-Qwen3.8-Flash-Next-Q6_K-00004-of-00005.gguf` `8dc4f1ed14d0187e...`; `Swift-1.5-Qwen3.8-Flash-Next-Q6_K-00005-of-00005.gguf` `44d0706c626f8bb0...`
- Track: **custom** (Strata 2.1 stack, engine binary from the 2.1 release (0.1.46 per release notes; the server self-reports its version string as 0.1.45, known labeling bug). Launch: ~/bin/strata-2-q6-bmo with strata-swift-q6-final7.json: resident-budget 100 GiB, expert-cache auto + expert profile, spec 4 + MTP, kv int8, conversation-cache 12288 MiB / 2 slots / min 4096 tokens, pipeline-windows 2, vram-reserve 896/1024, prefill auto:16384, spec-min-p 0.70, adapt every 1 / swaps 80 / decay 0.97, pcie-frac 0; env STRATA_MMQ_KQ, STRATA_PREFILL_HELP, STRATA_EXCHANGE_ROTATE, STRATA_PROMPT_ATTN_IMMA, STRATA_ATTN_DAILY, STRATA_QSA_SPLIT_ATTN all 1; pack packs/swift-q6; 2x RTX 3090 layer split)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 25 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s |
| --- | --- | --- | --- | --- | --- |
| short | 33.2 (25.8-41.2) n=3 | 78.2 (75.4-82.5) n=3 | 2.35 (1.90-3.04) n=3 | 3.51 (2.79-3.86) n=3 | 5.7 (5.2-6.5) n=3 |
| prompt-4k | 500.2 (476.9-500.4) n=3 | 81.5 (77.3-89.6) n=3 | 8.23 (8.23-8.64) n=3 | 11.02 (10.85-12.10) n=3 | 14.5 (13.1-15.2) n=3 |
| prompt-16k | 1316.0 (1163.0-1326.1) n=3 | 84.7 (83.1-93.0) n=3 | 12.52 (12.42-14.17) n=3 | 15.89 (15.06-17.53) n=3 | 18.6 (18.0-20.2) n=3 |

Cells are median (min-max) n=samples.

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
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 61 request(s) before this run.
