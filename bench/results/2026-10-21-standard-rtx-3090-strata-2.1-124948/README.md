# Strata standard benchmark: NVIDIA GeForce RTX 3090, NVIDIA GeForce RTX 3090

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261021T120047Z-66e200f7`, started 2026-10-21T12:00:47Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3090 (24.0 GiB dedicated); NVIDIA GeForce RTX 3090 (24.0 GiB dedicated)
- CPU: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz; RAM: 123.4 GiB
- Storage of the model files: nvme Micron_7300_MTFDHBA960TDF (nvme)
- OS: Ubuntu 24.04.4 LTS 7.0.0-38-generic; drivers: gpu-driver 580.173.02; runtime: cuda 13.0
- Strata commit: `90767252f420e25223c739d604145ad15911d48b`; engine 0.1.46; release build
- Background work: strata agent sessions sharing this server during the run window

## Model and configuration

- Model: swift-1.5-q6_k; quantization Q6_K; revision 0142554c963a8653912c0c8283f05f13f63653f3; source revision not measured
- Files: `Swift-1.5-Qwen3.8-Flash-Next-Q6_K-00001-of-00005.gguf` `da4cedfee7e5298c...`; `Swift-1.5-Qwen3.8-Flash-Next-Q6_K-00002-of-00005.gguf` `8aeb4c8e478809a9...`; `Swift-1.5-Qwen3.8-Flash-Next-Q6_K-00003-of-00005.gguf` `c93bf8b3d818fa22...`; `Swift-1.5-Qwen3.8-Flash-Next-Q6_K-00004-of-00005.gguf` `8dc4f1ed14d0187e...`; `Swift-1.5-Qwen3.8-Flash-Next-Q6_K-00005-of-00005.gguf` `44d0706c626f8bb0...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (Strata 2.1 stack (release commit 90767252f420e25223c739d604145ad15911d48b, tag v0.2.1; engine binary 0.1.46 signed prebuilt cuda13 package), pack packs/swift-q6, config strata-swift-q6-final7.json, 2x RTX 3090 layer split; follow-up of run 20261020T211303Z-ca1ea1e8; engine instance restarted 2026-10-21T11:34:32Z so no standard-v1 payload reuse from the earlier run)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 35 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s |
| --- | --- | --- | --- | --- | --- |
| short | 22.8 (18.2-38.0) n=3 | 80.9 (79.9-84.9) n=3 | 3.40 (2.05-4.25) n=3 | 4.39 (2.96-5.56) n=3 | 8.4 (6.7-9.5) n=3 |
| prompt-4k | 517.1 (500.9-518.1) n=3 | 79.2 (73.9-81.1) n=3 | 7.95 (7.93-8.22) n=3 | 11.57 (11.31-11.65) n=3 | 14.4 (14.2-15.1) n=3 |
| prompt-16k | 1352.7 (1117.8-1381.9) n=3 | 85.9 (81.7-94.8) n=3 | 12.17 (11.91-14.70) n=3 | 17.91 (14.01-17.99) n=3 | 18.2 (16.9-20.6) n=3 |

Cells are median (min-max) n=samples.

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
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 24 request(s) before this run.
