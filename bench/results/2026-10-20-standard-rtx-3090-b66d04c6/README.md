# Strata standard benchmark: NVIDIA GeForce RTX 3090

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261020T135103Z-b66d04c6`, started 2026-10-20T13:51:03Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3090 (24.0 GiB dedicated)
- CPU: Intel(R) Core(TM) i9-9960X CPU @ 3.10GHz; RAM: 123.2 GiB
- Storage of the model files: nvme SAMSUNG MZVL2512HCJQ-00BL2 (nvme)
- OS: Ubuntu 26.04.1 LTS 7.0.0-38-generic; drivers: gpu-driver 595.99.02; runtime: cuda 13.2
- Strata commit: `9ae93f7b1fe989d04a8e404d32cddaeeff10505f`; engine 0.1.46.1; source build
- Background work: GPU power capped at 200 W for the run; host: i9-9960X, 123 GiB RAM, Ubuntu 26.04, driver 595.99.02

## Model and configuration

- Model: qwen3.8-flash-next-ud-q4-kxl-21; quantization UD-Q4_K_XL (Unsloth dynamic, GGUF in place); revision not measured; source revision 9ae93f7b
- Files: `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` `4448186216b3af4c...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf` `3f342f1c1580473f...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf` `56758f40269cad5c...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf` `753bda48b98ba4f1...`
- Track: **custom** (Strata 0.2.1 source install, engine 0.1.46.1. Hand-written run config: GGUF in place (--native, no experts.bin), --resident-budget-gib 80 so all experts stay in RAM, --kv int8 --kv-resident 32768, MTP draft layer, control vector (experimental speed projection) on. GPU power limit 200 W (nvidia-power-limit.service). Server bound to 127.0.0.1 for the run; no other clients.)
- Context 1048576 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 4 min 26 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s |
| --- | --- | --- | --- | --- | --- |
| short | 85.1 (79.0-91.2) n=3 | 58.2 (51.2-60.8) n=3 | 0.93 (0.87-1.00) n=3 | 2.63 (2.45-6.78) n=3 | 9.1 (8.0-11.0) n=3 |
| prompt-4k | 448.8 (448.1-448.9) n=3 | 56.7 (52.8-58.3) n=3 | 9.16 (9.15-9.17) n=3 | 14.36 (13.94-16.83) n=3 | 18.2 (17.9-18.8) n=3 |
| prompt-16k | 782.9 (773.5-786.7) n=3 | 54.8 (52.8-57.2) n=3 | 20.97 (20.87-21.23) n=3 | 26.22 (25.68-27.66) n=3 | 30.3 (29.8-30.9) n=3 |

Cells are median (min-max) n=samples.

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
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The runner could not tell whether the server was freshly loaded.
