# Strata standard benchmark: NVIDIA GeForce RTX 3090

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261020T142528Z-ce9e57fe`, started 2026-10-20T14:25:28Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3090 (24.0 GiB dedicated)
- CPU: Intel(R) Core(TM) i9-9960X CPU @ 3.10GHz; RAM: 123.2 GiB
- Storage of the model files: nvme SAMSUNG MZVL2512HCJQ-00BL2 (nvme)
- OS: Ubuntu 26.04.1 LTS 7.0.0-38-generic; drivers: gpu-driver 595.99.02; runtime: cuda 13.2
- Strata commit: `9ae93f7b1fe989d04a8e404d32cddaeeff10505f`; engine 0.1.46.1; source build
- Background work: GPU at the stock 300 W power limit for this run; host: i9-9960X, 123 GiB RAM, Ubuntu 26.04, driver 595.99.02

## Model and configuration

- Model: qwen3.8-flash-next-ud-q4-kxl-baseline-21; quantization UD-Q4_K_XL (Unsloth dynamic, GGUF in place); revision not measured; source revision 9ae93f7b
- Files: `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` `4448186216b3af4c...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf` `3f342f1c1580473f...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf` `56758f40269cad5c...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf` `753bda48b98ba4f1...`
- Track: **custom** (Same-machine baseline for the tuned run in this PR: identical configuration except the experimental speed-projection control vector is off and the GPU power limit is the stock 300 W (the tuned run used 200 W). 1M context, KV int8 with 32768 resident cells, 80 GiB RAM budget (all experts resident), MTP draft layer. Not a setup-defaults run: the config is hand-written.)
- Context 1048576 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 4 min 9 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s |
| --- | --- | --- | --- | --- | --- |
| short | 73.3 (33.4-84.7) n=3 | 55.3 (52.0-61.3) n=3 | 1.08 (0.94-2.34) n=3 | 3.39 (2.60-4.02) n=3 | 7.0 (6.0-7.7) n=3 |
| prompt-4k | 450.8 (449.1-451.0) n=3 | 58.5 (52.1-62.7) n=3 | 9.11 (9.11-9.15) n=3 | 13.83 (13.05-14.61) n=2 | 17.9 (16.6-18.9) n=3 |
| prompt-16k | 854.2 (853.9-854.8) n=3 | 56.4 (51.4-62.5) n=3 | 19.22 (19.21-19.23) n=3 | 26.09 (22.91-28.50) n=3 | 28.3 (27.4-29.2) n=3 |

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
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The runner could not tell whether the server was freshly loaded.
