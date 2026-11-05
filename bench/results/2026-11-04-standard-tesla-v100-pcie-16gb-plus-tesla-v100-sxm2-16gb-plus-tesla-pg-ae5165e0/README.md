# Strata standard benchmark: Tesla V100-PCIE-16GB, Tesla V100-SXM2-16GB, Tesla PG500-216

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261104T034410Z-ae5165e0`, started 2026-11-04T03:44:10Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: Tesla V100-PCIE-16GB (16.0 GiB dedicated); Tesla V100-SXM2-16GB (16.0 GiB dedicated); Tesla PG500-216 (32.0 GiB dedicated)
- CPU: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz; RAM: 62.8 GiB
- Storage of the model files: nvme Fanxiang S500PRO 512GB (nvme)
- OS: Ubuntu 22.04.5 LTS 6.8.0-138-generic; drivers: gpu-driver 580.178.04; runtime: cuda 13.0
- Strata commit: `4b9e05a2a806a3fe24f7b2560d98626c37dade35`; engine 0.1.51; source build
- Background work: none

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-q4_k_xl; quantization UD-Q4_K_XL; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` `4448186216b3af4c...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf` `3f342f1c1580473f...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf` `56758f40269cad5c...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf` `753bda48b98ba4f1...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (3-card layer split (16GB PCIe + 16GB SXM2 + 32GB), trim-stage-weights, 256K context, budget 40, HC_Q8=0, SM70_TABLE=1; official UD-Q4_K_XL (not the abliterated v2 default))
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 57 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 62.4 (60.3-64.4) n=3 | 1.52 (1.31-1.55) n=3 | 2.98 (2.67-3.18) n=3 | 6.7 (6.4-7.5) n=3 | 0/3 |
| prompt-4k | 826.2 (825.2-840.7) n=3 | 60.4 (56.7-61.2) n=3 | 4.99 (4.90-5.01) n=3 | 11.25 (11.25-11.25) n=1 | 13.4 (13.2-14.0) n=3 | 3/3 |
| prompt-16k | 1652.4 (1649.9-1653.3) n=3 | 64.1 (62.9-67.0) n=3 | 9.96 (9.96-9.99) n=3 | 14.98 (13.57-15.35) n=3 | 18.0 (17.6-18.1) n=3 | 3/3 |

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
