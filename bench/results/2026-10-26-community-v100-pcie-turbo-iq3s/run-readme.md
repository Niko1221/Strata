# Strata standard benchmark: Tesla V100-PCIE-32GB

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261026T073044Z-b33207f9`, started 2026-10-26T07:30:44Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: Tesla V100-PCIE-32GB (32.0 GiB dedicated)
- CPU: Intel(R) Xeon(R) CPU E5-2696 v4 @ 2.20GHz; RAM: 127.9 GiB
- Storage of the model files: ssd WDC WDS240G2G0C-00AJM0 (nvme)
- OS: Windows 10.0.22631; drivers: gpu-driver 576.57; runtime: cuda 12.9
- Strata commit: `7ddc17a38fe7f31637e895ecfba58c2cd43d65ec`; engine 0.1.48.1; source build
- Background work: not measured

## Model and configuration

- Model: Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S; quantization IQ3_S; revision b1b82c36770c826c41e1874f250228048cdb3f5c; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf` `78706315c2fe9127...`; `Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (engine 0.1.48.1 (2026-10-25 build, sm_70), STRATA_SM70_TABLE=1, ple-io direct on NVMe, MTP rt, kv int8 resident 32768, spec 4 min-p 0.75, DavidAU Turbo Brilliance 50x abliterated IQ3_S (tree: Qwen -> ISTA GSQ-RCO -> SC117 abliterated -> DavidAU))
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 34 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 74.7 (74.2-77.6) n=3 | 0.90 (0.75-0.94) n=3 | 2.66 (2.22-3.79) n=3 | 6.3 (5.2-6.4) n=3 | 0/3 |
| prompt-4k | 1221.4 (1218.8-1224.3) n=3 | 73.5 (68.6-79.2) n=3 | 3.39 (3.39-3.42) n=3 | 7.91 (6.53-9.29) n=2 | 10.4 (9.8-10.8) n=3 | 3/3 |
| prompt-16k | 1624.9 (1623.7-1627.1) n=3 | 65.9 (63.8-72.3) n=3 | 10.14 (10.13-10.15) n=3 | 15.25 (15.25-15.25) n=1 | 17.9 (17.2-18.1) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
