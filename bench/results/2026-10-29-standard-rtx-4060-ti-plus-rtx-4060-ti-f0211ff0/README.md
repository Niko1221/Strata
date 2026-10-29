# Strata standard benchmark: NVIDIA GeForce RTX 4060 Ti, NVIDIA GeForce RTX 4060 Ti

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261029T103105Z-f0211ff0`, started 2026-10-29T10:31:05Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 4060 Ti (16.0 GiB dedicated); NVIDIA GeForce RTX 4060 Ti (16.0 GiB dedicated)
- CPU: AMD Ryzen Threadripper PRO 3975WX 32-Cores; RAM: 125.6 GiB
- Storage of the model files: nvme CT500P310SSD8 (nvme)
- OS: Pop!_OS 24.04 LTS 7.1.5-76070105-generic; drivers: gpu-driver 610.57.04; runtime: cuda 13.3
- Strata commit: `46c152a1310f77cd2f0ce2c7cd213962df14fe83`; engine 0.1.53; source build
- Background work: FreeCAD GUI and Strata Studio open and idle; Codex CLI idle; no other inference client

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-iq4_xs; quantization UD-IQ4_XS; revision not measured; source revision 46c152a1
- Files: `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` `5ce89370720f8bf9...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf` `577a38a2392b40ca...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf` `d4634e6d84f0ebb0...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (v0.2.7.1 source build; production config: --kv int8 --kv-resident 32768, --remote-expert-opt --pcie-frac 0.00, --spec-min-p 0.50, MTP rt-iten (Italian+English draft subset, 54675 tokens), --vision on CPU, 262K context; env: STRATA_PROMPT_ATTN_IMMA=1 STRATA_HC_UPMIX=1 and the nine sm_86 switches of #4290)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 55 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 61.6 (59.9-64.9) n=3 | 1.39 (1.19-1.60) n=3 | 2.66 (2.51-3.59) n=3 | 6.6 (6.1-7.8) n=3 | 0/3 |
| prompt-4k | 904.5 (899.0-923.2) n=3 | 51.0 (50.9-64.6) n=3 | 4.56 (4.47-4.59) n=3 | 8.34 (8.34-8.34) n=1 | 14.6 (12.4-14.6) n=3 | 3/3 |
| prompt-16k | 1792.3 (1791.0-1811.7) n=3 | 56.7 (55.2-59.6) n=3 | 9.19 (9.09-9.19) n=3 | 16.32 (15.81-16.82) n=2 | 18.2 (17.6-18.4) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
