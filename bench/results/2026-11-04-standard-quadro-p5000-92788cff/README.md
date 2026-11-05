# Strata standard benchmark: Quadro P5000

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261104T214459Z-92788cff`, started 2026-11-04T21:44:59Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: Quadro P5000 (16.0 GiB dedicated)
- CPU: Intel(R) Xeon(R) CPU           X5650  @ 2.67GHz; RAM: 110.0 GiB
- Storage of the model files: hdd WDC WD5000AAKS-75V0A0 (sata)
- OS: Ubuntu 24.04.5 LTS 6.8.0-94-generic; drivers: gpu-driver 580.178.04; runtime: cuda 13.0
- Strata commit: `974074c549754029882c1cea1ce0e1e7674d7ba2`; engine 0.1.56; release build
- Background work: SearXNG idle on the same host; ComfyUI stopped

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision not measured; source revision ffa80d7
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (Source build for a CPU without AVX (STRATA_ISA_FLOOR=none, SSE4.2) + CUDA 12.9 engine for sm_61; setup.sh choices --context 32768 --vision cpu, KV int8; port 8081; otherwise setup-generated config)
- Context 32768 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 15 min 46 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 10.2 (9.0-10.6) n=3 | 3.25 (2.72-3.50) n=3 | 10.43 (7.46-13.53) n=3 | 35.0 (33.1-36.5) n=3 | 0/3 |
| prompt-4k | 273.7 (267.1-276.8) n=3 | 9.6 (9.6-9.8) n=3 | 15.08 (14.91-15.51) n=3 | 48.54 (48.54-48.54) n=1 | 68.3 (67.3-68.9) n=3 | 3/3 |
| prompt-16k | 275.9 (275.6-276.3) n=3 | 10.4 (9.5-11.0) n=3 | 59.57 (59.49-59.61) n=3 | 100.53 (92.57-108.50) n=2 | 108.5 (106.0-113.1) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
