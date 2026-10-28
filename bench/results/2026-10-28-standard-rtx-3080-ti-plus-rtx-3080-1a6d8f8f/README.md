# Strata standard benchmark: NVIDIA GeForce RTX 3080 Ti, NVIDIA GeForce RTX 3080

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261028T175256Z-1a6d8f8f`, started 2026-10-28T17:52:56Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3080 Ti (12.0 GiB dedicated); NVIDIA GeForce RTX 3080 (10.0 GiB dedicated)
- CPU: Intel(R) Xeon(R) CPU E5-2695 v4 @ 2.10GHz; RAM: 127.8 GiB
- Storage of the model files: ssd ATA SAMSUNG MZ7PC128 (sata)
- OS: Windows 10.0.19045; drivers: gpu-driver 616.92; runtime: cuda 13.0, cuda-driver-supports 13.4
- Strata commit: `55a557bb6d77facc73d291c3f9894289496c770c`; engine 0.1.52; release build
- Background work: No intentional GPU workload; benchmark launched remotely via WinRM

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s; quantization IQ3_S; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e11...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `iq3_s-multigpu.bin` `f6cac04845cf82b5...`
- Track: **custom** (X99 RTX 3080 10GB + RTX 3080 Ti 12GB; layer_split auto; STRATA_SPLIT_OWN=auto; STRATA_SPLIT_MTP_BATCH=1; engine 0.1.52)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 17 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 108.5 (102.4-111.4) n=3 | 2.00 (1.81-2.00) n=3 | 3.11 (2.95-3.33) n=3 | 6.4 (5.3-6.6) n=3 | 1/3 |
| prompt-4k | 674.6 (672.0-682.7) n=3 | 98.8 (93.2-98.9) n=3 | 6.09 (6.01-6.12) n=3 | 9.23 (8.23-10.23) n=2 | 11.3 (11.2-11.5) n=3 | 3/3 |
| prompt-16k | 963.0 (958.2-965.1) n=3 | 104.9 (102.1-112.2) n=3 | 17.06 (17.02-17.16) n=3 | 19.34 (19.16-20.00) n=3 | 21.9 (21.7-22.0) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

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
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
