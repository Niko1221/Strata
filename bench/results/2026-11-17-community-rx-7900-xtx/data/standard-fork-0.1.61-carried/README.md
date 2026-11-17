# Strata standard benchmark: N/A

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261117T020022Z-c6def268`, started 2026-11-17T02:00:22Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: N/A
- CPU: AMD Ryzen 9 7900X 12-Core Processor; RAM: 93.9 GiB
- Storage of the model files: nvme CT1000T710SSD8 (nvme)
- OS: <host> Linux 45 (KDE Plasma Desktop Edition) 7.3.3-300.fc45.x86_64; drivers: gpu-driver 7.3.3-300.fc45.x86_64; runtime: not measured
- Strata commit: `addf31dca80ca41c36cadcf9f210b9272422492c`; engine 0.1.61; source build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-ud-iq3xxs; quantization UD-IQ3_XXS (iq_pack, HC tensors compat-bf16); revision not measured; source revision v0.2.15 + carried fork line (addf31dca; github.com/xyzzing/qwen-rdna3-optimization-ledger)
- Files: `Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf` `9d5644b371ffac92...`; `Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00004.gguf` `aacaff3bdb9032c2...`; `Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00004.gguf` `ef8f14e790ab471b...`; `Qwen3.8-Flash-Next-UD-IQ3_XXS-00004-of-00004.gguf` `5de8b5dbf89409ac...`
- Track: **default** (engine v0.2.15 + carried fork line (addf31dca; github.com/xyzzing/qwen-rdna3-optimization-ledger); production env verbatim (gfx1100 knob set, see github.com/xyzzing/qwen-rdna3-optimization-ledger); UD-IQ3_XXS pack (compat-bf16); fresh server, zero prior requests)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 5 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 72.1 (43.7-96.2) n=3 | 0.76 (0.65-0.76) n=3 | 3.28 (1.45-10.55) n=3 | 5.9 (5.1-12.5) n=3 | 1/3 |
| prompt-4k | 2096.6 (2087.4-2155.2) n=3 | 74.2 (64.5-78.0) n=3 | 1.97 (1.92-1.98) n=3 | 6.70 (4.95-7.49) n=3 | 8.9 (8.5-9.8) n=3 | 3/3 |
| prompt-16k | 3302.1 (3175.2-3303.0) n=3 | 75.8 (51.4-84.1) n=3 | 4.99 (4.99-5.19) n=3 | 9.55 (8.53-11.50) n=3 | 11.7 (11.1-15.1) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 2 | 7 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 4 | 8 | 0 | 0 |

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

Concurrency rounds (four requests released together):

| Round | Wall s | Output tokens | Aggregate tok/s | Failed |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 31.669996 | 1954 | 61.698776 | 0 |
| 2 | 26.566874 | 2007 | 75.545207 | 0 |
| 3 | 25.400141 | 1938 | 76.298789 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
