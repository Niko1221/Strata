# Strata standard benchmark: AMD Radeon(TM) Graphics

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261106T201931Z-42a46d4e`, started 2026-11-06T20:19:31Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Context

This is round 2 of [the helper-cache benchmark](../2026-11-06-community-6800xt-6600-helper-cache/README.md):
identical machine, model, engine and helper shape, with `--remote-expert-opt` added on top. The flag
makes the 6600 sum the experts it holds and return one vector per token, and it switches
`--pipeline-windows` off ([SECOND_GPU.md](../../../docs/SECOND_GPU.md)). It was expected to help on this
path, as it does on the CUDA machines in that document. It does not: the prompt is unchanged and
decode falls by 10 to 27 percent.

| Metric | Helper cache (round 1) | + `--remote-expert-opt` | Change |
| --- | ---: | ---: | ---: |
| prompt-4k, tok/s | 631.2 | 629.3 | ±0 |
| prompt-16k, tok/s | 772.4 | 762.2 | -1% |
| decode short, tok/s | 52.6 | 38.4 | -27% |
| decode-4k, tok/s | 43.5 | 35.8 | -18% |
| decode-16k, tok/s | 38.7 | 34.7 | -10% |
| TTFT-4k, s | 6.52 | 6.56 | ±0 |
| total-4k, s | 18.1 | 21.0 | +16% |
| total-16k, s | 34.1 | 36.3 | +6% |

Against the layer split of the same day the flag still wins prefill (762 against 640 tok/s at 16K) but
loses decode at 16K (34.7 against 38.6), so the plain helper shape dominates it on this machine.

Engine differences in this run, from the log: the primary cache shrinks from 3,983 to 3,940 slots
(6.42 GiB), the prompt chunk stays 6,144 tokens with a 96-slot ring, and `strata serve` reports
`adaptive overlap: off: not with --remote-expert-opt`. The 6600 keeps its 4,097 experts (6.57 GiB).
Sanity checks passed 12 of 12; one short-prompt request stalled for 123 s while the other two took
1.62 s, and that sample is the reason the short-case range is wide.

## Hardware and software

- GPU: AMD Radeon(TM) Graphics (2.0 GiB dedicated)
- CPU: AMD Ryzen 7 9700X 8-Core Processor; RAM: 61.7 GiB
- Storage of the model files: ssd NVMe KINGSTON SFYRD20 (nvme)
- OS: Windows 10.0.26300; drivers: gpu-driver 32.0.21045.16010; runtime: not measured
- Strata commit: `c3b3f5035506f3e966f1b4b7b188aeaabae92f1d`; engine 0.1.57; release build
- Background work: not measured

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs; quantization IQ3_XXS; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `219ea929900dfa9e...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (no notes given by the operator)
- Context 131072 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 7 min 10 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 38.4 (36.1-44.7) n=3 | 1.62 (1.62-123.02) n=3 | 4.36 (3.69-125.67) n=3 | 13.0 (10.2-133.4) n=3 | 1/3 |
| prompt-4k | 629.3 (619.0-629.8) n=3 | 35.8 (33.6-40.4) n=3 | 6.56 (6.56-6.67) n=3 | 13.48 (12.39-14.56) n=2 | 21.0 (19.2-21.8) n=3 | 3/3 |
| prompt-16k | 762.2 (759.7-768.0) n=3 | 34.7 (34.0-39.4) n=3 | 21.55 (21.41-21.64) n=3 | 29.69 (29.69-29.69) n=1 | 36.3 (34.5-36.4) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 2 request(s) before this run.
