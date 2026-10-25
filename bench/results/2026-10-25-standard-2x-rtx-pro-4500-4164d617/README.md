# Strata standard benchmark: NVIDIA RTX PRO 4500 Blackwell

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261025T214240Z-4164d617`, started 2026-10-25T21:42:40Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA RTX PRO 4500 Blackwell (31.9 GiB dedicated)
- CPU: AMD Ryzen Threadripper 7960X 24-Cores; RAM: 61.0 GiB
- Storage of the model files: nvme SSD_M.2_PCIe4_4TB_InnovationIT_Y (nvme)
- OS: Ubuntu 26.04.1 LTS 7.0.0-38-generic; drivers: gpu-driver 595.91.07; runtime: cuda 13.2
- Strata commit: `562a0c2ed7169ff6373f893b162106d55a828297`; engine 0.1.50; release build
- Background work: rustdesk --server (idle remote-desktop service); GNOME desktop session (idle)

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs; quantization IQ3_XXS; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `219ea929900dfa9e...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **default** (0.2.6 release (Terminal 0.1.51, engine 0.1.50), setup defaults (strata launch, 128K, parallel auto, tune skip), official ISTA IQ3_XXS, 2x RTX PRO 4500)
- Context 131072 tokens; server capacity 4 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 0 min 59 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 185.7 (170.1-190.4) n=3 | 0.37 (0.31-0.41) n=3 | 1.18 (0.71-2.16) n=3 | 2.4 (2.1-3.4) n=3 | 1/3 |
| prompt-4k | 3173.9 (3152.1-3346.1) n=3 | 174.3 (159.9-174.5) n=3 | 1.30 (1.24-1.31) n=3 | 3.11 (3.02-3.21) n=2 | 4.2 (4.2-4.4) n=3 | 3/3 |
| prompt-16k | 4796.3 (4783.8-4831.4) n=3 | 152.5 (146.5-169.1) n=3 | 3.43 (3.41-3.44) n=3 | 5.85 (5.14-6.56) n=2 | 6.8 (6.5-6.9) n=3 | 3/3 |

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
| 1 | 9.986575 | 1920 | 192.258107 | 0 |
| 2 | 9.831011 | 1859 | 189.095506 | 0 |
| 3 | 10.176596 | 1997 | 196.234576 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r2-c4: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c4: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 1 request(s) before this run.
