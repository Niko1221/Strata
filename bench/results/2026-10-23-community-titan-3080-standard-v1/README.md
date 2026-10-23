# Strata standard benchmark: NVIDIA GeForce RTX 3080, NVIDIA TITAN RTX

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261023T225256Z-919b5db8`, started 2026-10-23T22:52:56Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3080; NVIDIA TITAN RTX
- CPU: Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz; RAM: 125.7 GiB
- Storage of the model files: nvme SAMSUNG MZVL21T0HCLR-00B00 (nvme)
- OS: Ubuntu 24.04.5 LTS 6.8.0-147-generic; drivers: not measured; runtime: not measured
- Strata commit: `5c8dd322528cc84f68f461ad54a506b9da8fa8d4`; engine 0.1.48; source build
- Background work: gateway idle during bench; model download paused for clean measurement

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-iq4_xs; quantization UD-IQ4_XS; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` `5ce89370720f8bf9...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf` `577a38a2392b40ca...`; `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf` `d4634e6d84f0ebb0...`
- Track: **custom** (K32 dual live caps 210-280W engine 0.1.48)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 36 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 94.1 (94.0-103.3) n=3 | 1.34 (1.24-1.37) n=3 | 2.40 (2.00-4.45) n=3 | 4.2 (4.0-6.8) n=3 | 1/3 |
| prompt-4k | 846.0 (843.6-846.2) n=3 | 88.2 (83.4-101.8) n=3 | 4.86 (4.86-4.88) n=3 | 7.62 (6.59-8.65) n=2 | 10.7 (9.7-11.0) n=3 | 2/3 |
| prompt-16k | 1507.3 (1504.9-1507.5) n=3 | 82.3 (81.9-90.8) n=3 | 10.91 (10.91-10.93) n=3 | 15.23 (15.23-15.23) n=1 | 17.1 (16.5-17.2) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 3 | 9 | 0 | 0 |

Sanity checks: **11 of 12**.

| Case | Result | Reason |
| --- | --- | --- |
| I1 | pass | the expected JSON value |
| I2 | fail | the JSON value differs from the expected value |
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
| 1 | 26.5345 | 1974 | 74.393714 | 0 |
| 2 | 28.572197 | 2032 | 71.118087 | 0 |
| 3 | 27.59006 | 1924 | 69.73526 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c1: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c4: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 393 request(s) before this run.
