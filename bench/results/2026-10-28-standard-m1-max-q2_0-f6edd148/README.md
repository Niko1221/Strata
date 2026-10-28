# Strata standard benchmark: Apple M1 Max

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261028T164913Z-f6edd148`, started 2026-10-28T16:49:13Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: Apple M1 Max (64.0 GiB shared)
- CPU: Apple M1 Max; RAM: 64.0 GiB
- Storage of the model files: not measured
- OS: macOS 26.6.2; drivers: not measured; runtime: metal 26.6.2
- Strata commit: `a42cc3d7df70d1940de82cd0882f03e1f5dfc741`; engine 0.1.52-metal; release build
- Background work: Daily-driver macOS with Chrome, Safari, Adobe background services, WindowServer, Tailscale and oMLX present. oMLX health reported zero loaded models before the run; this benchmark sends no oMLX inference. Background CPU/GPU workload is not isolated or controlled.

## Model and configuration

- Model: qwen3.8-flash-next-q2_0; quantization Q2_0; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` `69820c02ec7d0b45...`; `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **default** (Official Strata 0.2.7 Terminal release at a42cc3d7df70d1940de82cd0882f03e1f5dfc741, Metal engine 0.1.52; unchanged setup-generated Q2_0 32768-context configuration, one slot, vision enabled with GPU helper, MTP spec 2, expert-cache 64, prefill auto, STRATA_METAL_STREAM=0; tuning skipped; fresh engine repeat run)
- Context 32768 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 7 min 33 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 38.6 (37.9-38.7) n=3 | 0.81 (0.80-0.82) n=3 | 2.23 (2.22-2.33) n=3 | 13.8 (13.8-13.9) n=3 | 0/3 |
| prompt-4k | 373.0 (365.6-377.0) n=3 | 37.8 (37.1-37.9) n=3 | 11.01 (10.89-11.23) n=3 | 15.08 (15.04-17.18) n=3 | 21.5 (21.1-25.0) n=3 | 1/3 |
| prompt-16k | 369.0 (367.7-369.6) n=3 | 36.0 (33.4-36.3) n=3 | 44.45 (44.38-44.60) n=3 | 52.08 (51.43-52.73) n=2 | 58.7 (58.6-59.7) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 5 | 4 | 0 | 0 |
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
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
