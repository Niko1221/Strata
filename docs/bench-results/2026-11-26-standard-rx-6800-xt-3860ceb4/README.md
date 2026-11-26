# Strata standard benchmark: AMD Radeon RX 6800 XT

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261126T171522Z-3860ceb4`, started 2026-11-26T17:15:22Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon RX 6800 XT (16.0 GiB dedicated)
- CPU: AMD Ryzen 7 5800X3D 8-Core Processor; RAM: 95.9 GiB
- Storage of the model files: ssd Samsung SSD 970 EVO Plus 500GB (nvme)
- OS: Windows 10.0.19045; drivers: gpu-driver 32.0.21045.16010; runtime: not measured
- Strata commit: `b28ca95e3e6847f2d5e317994f253b66716deeaa`; engine 0.1.69; release build
- Background work: Windows 10 desktop only, browser closed

## Model and configuration

- Model: qwen3.8-flash-next-unsloth-ud-q4_k_xl; quantization UD-Q4_K_XL; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` `4448186216b3af4c...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf` `3f342f1c1580473f...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf` `56758f40269cad5c...`; `Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf` `753bda48b98ba4f1...`; `draft_vocab_cyrillic.bin` `25d7fd1670a2e0ec...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (custom setup on RX 6800 XT (gfx1030, Windows HIP rdna2 engine): --mtp rt + draft_vocab_cyrillic.bin, --spec 4 --spec-min-p 0.5, --kv int8 --kv-resident 32768, --prefill 16384, --max-context 262144, --expert-cache auto, --vision, --prompt-cache-tail, conversation-cache 16 GiB, vram-reserve 1280 MiB, ram-keep 8 GiB; env STRATA_KV_HOST_DMA=1 STRATA_KQ256=1 STRATA_HOST_CORE=sibling STRATA_HC_PERSIST=1 STRATA_ROUTE_TAIL_SKIP=0 STRATA_SH_STREAM=0; server parallel=1)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 6 min 31 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 30.7 (28.9-30.9) n=3 | 3.06 (2.77-3.09) n=3 | 6.58 (5.75-8.75) n=3 | 14.8 (12.3-18.7) n=3 | 0/3 |
| prompt-4k | 434.6 (434.0-434.8) n=3 | 30.3 (28.3-31.3) n=3 | 9.48 (9.47-9.48) n=3 | 19.62 (17.17-22.06) n=2 | 26.3 (25.8-27.5) n=3 | 3/3 |
| prompt-16k | 706.1 (695.1-707.0) n=3 | 27.4 (27.1-27.4) n=3 | 23.28 (23.23-23.64) n=3 | 38.17 (38.17-38.17) n=1 | 42.1 (41.9-42.3) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
