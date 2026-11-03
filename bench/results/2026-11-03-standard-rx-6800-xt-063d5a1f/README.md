# Strata standard benchmark: AMD Radeon RX 6800 XT

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261103T151248Z-063d5a1f`, started 2026-11-03T15:12:48Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon RX 6800 XT
- CPU: AMD <host> 7 5700X 8-Core Processor; RAM: 60.7 GiB
- Storage of the model files: nvme CT1000P3PSSD8 (nvme)
- OS: Ubuntu 26.04.1 LTS 7.0.0-42-generic; drivers: gpu-driver 7.0.0-42-generic; runtime: rocm 7.2.4
- Strata commit: `ffa80d73133ea5c548aefe57889a16e8085a938f`; engine 0.1.56; source build
- Background work: not measured

## Model and configuration

- Model: swift-1.5-iq3_xxs; quantization IQ3_XXS; revision 99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d; source revision not measured
- Files: `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `3bddaa667c750f63...`; `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `b0b15f782af71eb4...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (release 0.2.10 source checkout ffa80d73, local HIP gfx1030 engine built by setup.sh; tuned config: --prefill auto:16384 --kv-resident 32768 --vram-reserve-mib 768 --kv int8 --max-context 262144, F16 route + gfx1030 rocBLAS table + GDN env switches, MTP draft vocab, expert profile)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 34 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 47.6 (45.8-51.5) n=3 | 1.02 (0.89-1.06) n=3 | 5.29 (3.03-7.28) n=3 | 11.2 (7.4-11.7) n=3 | 1/3 |
| prompt-4k | 1038.1 (1032.6-1060.8) n=3 | 47.8 (44.2-50.0) n=3 | 3.98 (3.89-4.00) n=3 | 9.75 (9.48-10.03) n=2 | 14.7 (14.2-15.5) n=3 | 3/3 |
| prompt-16k | 1246.3 (1243.2-1247.1) n=3 | 49.4 (43.5-51.2) n=3 | 13.19 (13.18-13.22) n=3 | 18.46 (18.00-23.38) n=3 | 23.5 (23.2-24.9) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
