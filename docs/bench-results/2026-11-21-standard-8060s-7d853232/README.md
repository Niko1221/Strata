# Strata standard benchmark: AMD Radeon

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261121T005308Z-7d853232`, started 2026-11-21T00:53:08Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: AMD Radeon
- CPU: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S; RAM: 61.9 GiB
- Storage of the model files: nvme KBG60ZNS1T02 KIOXIA (nvme)
- OS: Ubuntu 26.04.1 LTS 7.0.0-43-generic; drivers: not measured; runtime: not measured
- Strata commit: `c77572806996349b49995e4f98d41b5294d36a21`; engine 0.1.60; release build
- Background work: headless Ubuntu 26.04.1 (multi-user.target); no desktop session; no other GPU work

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs; quantization ISTA-DASLab Qwen3.8-Flash-Next-GSQ-RCO IQ3_XXS; revision ed59f92082b1e93c0e96d60a8b11aab089b52f09; source revision de4b8e4d43b917e7706784d8bb445c9af86a3540
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `219ea929900dfa9e...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (release v0.2.15 (engine 0.1.60 hip-halo); headless Ubuntu 26.04.1 multi-user, no desktop; --context 256k --kv int8 --vision gpu --drafting mtp (MTP layer packed locally from the BF16 checkpoint with tools/mtp_pack.py --experts q2_0 + tools/mtp_rt.py); --expert-cache 2000 --vram-reserve-mib 1536 --prefill 8192 --mtp-window 8192 --spec-min-p 0.5 --conversation-cache-mib 1024; STRATA_RESIDENT_HEADROOM_GIB=6; fast config env (PF_FUSED/PF_GEMM/HC_UPMIX/PA_FAST/HIP_WMMA/SELECT_WMMA/PF_SWITCH_MIN_T/PREFILL_STREAM_MIN/REFILL_PARTS + hipBLASLt gfx1151 table); amd_iommu=off; BIOS carve-out 512 MiB)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 4 min 44 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 42.4 (39.8-46.6) n=3 | 0.81 (0.70-0.81) n=3 | 2.84 (2.00-2.98) n=3 | 11.7 (9.5-12.9) n=3 | 2/3 |
| prompt-4k | 939.9 (840.5-962.8) n=3 | 35.2 (35.1-40.7) n=3 | 4.39 (4.29-4.91) n=3 | 11.65 (11.65-11.65) n=1 | 18.8 (17.5-18.9) n=3 | 3/3 |
| prompt-16k | 1180.5 (1155.8-1217.4) n=3 | 34.0 (34.0-39.3) n=3 | 13.92 (13.51-14.22) n=3 | 20.35 (20.35-20.35) n=1 | 28.5 (26.9-29.2) n=3 | 3/3 |

Cells are median (min-max) n=samples. Short is a prompt of about 80 tokens: its prompt rate is dominated by the fixed cost of a request, so it is not shown; read prompt speed from prompt-4k and prompt-16k. Capped = requests that reached the 512-token cap, reasoning and answer together. A capped request's total time is the time to the cap, not to an answer, and it has no first-answer time when it was still reasoning.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 1 | 8 | 0 | 0 |
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
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
