# Strata standard benchmark: Tesla V100-SXM2-32GB

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261105T182235Z-9395401e`, started 2026-11-05T18:22:35Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: Tesla V100-SXM2-32GB (32.0 GiB dedicated)
- CPU: AMD Ryzen 5 5600G with Radeon Graphics; RAM: 90.3 GiB
- Storage of the model files: nvme Samsung SSD 980 500GB (nvme)
- OS: Ubuntu 22.04.5 LTS 6.8.0-138-generic; drivers: gpu-driver 580.178.04; runtime: cuda 13.0
- Strata commit: `c3b3f5035506f3e966f1b4b7b188aeaabae92f1d`; engine 0.1.57; release build
- Background work: Ryzen 5 5600G desktop, display on iGPU Cezanne, no other GPU workloads; agent idle during the run

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs; quantization GSQ-RCO IQ3_XXS; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `219ea929900dfa9e...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `316b46f3a2dbd68c...`
- Track: **custom** (engine 0.1.57 linux-x86_64-cuda-cuda12 (feed seq 121); driver 580.178.04-server; Volta sm_70; env: STRATA_SM70_TABLE=1 FUSE_DECODE=1 ARGMAX_ROWS_SM70=1 GR_NORM_FAST=1; kv int8, kv-resident 32768, conv-cache 12 slots/16 GiB)
- Context 262144 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 25 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s | Capped |
| --- | --- | --- | --- | --- | --- | --- |
| short | n/a | 90.8 (83.0-93.5) n=3 | 0.71 (0.48-0.72) n=3 | 1.60 (1.23-4.77) n=3 | 6.1 (4.5-6.9) n=3 | 2/3 |
| prompt-4k | 1280.0 (1237.9-1288.0) n=3 | 72.8 (72.3-80.4) n=3 | 3.22 (3.20-3.33) n=3 | 8.18 (8.18-8.18) n=1 | 10.3 (9.6-10.3) n=3 | 3/3 |
| prompt-16k | 1584.4 (1558.3-1656.1) n=3 | 71.9 (71.7-75.1) n=3 | 10.37 (9.93-10.55) n=3 | 17.20 (17.20-17.20) n=1 | 17.5 (16.7-17.7) n=3 | 3/3 |

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
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered 1 request(s) before this run.

## Addendum: additional evidence on the same engine session, and limitations

- Optional correctness (per COMMUNITY_BENCHMARKS.md): `tools/needle_bench.py` on this server —
  32k/128k nominal (26,526 / 106,425 actual tokens) and 250k nominal (200,852 actual) at depths
  10/50/90: **9/9 FOUND**. Raw: `needle-32k-128k.json`, `needle-250k.json`. Fresh-read prefill at
  depth 10%: 1,560 / 1,364 / 1,271 tok/s at those lengths; depth 50/90 arms inherit the depth-10
  prefix and are reported but not used as throughput figures. The needle arms ran while desktop
  agent traffic was possible; the protocol performance stages above ran idle after a fresh engine
  start (`strata runtime restart` before `strata bench run --url`).
- This card is sm_70 Volta on the ready-made `linux-x86_64-cuda-cuda12` engine 0.1.57 from the
  signed feed (GitHub release assets for that engine 404); driver `nvidia-driver-580-server`
  580.178.04 — the last NVIDIA branch supporting Volta. Track is custom: bit-exact sm_70 switches
  (`STRATA_SM70_TABLE/FUSE_DECODE/ARGMAX_ROWS_SM70/GR_NORM_FAST`), conversation-cache user_args.
- PCIe link: the card sits on an SXM2-to-PCIe adapter. The first seating negotiated `x1` and we
  initially measured decode 47.5 tok/s on it; after reseating, `8GT/s x16 (ok)` — all figures in
  this report are the x16 state. ECC enabled, 0 errors; power limit 300 W; no throttling flags.
- Limitations: single machine, single run of the protocol (n=3 per case as the protocol defines);
  repeated decode-anchor series beyond the protocol not included here yet; model revision fields
  are `null` with reason (community pack); the 4K/16K performance arms ran to the 512-token cap
  while still reasoning (standard for this draft, see `Capped`).
- Operator journal (append-only, failures included): https://github.com/ildargiv/ai-flight-log-v100
