# Community benchmark: RTX 4070 Ti 12 GB + Core i7-13700K — Ornith-1.5-35B-A3B Q4_K_M on the q36 engine (2026-10-21)

First community datapoint for Ornith-1.5-35B-A3B (ornith-ai, MIT) on the q36 engine, measured while bringing the
model up on a single 12 GB card. The runs double as the before/after evidence for the quantized KV caches
(`--kv int8 / fp8`, proposed in the companion engine PR) and for Q4_K expert down projections: at these context
lengths the fp16 KV cache does not fit the card at all, so every row below is a quantized-KV run.

## Hardware and software

- **GPU:** NVIDIA GeForce RTX 4070 Ti, 12,282 MiB, driver 617.14 (single card, sm_89)
- **CPU:** 13th Gen Intel Core i7-13700K
- **RAM:** 64 GB (63.8 GiB usable)
- **OS:** Windows 11 (10.0.26200)
- **Engine:** `strata-q36` (src/qwen35) built from source — MSVC 2022 BuildTools, Ninja, CUDA 12.8, sm_89.
  Two builds that day; every record carries its binary sha256 (see `results.json` → `runs[].binary.sha256`):
  `024f24d4…` (quantized KV, before the attention rework; the 150K/180K rows) and `5d169a8d…` (quantized KV +
  the chunked-attention kernels; the 30K row).
- **Model:** Ornith-1.5-35B-A3B Q4_K_M GGUF, 20.22 GiB — `qwen35moe`: 41 layers, 256 routed experts (8 active),
  shared expert, GQA 16:2, head dim 256, SSM hybrid with full attention every 4th layer, native 262,144 context,
  one MTP draft layer. Q4_K down projections on 20 layers (the companion kernels PR).
- **Sampling:** greedy (temperature 0, top_p 0.8, top_k 20), penalties neutral, reasoning budget 0.

## Method

Local harness (one request per row, fresh engine start per row, `prompt_reused = 0` in every record, needle
retrieval expected in each fixture's answer):

- prefill = the engine's `timings.prompt_per_second`
- decode = the engine's `timings.predicted_per_second` (produced tokens over decode time; with MTP the accepted
  draft tokens are included)
- draft = accepted/offered, from the engine's log line

Matrix rows are single runs; the 52K A/B pairs are 2 back-to-back repeats per KV type. `results.json` holds the
complete records (config, per-run binary and model sha256, timings, needle flags). Local absolute paths were
shortened to `<workdir>` and generated text bodies dropped from the records; nothing else was edited.

## Results — fresh long prompts

All rows: `--expert-vram 3.72`, chunk 1024, `prompt_reused = 0`, needle PASS, engine `024f24d4…` unless noted.

| prompt tokens | KV | MTP | prefill tok/s | decode tok/s | draft |
| ---: | --- | ---: | ---: | ---: | --- |
| 150,410 | int8 | 0 | 575.9 | 16.0 | – |
| 150,410 | fp8 | 0 | 579.4 | 13.5 | – |
| 180,067 | int8 | 0 | 475.0 | 13.3 | – |
| 180,067 | fp8 | 0 | 475.4 | 12.0 | – |
| 150,410 | int8 | 0 (ev 4.5) | 573.3 | 15.0 | – |
| 150,410 | int8 | 0 (ev 5.0) | 573.9 | 15.1 | – |
| 150,410 | int8 | 2 | 533.3 | 13.7 | 44/56 |
| 150,410 | int8 | 0 (pool workers 8) | 568.6 | 14.3 | – |
| 150,410 | int8 | 0 (chunk 2048) | 683.1 | 15.0 | – |
| 29,746 | int8 | 2 (ev 5.5, chunk 2048, engine `5d169a8d…`) | 2090.9 | 79.4 | 41/46 |

52,460-token prompt A/B pairs (window 131,072, 2 repeats per KV type, prefill only):

| KV | run 1 tok/s | run 2 tok/s |
| --- | ---: | ---: |
| int8 | 1067.1 | 1064.3 |
| fp8 | 1076.1 | 1076.0 |

What the rows say (single runs, treat as indicative):

- **int8 vs fp8 at equal settings:** prefill is a wash (579.4 vs 575.9; 475.4 vs 475.0; the 52K pairs likewise),
  decode is 11–16% slower on fp8 (13.5 vs 16.0; 12.0 vs 13.3) — the per-token scales cost more in the attention
  path than int8's per-64 scales.
- **Chunk 2048 vs 1024** (150K, ev 3.72): prefill 683.1 against 575.9, +19%.
- **Expert VRAM 4.5 / 5.0 over 3.72** (150K, MTP 0): decode 15.0 / 15.1 against 16.0 — no gain here.
- **MTP 2 at 150K:** 13.7 against 16.0 — negative at this depth on the pre-rework binary; at 29,746 tokens
  (newer binary) it decodes at 79.4 with 41/46 accepted. Different engine build and prompt length, so these two
  rows are not a controlled MTP verdict.

## Correctness

- Needle retrieval: 10/10 PASS (every matrix row above, including both 180,067-token rows).
- MTP identity gate at 2K, greedy: the mtp0 and mtp1 runs produced byte-identical answers (content sha256 equal;
  `results.json` → `quality.mtp_identity_2k`).

## Limits

- Single runs per matrix row (the A/B pairs are the only repeats); daytime rows shared the machine with light
  interactive use.
- Two engine builds across the day (sha256 per record); the 150K/180K rows predate the chunked-attention rework.
- The harness and fixtures are local and not part of this repo; the records carry everything needed to re-run the
  shapes by hand.
- Host RAM pressure: with the 20.22 GiB GGUF mapped and experts pinned, the machine was near its 64 GB ceiling
  during the 180K rows; no measurement of that effect is included.
