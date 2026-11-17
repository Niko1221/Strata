# 2026-11-17: one AMD Radeon RX 7900 XTX 24 GB (gfx1100 / RDNA3), Ryzen 9 7900X, 96 GB RAM

Strata **0.1.61** (upstream v0.2.15, tag `c77572806`), **Unsloth UD-IQ3_XXS**
(Qwen3.8 Flash-Next MoE), 131,072-token context, Linux (Fedora 45, kernel 7.3.3),
HIP SDK nightly 10.2.0a (20260930) + hipBLASLt tuning table for gfx1100.

This is the first discrete-RDNA3 entry in this matrix, and it carries **two engines
built from the same source tree and measured in the same window** so the fork delta is
data, not prose:

| engine | source revision | what it is |
| --- | --- | --- |
| vanilla | v0.2.15 (`c77572806`), no fork commits | the clean comparable number |
| fork | v0.2.15 + the carried fork line (`addf31dca`) | what production runs here; the two-month knob campaign behind it is public at [xyzzing/qwen-rdna3-optimization-ledger](https://github.com/xyzzing/qwen-rdna3-optimization-ledger) (~129 recorded experiments, ~20 adopted, ~85 rejected with paired measurements; `RDNA3-DEPLOYMENT.md` in this folder is the resulting production knob matrix) |

## Standard benchmark (protocol `strata-qwen38-thinking-v1-draft.1`, results.json in `data/`)

Fresh server (zero prior requests), thinking on, the protocol's frozen fixtures and
sampling; `--optional-concurrency` added. Quality: **12/12 sanity checks passed on
both engines** (instruction 4/4, retrieval 4/4, tool calls 4/4). Medians of 3, min-max
in brackets:

| case | metric | vanilla | fork |
| --- | --- | --- | --- |
| short | prompt tok/s | 107.2 [105.4–124.6] | 104.2 [103.7–122.7] |
| short | decode tok/s | 82.9 [76.6–92.4] | 72.1 [43.7–96.2] |
| prompt-4k | prompt tok/s | 2087.1 [2083.2–2158.5] | 2096.6 [2087.4–2155.2] |
| prompt-4k | decode tok/s | 85.7 [74.9–94.1] | 74.2 [64.5–78.0] |
| prompt-16k | prompt tok/s | 3310.9 [3302.5–3323.6] | 3302.1 [3175.2–3303.0] |
| prompt-16k | decode tok/s | 68.6 [58.0–76.0] | 75.8 [51.4–84.1] |
| concurrency (4 at once) | tok/s per round | 67.7 / 70.2 / 73.2 | see data/standard-fork |

Read honestly: at the protocol's short contexts the two engines sit within run-to-run
spread of each other (the fork's short-decode median carries one 43.7 tok/s outlier;
its spread is wide and we do not claim a win from it). The fork's measured edge on this
card is at depth and on prefill — see the sweep below and the paired curve in the ledger
repo (decode +4.9% at 128K, +8.3% at 262K vs the prior production pack, same engine).

## Engine-level sweep (our production workload, not protocol fixtures; 3 runs each)

Pre-parsed 4,096 / 32,768 / 131,072-token prompts, 256 new tokens, one-shot engine
cells (`scripts/decode-curve-0161.sh` machinery; raw logs in `data/sweep-*.log`):

| prompt tokens | vanilla prefill / decode (medians) | fork prefill / decode (medians) |
| --- | --- | --- |
| 4k | 1666 / 80.2 tok/s | 1659 / 85.5 tok/s |
| 32k | 2142 [1814–2578] / 74.1 tok/s | 2540 / 72.2 tok/s |
| 128k | 2540 / 71.8 tok/s | 2522 / 70.1 tok/s |

The fork's 32K prefill median (+19% over vanilla) rides on vanilla's wide r1 outlier —
treat it as directional, not a claim. Recall: the standard benchmark's retrieval checks
(4/4) passed on both engines; we did not run a separate needle suite for this report.

## Settings (identical for both engines)

`--spec 5` (MTP drafter, sidecar), `--kv int8`, `--kv-resident 49152`,
`--prefill 8192` (sweep) / serve defaults (standard), `--expert-cache auto`,
`--lookup-chain 2`, `--spec-min-p 0.70`, hipBLASLt tuning table. The env knob set is
reproduced in `config/serve-config-sanitized.json` and explained, with per-knob paired
measurements, in `RDNA3-DEPLOYMENT.md` (this folder) and the ledger repo above.

## Known limits

- One machine, one card revision, desktop not fully quiet (browsers open, memory-gated
  window: cells ran only above 68 GiB MemAvailable).
- The card's VRAM size and power limit could not be auto-recorded (the runner says so in
  `missing`); they are 24 GiB and 355 W BIOS limit, typed here by hand.
- The UD-IQ3_XXS pack was built with `--compat-bf16` (195 small tensors dequantized
  Q8_0→BF16, audited in the pack's `conversions.json`: max abs err 0.0204; expert and
  PLE bytes unchanged).
- The protocol is still a draft; these results use the draft as-is and are offered as
  the AMD-discrete pilot data point its activation checklist asks for.

## Reproduce

`scripts/` carries the exact drivers (standard-bench window + the sweep cell runner).
Model: `unsloth/Qwen3.8-Flash-Next-GGUF` revision `f5d175de` (UD-IQ3_XXS, 4 shards,
sha256-verified) packed with the tree's `tools/iq_pack.py`.
