# 2026-10-18 — community 2× PCIe V100 (PG500), Qwen3.8-Flash-Next IQ3_XXS / UD-IQ4_XS, same-engine single-card comparison

Two Tesla PG500-216 (V100-class, 32 GB, 1,132 GB/s HBM2) on PCIe 3.0 x16, TCC mode, layer
split — measured against **the same engine on one of the same cards**, same suite, n=2 per
arm. That paired design is the point: the community has dual-GPU DDR4 reports (#1300, #1028)
but no same-machine single-card control, so the true layer-split gain was not visible before.
Also the first Windows + Volta (sm_70) dual-GPU datapoint we know of, and a `--pipeline-windows 2`
arm on NVIDIA (opt-in there), which is where the biggest decode number below comes from.

## Machine
- 2× Tesla PG500-216 32 GB (V100-class, 72 SMs, 1,132 GB/s), PCIe 3.0 x16 each (probed
  11.6-12.4 GB/s), TCC, driver 581.80 kernel + 582.78 package, GTX 1060 display card unused
  by the engine
- 2× Xeon E5-2696 v3 (18c/36t each, AVX2, no AVX-512), 128 GB DDR3L-1600 ECC (8 channels
  per socket) — the host-bandwidth-limited platform we reported single-card numbers for
  before (#1355); the expert cache is what decides how much of that bandwidth is still touched
- Windows 10 Enterprise LTSC 21H2, SATA-class SSD

## Engine / config
- Strata **0.1.44.1** (official prebuilt `strata-windows-x64-cuda12.zip`, sm_70 path).
- IQ3_XXS 262K: `--expert-cache auto --prefill auto --spec 4 --mtp --max-context 262144
  --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 2000 --pcie-frac 0.33
  --spec-min-p 0.70` + `STRATA_SM70_TABLE=1` (#1401), split auto → **K=24**, each card
  12,288 slots → **the caches hold 24,576 of 24,576 profiled pairs (100% of the routed mass)**.
- UD-IQ4_XS 131K: same shape + `--resident-budget-gib 55 --vram-reserve-mib 700
  --spec-min-p 0.5 --pool-workers 18`; split auto → **K=24**, caches hold 21,849-22,606 of
  24,576 pairs (**~99.8% of the routed mass**).
- The `--pipeline-windows 2` arm adds just that flag to the calibrated dual XXS config
  (272 MiB kept out of each card's cache; log: "two verifiers per stage, 2 stages overlap
  across windows (decode too)").

## Method
Same suite as our earlier single-card reports (`remeasure_0144.py` in the flash-ab folder:
short / 2-4k-token long-form / 4.3K-token document leg / H1 math probe with
`reasoning_budget_tokens: 4000`; temp 0.6, seed 42; engine-side timings). n=2 runs per arm,
both published in the JSONs; the table shows means. Single-card baselines re-measured **on
0.1.44.1 the same day** (UD) or taken from the same-day 0.1.44.1 report (XXS, v100s{a,b}) —
no cross-version comparisons anywhere.

## Paired results (decode tok/s; single card → dual)

| leg | XXS 1× | XXS 2× | XXS 2× +pw2 | UD 1× | UD 2× |
|---|---:|---:|---:|---:|---:|
| short (~150 tok out) | 65.2 | 83.5 (**1.28×**) | 87.7 (**1.35×**) | 36.7 | 56.1 (**1.53×**) |
| long (2-4k tok out) | 70.8 | 81.0 (1.14×) | 83.2 (1.17×) | 44.8 | 56.4 (1.26×) |
| 4.3K doc, decode | 77.0 | 108.6 (**1.41×**) | **129.0 (1.67×)** | 40.0 | 79.1 (**1.98×**) |
| H1 probe, decode | 78.3 | 101.2 (1.29×) | 115.3 (**1.47×**) | 44.6 | 80.9 (**1.81×**) |
| 4.3K doc, prefill tok/s | 1,113 | 1,091 (0.98×) | 1,045 | 546 | 804 (**1.47×**) |

- H1 (trailing zeros of n!) answered correctly in **every arm, every round** (17/17 runs).
- Draft acceptance bands unchanged by the split (XXS long 0.75-0.81, UD long 0.63-0.75).
- UD gains more than XXS: its 4-bit experts are the largest per-token host traffic, so the
  single-card run was the most host-bound. On this DDR3L box the UD tier roughly doubles.

## Calibration on a split
- XXS: `--pcie-frac` sweep is **flat at 92.3-92.4 tok/s from 0 to 1** with 100% cache
  residency — the single-card era's most sensitive knob stops mattering; calibrate wrote
  `--pcie-frac 0.00 --spec-min-p 0.70` (97.9 tok/s on its probe).
- UD: "the default settings are already the fastest here" (67.6 tok/s probe); workers
  9/17/18/23/35 = 68.3/68.3/68.5/66.7/69.6 — **the single-card UD tier's 20% pool-workers
  sensitivity (18 vs 35) disappears** once both caches are full.
- The engine still caps the expert-arena `cudaHostRegister` pin at 8 GiB on any multi-GPU
  Windows start (TCC included); `STRATA_ARENA_PIN_GIB` was not tried.

## Thermals
Peak 62/60 °C, max 202 W per card at the 250 W default limit, stock fan curves on the
front-intake open rig; no throttling, both cards within 2 °C of each other at all times.

## In daily use (engine 0.1.45, 2026-10-19)

Numbers from ordinary use of this box rather than the suite, on UD-Q6_K_XL and UD-IQ4_XS, both at
262,144 context:

- **A long answer**: 1,665 tokens at **44.1 tok/s** on Q6 with the expert cache at 96.5% hit; a
  1,500-token answer on the same tier 38.4-45.5 tok/s by run (draft acceptance 57-62%).
- **The CPU is not the wall**: during that decode both sockets together burned ~39.5 cores' worth
  of time, i.e. **0.9 core-seconds per output token**. Forcing `--pool-workers 9` instead of the
  default 35 cost only **13%** of decode while CPU use fell from ~39.5 to ~9 cores — the rest is
  spin waiting, not work. On a dual-socket host this is the cheapest way to buy power and fan
  noise back.
- **Three conversations at once** (`parallel: 3`): **67.1 tok/s aggregate = 1.41x** one stream,
  23 tok/s each, reproducible (63.9 on an earlier day). `--batch-mtp --batch-groups 1` does run the
  slot drafters — the engine logs `MTP proposals accepted 933 of 1463 (63.8%)` — but aggregates
  **63.6 tok/s**, 5% below the pipelined default on this tier: the serial windows cost more than
  the drafts return.
- **Restart variance is real on a two-node Windows box**: same binary, same config, four restarts,
  3x800-token probes: **43.9 / 45.5 / 47.4 / 48.0 tok/s**. The two high runs had the expert arena
  on NUMA node 0 and the two low ones on node 1 (**+6.7% for node 0**); the engine pins its workers
  across both processor groups either way. Benchmarks on this class of machine should record which
  node the arena landed on, or average two restarts.
- **`--pipeline-windows 2` is a loss on this tier**: **-5%** with the draft layer on the last card
  and -5% again with `STRATA_MTP_STAGE0=1`, while the XXS tier in the paired table above gained
  (129.0 vs 108.6). Tier-dependent, and opt-in on NVIDIA.

## Not tested
Batching on a split (#776), `STRATA_PREFILL_HELP`, `STRATA_SPLIT_MTP_BATCH`,
`STRATA_ARENA_PIN_GIB`, three cards. Long context (256K native / 512K / 1M) is covered in the
[2026-10-19 follow-up](../2026-10-19-community-2x-pcie-v100-longctx/README.md).

## Files
`results-flash-0144-*.json` — every arm's raw records (tags: short / long / L1-4k-doc /
H1-cap4000); `summarize-dual-0144.py` regenerates the table from them. Config diffs vs the
single-card configs are exactly `"gpu": [0, 1]` + `"layer_split": "auto"` (+ the pw2 flag in
its arm).
