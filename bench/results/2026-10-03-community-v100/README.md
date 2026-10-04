# 2026-10-03: Tesla V100-PCIE-32GB, dual Xeon E5-2696 v3, 128 GB RAM (Windows)

Field measurements from a daily-use install and a capability battery - not a fixed-cap
benchmark harness. Each number is a single run unless stated; the `--calibrate` sweep
is the engine's own repeated measurement. Reported per the guide in
[COMMUNITY_BENCHMARKS.md](../../../docs/COMMUNITY_BENCHMARKS.md).

## Hardware

- Tesla V100-PCIE-32GB (sm_70, TCC mode), solo, PCIe **Gen3** x16
- dual Xeon E5-2696 v3 (36C/72T, two sockets, NUMA)
- 128 GB DDR4
- Windows 10 Enterprise LTSC (19044)

## Software

- Strata **v0.1.38**, engine built **from source** for sm_70 (`STRATA_EXPERIMENTAL_SM60=1`),
  CUDA Toolkit 12.4, MSVC 14.44; driver 581.80
- Build note: linking needed `-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared` on MSVC, see #585

## Model

- Qwen3.8-Flash-Next **IQ3_XXS** (ISTA-DASLab GSQ-RCO, pinned revision ed59f920, verified by setup)
- Context **131,072**; KV int8 streamed to RAM (`--kv-resident 32768`); MTP draft (q2_0)
- 15,553 / 24,576 experts (63%) cached in VRAM (25.2 GiB); expert cache hit 95-97% measured

## Settings

Before / after `--calibrate` (kept: `--pcie-frac 0.55 --spec-min-p 0.70 --pool-workers 18`):

| Setting | Sweep (tok/s) |
|---|---|
| pcie-frac | 0.00: 17.8 / 0.20: 16.2 / 0.34: 20.0 / 0.35: 19.2 / **0.55: 25.4** / 0.75: 21.5 |
| spec-min-p | 0.30: 19.2 / 0.50: 18.3 / **0.70: 21.1** |
| pool-workers | 35: below 28.6 / 23: 28.6 / **18: 33.5** |

Takeaway: on this dual-socket NUMA box 18 CPU workers beat 35 by 20%+; defaults tuned on a
6-core Ryzen were ~2x off. **Calibration doubled decode: 16.5 -> 33.5 tok/s.**

## Decode speed (real requests, thinking on, temperature 0.6)

| Workload | Output | tok/s (pre-calibration) |
|---|---|---|
| First request after engine start (JIT warm-up) | 175 | 7.5 |
| Short answer | 39 | 10.1 |
| Medium answer | 289 | 13.7 |
| ~400-word essay (thinking + prose) | 322 | 14.2 |
| Long-form writing | 1,492 | 15.8 |
| Long multi-method math reasoning | 1,761 | 16.8 |
| 31k-token interactive session (web UI) | 31,796 | 16.7 steady |

After calibration, a 196-token prose request measured 22.0 tok/s (same class of workload
as the 39/289-token rows above, which were 10.1-13.7 before). Speed is flat through a
31k-token session (KV streaming).

## Prompt processing (scaling with document size)

| Prompt | tok/s |
|---|---|
| ~90 tokens | 57 |
| ~280 tokens | 133 |
| 4,340 tokens (synthetic log document) | **863** (5.03 s) |

For comparison, Qwen3.8-27B (UD-Q4_K_M, llama.cpp b10545, thinking build, same box) read
the same 4,340-token document at 629.5 tok/s.

## MTP

- Acceptance 97.8% on repetitive output, ~65% on free prose (measured over the battery above)

## Notes

- Decode is memory/PCIe-bound on this card: GPU util ~19%, 42-48 W of the 250 W limit, 49 C.
- Full capability comparison against Qwen3.8-27B (llama.cpp) on the same machine, and the
  thinking-budget pitfall (`reasoning_budget_tokens`), are described in issue #617.
