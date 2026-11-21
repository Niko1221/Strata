# RX 7900 XTX (gfx1100), unsloth UD-IQ3_XXS: production-config tiers and a prod-env validation sweep (engine 0.1.63)

AMD Radeon RX 7900 XTX 24 GB (gfx1100, PCIe 4.0 x16), Ryzen 9 7900X, 96 GB RAM, Fedora Linux 45, ROCm 7.17.26392
nightly SDK (HIP). Linux, source build.

**Engine 0.1.63 = upstream v0.2.17 (tag `12d491da4`) + 11 locally replayed fork commits** (the fork's
kv-ring line [PR #3919], a route-prior redundant-top-k skip, a q8_1 dedup for the shared-expert stream, and two
diag switches; every commit is disclosed in the fork's refresh-0163 branch). Disclosed because the fork commits
touch the decode path: the tables below are this composed build, not a stock binary. The binary's `--version`
reports `strata 0.1.63`; the build is Release, HIP gfx1100, vendored SDK.

Model: Qwen3.8-Flash-Next **UD-IQ3_XXS** (unsloth dynamic quant), 83.6 GB across 4 shards (shard 4 is the PLE
engram), MTP drafter sidecar, quant-independent expert profile. Production-shaped one-shot runs, greedy, 256
generated tokens: `--spec 5 --spec-min-p 0.70 --lookup-chain 2 --kv int8 --kv-resident 49152 --prefill auto
--expert-cache auto --vram-reserve-mib 3072` plus the route/cache env set (route prior 0.5, route tail-skip 7,
GDN chunked, PF gemm/pad/fused, prefill ring, PA fast, KV ring adapt — the fork production set; each knob is
registered upstream or carried in the fork's own settings register). Real-text workloads (coding and prose
tracks, 32,768 and 131,072 tokens) and pretokenized tier files (1K/64K/262K); 262K uses a 261,888-token file
(the trained context is 262,144, so prompt + generation must fit it).

Every cell is one engine process under a 66 GB memory cap with the desktop's page cache drained; arms are
interleaved within each tier so monotone drift cancels; an identical-config control arm calibrates each window's
noise threshold (max(2%, 2xSD of the control's paired log-ratios)). Full per-cell records:
[`cells.json`](cells.json).

## Prompt (tokens/s)

| Context | 1K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: |
| **UD-IQ3_XXS** | 869 | 3,201 | 3,124 | 3,344 | 3,133 |

Medians of the production-config arm (n=2/4/1/6/1). 64K and 262K are single cells.

## Output (tokens/s)

| Context | 1K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: |
| **UD-IQ3_XXS** | 45.6 | 47.1 | 48.3 | 56.7 | 43.9 |

- Output speed at 32K moved 42.4-51.5 across four same-config cells (n=4) and at 128K 54.9-59.2 across six:
  **a single deep-tier cell on this card varies +-14-18% with the text and the trajectory**, so paired
  same-window comparisons are the only kind these numbers support. The 128K tier is the most stable (six cells).

## Production-env validation sweep (the interesting part)

The production config (spec width 5, draft min-p 0.70, lookup-chain 2) was tuned on an earlier IQ3_S pack of the
same model. This sweep re-validated each knob on the UD-IQ3_XXS pack, one variable at a time, 150 paired cells,
two sweeps, control-calibrated:

| knob | tried | verdict on UD-IQ3_XXS |
| --- | --- | --- |
| spec width | 3 / 4 / 6 vs 5 | **5 stands.** Width 4/6 showed +10-13% at 128K in the first sweep (4/4 pairs), but the higher-power re-measure (per-track, 2 reps, 262K included) falsified it: on the coding track both widths are SLOWER (width 4: 46.7/47.6 vs 55.9/55.4 tok/s), the pooled median was carried by the prose track, and both widths went negative at 262K (2/2 pairs each). The first sweep's "lead" is a worked example of a pooled track median manufacturing a win. |
| draft min-p | 0.65 / 0.75 vs 0.70 | **0.70 stands.** 0.65 loses 5.2% at 1K (draft acceptance visibly drops, 0.79 -> 0.63 on the first cell) and 3.9% at 128K; 0.75 is flat everywhere. |
| lookup-chain | 0 / 1 vs 2 | **2 stands.** Off is bimodal and track-dependent (+16%/-7% pairs at 128K); 1 is flat. |

- Net: nothing changed in the config; the sweep is a negative result that pins three defaults on this
  quant/card combination with measurement power, plus the single-cell variance caveat above.
- No default was moved, so this is not a tuning claim — it is a validation that the shipped draft-policy
  defaults hold on unsloth's UD-IQ3_XXS quant at 32K-262K on gfx1100.

## What this report is not

- Not a stock-binary result (see the fork-commit disclosure above).
- Not concurrent-serve measurements (one-shot cells).
- Not a quality evaluation (greedy, 256 tokens, no battery).
