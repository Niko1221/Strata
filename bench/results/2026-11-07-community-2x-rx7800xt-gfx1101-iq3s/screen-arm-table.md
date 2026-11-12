# Configuration screen on short prompts, 2026-11-07

One fresh engine start per arm, identical request bytes per arm, `temperature=0`, non-streaming, cold
arms verified `reused=0`. Decode mean is over four requests (4K x128, 32K x128, 64K cold x128, 64K
cached x512); the prefill column is the 64K cold request.

## Table 1 - warmed protocol, control = `base0s` (throwaway warm-up before measuring)

Single-shot arms drift by +-5% on this box, so only differences above ~8% are treated as real; the
two candidates that survived (parking discipline, WMMA) were re-tested by other instruments.

| arm | decode mean | vs control | 4K | 32K | 64K cold | 64K cached | 64K prefill | accept | QA | #267 | what it tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ccparkgain` | 63.95 | +3.6% | 65.2 | 68.1 | 61.9 | 60.6 | 1620.2 | 82.1 | - | 0 | parking: slots 2 + 2560 floor + min-tokens 4096 + park-gain 8192 |
| `cslots1` | 62.95 | +2.0% | 66.4 | 71.1 | 58.1 | 56.2 | 1612.9 | 83.0 | - | 0 | parking: slots 1 |
| `pinphys` | 62.60 | +1.4% | 64.4 | 70.2 | 55.7 | 60.1 | 1613.8 | 80.3 | - | 0 | CPU expert pool pinned to one SMT sibling per physical core, host loop on the last core |
| `pool6` | 61.85 | +0.2% | 62.8 | 66.2 | 59.5 | 58.9 | 1617.4 | 79.8 | - | 0 | --pool-workers 6 |
| `base0s` | 61.73 | +0.0% | 65.0 | 68.1 | 56.1 | 57.7 | 1621.0 | 78.3 | - | 0 | control on the promoted stock profile |
| `ccstock` | 61.70 | -0.0% | 65.0 | 68.7 | 54.8 | 58.3 | 1621.1 | 82.8 | - | 0 | vendor parking defaults written out (4 slots, 2560 floor, min-tokens 4096) |
| `mtpq4proj` | 61.50 | -0.4% | 65.1 | 65.4 | 56.7 | 58.8 | 1617.0 | 85.5 | 6/6 | 0 | --mtp-q4 (quarter-precision draft projections) |
| `auxauto` | 61.23 | -0.8% | 67.7 | 70.9 | 52.2 | 54.1 | 1569.9 | 79.9 | - | 0 | --aux-cpus auto (helpers on the host core's siblings) |
| `ccoff` | 61.17 | -0.9% | 62.3 | 71.7 | 52.0 | 58.7 | 1631.5 | 79.4 | - | 0 | --conversation-cache-mib 0 (parking off) |
| `arenapin24` | 60.65 | -1.7% | 64.4 | 70.1 | 53.3 | 54.8 | 1617.7 | 79.1 | - | 0 | --resident-budget-gib 44 (bigger pinned expert arena) |
| `split26` | 60.38 | -2.2% | 60.7 | 66.3 | 54.8 | 59.7 | 1537.9 | 81.3 | - | 0 | --layer-split 26 (26/22 instead of 24/24) |
| `wmma` | 59.90 | -3.0% | 61.2 | 67.9 | 54.6 | 55.9 | 1916.7 | 82.8 | 6/6 | 0 | STRATA_HIP_WMMA=1 (gfx11 matrix cores, PR #4859) |
| `k8v4` | 59.90 | -3.0% | 61.7 | 63.2 | 58.2 | 56.5 | 1625.7 | 81.3 | 6/6 | 0 | --kv k8v4 (816 B/cell vs INT8's 1,056) - outside the quant constraint, for completeness |
| `ccstock2` | 59.55 | -3.5% | 62.9 | 61.2 | 55.5 | 58.6 | 1608.5 | 78.0 | - | 0 | vendor parking discipline with slots 2 |
| `pipe3` | 58.40 | -5.4% | 59.7 | 61.0 | 54.9 | 58.0 | 1608.9 | 81.4 | - | 0 | --pipeline-windows 3 (engine clamps back to 2 on a two-stage split) |
| `spec4` | 58.38 | -5.4% | 58.3 | 70.4 | 56.5 | 48.3 | 1595.7 | 69.3 | - | 0 | --spec 4 |
| `dauto` | 57.10 | -7.5% | 63.9 | 56.5 | 56.4 | 51.6 | 1601.6 | 77.7 | - | 0 | --draft auto (serve picks per request) |
| `df2` | 56.98 | -7.7% | 56.5 | 65.4 | 50.8 | 55.2 | 1612.8 | 84.5 | 6/6 | 0 | DFlash2 drafter pack (q4_0-bf16sel), stock-matched |
| `minp35` | 56.95 | -7.7% | 58.8 | 69.3 | 50.2 | 49.5 | 1618.1 | 73.7 | - | 0 | --spec-min-p 0.35 |
| `spec4minp35` | 55.85 | -9.5% | 61.0 | 54.7 | 54.5 | 53.2 | 1598.7 | 59.0 | - | 0 | --spec 4 with --spec-min-p 0.35 |
| `spec6` | 54.20 | -12.2% | 52.6 | 64.5 | 51.4 | 48.3 | 1606.4 | 64.2 | - | 0 | --spec 6 |

## Table 2 - first protocol (no warm-up; kept because it exposes the cold-page-cache artefact)

The first request after an engine start pays a cold expert page-cache penalty, which is why the
protocol was changed: `concache3g` measured 22.1 t/s decode on its first 64K request while its 32K
request was the best of the run (72.1). The ranking was unusable, not the machine.

| arm | decode mean | vs control | 4K | 32K | 64K cold | 64K cached | 64K prefill | accept | QA | #267 | what it tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cslots2` | 63.58 | +5.0% | 67.0 | 71.3 | 54.2 | 61.8 | 1608.1 | 83.1 | - | 0 | --conversation-cache-slots 2 |
| `base0s` | 60.55 | +0.0% | 67.6 | 66.9 | 54.6 | 53.1 | 1595.1 | 82.7 | - | 0 | control on the promoted stock profile |
| `concache1g` | 56.25 | -7.1% | 59.6 | 66.3 | 47.8 | 51.3 | 1392.0 | 79.9 | - | 0 | --conversation-cache-mib 1024 |
| `wmma` | 55.75 | -7.9% | 58.5 | 65.4 | 42.3 | 56.8 | 1896.5 | 77.3 | 6/6 | 0 | STRATA_HIP_WMMA=1 (gfx11 matrix cores, PR #4859) |
| `concache3g` | 54.73 | -9.6% | 66.5 | 72.1 | 22.1 | 58.2 | 1519.5 | 81.8 | - | 0 | --conversation-cache-mib 3072 |

## Arms not measured, with the reason

| lever | why it was not measured |
| --- | --- |
| `STRATA_SCORES_THREAD=1` | never engages here: the HIP gate at `src/kernels/cuda/qsa_select.cu:1224-1236` accepts `gfx12` or `gfx1100` by name, `gfx1101` resolves to 0, and `grep -c "qsa exact thread"` over the 86 MB engine log is 0 |
| `--batch` / `--slots` 2/4/8 (concurrency) | measured earlier the same day: 8 slots with `--kv-resident` streaming thrashed (solo probe 174 s to first token, `loaded=false`); two engines, one per card, decoded 41-43 t/s per stream against 55-65 for one engine across both cards |
| `STRATA_EXCHANGE_ROTATE` | its own engine string says it moves experts between slots and the split needs fixed slots |
| `STRATA_SPLIT_OVERLAP` | the engine prints `adaptive overlap: off: not on a layer split` |
| `--prefill-pipe R` | measured by the vendor calibration: 826 tok/s off, 830 evened-out, 793 at R=768/1536/3072/6144 -> no gain here |
| Avalon expert-cache architecture | evaluated 2026-11-05 on this same machine: 0.59x decode, 0.26x prefill against Strata; artifacts removed |
