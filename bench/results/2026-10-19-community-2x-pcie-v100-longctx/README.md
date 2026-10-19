# Long context on a dual PCIe V100 layer split: 256K native, YaRN x2 at 512K, YaRN x4 at 1M (0.1.45)

Follow-up to [2026-10-18-community-2x-pcie-v100-ddr3l](../2026-10-18-community-2x-pcie-v100-ddr3l/README.md),
which covered the split itself on engine 0.1.44.1. This folder is the same machine on **0.1.45**, asking a
different question: how far the context can go on the split, and what the YaRN extrapolation costs.

## Machine and engine

- 2x Tesla PG500-216 32 GB (V100-class, sm_70, TCC), PCIe 3.0 x16 each (`nvidia-smi` reports Gen3 x16,
  current = max); the display card is a GTX 1060 on the other CPU's slot and holds no model data.
- 2x Xeon E5-2696 v3 (18C/36T each, 72 logical, two processor groups of 36), 128 GB DDR3L-1600 (8 channels,
  2 NUMA nodes), Windows 10.
- Engine **0.1.45**, official prebuilt `strata-windows-x64-cuda12`.
- `--kv int8 --kv-resident 32768`, `--spec 4`, `--mtp .../mtp/rt`, `--expert-cache auto`, `--prefill auto`,
  `--vram-reserve-mib 700`; vision encoder loaded; `layer_split: auto`.

Context configurations measured (one restart each):

| Arm | engine args |
| --- | --- |
| native 256K | `--max-context 262144` |
| 512K | `--max-context 524288 --rope-scaling yarn --rope-scale 2` |
| 1M | `--max-context 1048576 --rope-scaling yarn --rope-scale 4` |

The engine logs the scaling it applied, e.g. `rope scaling yarn, factor 2 (freq_scale 0.5, base 1e+07,
mscale 1.069315), --max-context 524288 against a trained context of 262144`.

Quantizations: **UD-Q6_K_XL** (6-bit, 102 GiB of experts) and **UD-IQ4_XS** (4-bit, 55 GiB of experts),
Unsloth's packs, `--resident-budget-gib` 90 and 55 respectively.

## Method

- `bench-ctxdecay256.py <tag> <ctx,...>`: builds a document from 30 repeated knowledge paragraphs, asks for
  a 400-token structured summary, and reports the engine's own `prompt_per_second` / `predicted_per_second`
  and draft acceptance. Ladder targets are actual token counts: the build estimate is scaled by the measured
  0.937 actual/estimate ratio, and a rejected (over-limit) point is retried one notch lower.
- `h1-deep.py <doc_tokens> <max_tokens> <tag>`: the same corpus, then "ignore all the material above" and the
  H1 question (smallest n whose factorial ends in exactly 2019 zeros, answer 8090). The answer is checked for
  `8090`; `max_tokens` keeps prompt + output inside the served context.
- One request per ladder point per arm, greedy-ish sampling (`temperature 0.6`, `top_p 0.95`, `seed 42`),
  `reasoning_budget_tokens 400`. No retries except the over-limit fallback.

## Results: decode and prefill (tok/s)

### UD-Q6_K_XL (6-bit)

| served ctx | input tokens | prefill | decode | accept |
| --- | ---: | ---: | ---: | ---: |
| native 262144 | 16,013 | 1,239 | 33.4 | 0.646 |
| | 64,061 | 1,598 | 42.3 | 0.620 |
| | 129,119 | 1,493 | 42.0 | 0.565 |
| | 194,132 | 1,329 | 41.4 | 0.709 |
| | 258,139 | 1,190 | **40.7** | 0.629 |
| 524288 (YaRN x2) | 64,061 | 1,522 | 30.3 | 0.601 |
| | 194,132 | 1,458 | 40.9 | 0.668 |
| | 303,932 | 1,182 | 41.8 | 0.639 |
| | 425,876 | 1,040 | 37.7 | 0.590 |
| | 517,320 | 887 | **36.6** | 0.707 |
| 1048576 (YaRN x4) | 194,132 | 1,498 | 36.0 | 0.591 |
| | 517,320 | 1,062 | 39.0 | 0.637 |
| | 1,021,605 | 707 | **34.5** | 0.590 |

### UD-IQ4_XS (4-bit)

| served ctx | input tokens | prefill | decode | accept |
| --- | ---: | ---: | ---: | ---: |
| native 262144 | 16,013 | 1,042 | 69.3 | 0.777 |
| | 64,061 | 2,272 | 65.9 | 0.708 |
| | 129,119 | 2,052 | 60.4 | 0.651 |
| | 194,132 | 1,792 | 59.4 | 0.678 |
| | 258,139 | 1,587 | **58.8** | 0.644 |
| 524288 (YaRN x2) | 64,061 | 1,831 | 47.0 | 0.594 |
| | 194,132 | 2,012 | 52.9 | 0.545 |
| | 303,932 | 1,591 | 53.6 | 0.598 |
| | 425,876 | 1,345 | 52.9 | 0.650 |
| | 517,320 | 1,147 | **49.2** | 0.577 |
| 1048576 (YaRN x4) | 194,132 | 2,065 | 53.1 | 0.632 |
| | 517,320 | 1,427 | 50.3 | 0.601 |
| | 711,865 | 1,012 | 45.7 | 0.597 |
| | 1,021,605 | 819 | **41.0** | 0.564 |

Both 1M arms ran without a crash or OOM; before the last UD fill the machine had 82.6 GB free.

## Results: H1 at depth

| Arm | input tokens | output | prefill | decode | accept | answer |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Q6, 245K | 248,032 | 2,420 | 311 | 40.7 | 0.789 | correct |
| Q6, 507K | 507,206 | 4,511 | 628 | 39.8 | 0.783 | correct |
| Q6, 1.01M | 1,011,336 | 3,642 | 362 | 39.1 | 0.821 | correct |
| IQ4_XS, 245K | 248,032 | 2,782 | 563 | 71.2 | 0.808 | correct |
| IQ4_XS, 507K | 507,206 | 4,404 | 752 | 68.2 | 0.829 | correct |
| IQ4_XS, 1.01M | 1,011,336 | 2,371 | 422 | 56.9 | 0.792 | correct |

All six answers are `n = 8090` with the Legendre/v5 argument; recall does not degrade with depth on either
quant in this range.

## Takeaways

1. **256K native is the sweet spot.** Q6 is flat across 64K-255K (42.3 -> 40.7, -3.8%). IQ4_XS decays
   69.3 -> 58.8 (-15%) but the loss is front-loaded: 16K-128K costs 13%, 128K-258K only 2% more.
2. **YaRN has a price inside the trained window.** At ~194K: Q6 41.4 native vs 40.9 (x2) vs 36.0 (x4);
   IQ4_XS 59.4 native vs 52.9 (x2) vs 53.1 (x4). The x4 config costs Q6 ~13% at a depth it could reach
   natively, so the 1M form is for prompts that actually need it.
3. **Beyond 256K the split still works.** At 517K tokens: Q6 36.6, IQ4_XS 49.2 tok/s. At 1.02M tokens
   (YaRN x4): Q6 34.5, IQ4_XS 41.0 tok/s. Against each quant's own 256K-native value (Q6 40.7 at 258K,
   IQ4_XS 58.8) that is -10% / -16% at 517K and -15% / -30% at 1M — the price of the extrapolation, not of
   the split.
4. **IQ4_XS is 1.2-1.6x faster than Q6 at every depth measured** (258K: 58.8 vs 40.7; 1M: 41.0 vs 34.5),
   and both pass the H1 probe at 1M. Q6's advantage here is flatness, not speed.
5. **Prefill is best at long prompts**: 1,042 (16K) -> 2,272 (64K) -> 1,587 (258K) -> 819 (1.02M) on IQ4_XS.
   The short-prompt points are dominated by fixed per-request overhead, not bandwidth.

## Limitations

- **One run per point.** Decode moves with draft acceptance (reported per row); a different prompt moves the
  numbers by several percent. The first ladder point after a fresh start is warm-up contaminated on both
  quantizations (the engine swaps the adaptive expert tier in during the first request): the 16K points and
  the 64K point of the 512K form are low for that reason.
- **The H1 probes reuse the ladder's KV prefix** (wall times of 43-86 s for 250K-1M prompts show this), so
  their prefill column is not a cold-prefill measurement. The answer still has to attend over the whole
  document, which is what the probe checks.
- No needle test and no multi-needle test; no quality suite beyond the engine's own checks and the H1 probe.
- Single machine, one engine version, one day. The 0.1.44.1 split report in the parent folder is the
  same-machine single-card control for the split itself; this folder has no single-card arms.
- The 1M arms ran on a machine whose free RAM was 82.6 GB before the fill; a machine with less RAM may spill
  the KV to the page file at 1M.

## Files

- `results-ctxdecay256-{q6-256k,q6-512k,q6-1m,ud-256k,ud-512k,ud-1m}.json` — the ladders.
- `results-h1-{q6-256k,q6-512k,q6-1m,ud-245k,ud-500k,ud-990k}.json` — the H1 probes.
- `bench-ctxdecay256.py`, `h1-deep.py` — the two drivers, unmodified apart from the tag/path arguments.
