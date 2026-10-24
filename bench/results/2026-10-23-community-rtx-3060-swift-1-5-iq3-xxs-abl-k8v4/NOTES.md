# Measured trade-off on no-AVX2 (2x RTX 3060): `--kv k8v4` vs `--kv q4_0`

Same rig/method as PR #3863. Engine 0.1.47, Swift-IQ3_XXS-abliterated, same prod configuration,
only `--kv q4_0` swapped for `--kv k8v4` (INT8 keys + rotated Q4_0 values). Baseline: the
2026-10-22 abliterated IQ3_XXS run (q4_0).

| Metric | q4_0 | k8v4 | Delta |
| --- | ---: | ---: | ---: |
| short chat decode tok/s | 44.9 | 48.1 | **+7.1 %** |
| agent decode tok/s | 47.0 | 50.7 | **+7.9 %** |
| 4K prompt tok/s | 522.9 | 466.0 | **-10.9 %** |
| 16K prompt tok/s | 781.9 | 730.5 | -6.6 % |
| chat/4K/16K TTFT (s) | - | - | +7..+12 % |
| agent cache hit % | 69.5 | 69.5 | ~0 |

k8v4 frees KV VRAM (smaller keys) and speeds decode, but slows fresh-prompt reading and raises
TTFT - the same decode-vs-read pattern as the 0.1.48.1 engine (issue #3866). For a workload that
re-sends large documents (50-95k-token prompts) the reading cost outweighs the decode gain, so we
keep `q4_0` in production. Recorded as a data point; adopt only if decode-over-read is desired.
Per-run data in results.json.