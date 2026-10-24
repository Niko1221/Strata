# Measured and rejected on no-AVX2 (2x RTX 3060): `--pool-workers 11`

Same rig/method as PR #3863 (2x RTX 3060 sm_86 layer split, Xeon E5-2697 v2 no AVX2, 128 GB
DDR3-1866 quad-channel, Windows 11, source build `STRATA_ISA_FLOOR=avx`). `tools/strata_bench.py
--url` against a 127.0.0.1-bound server, one engine (0.1.47), one model (Swift-IQ3_XXS-abliterated),
same prod configuration, the only change being `--pool-workers 11` on top of the default (0 = all
physical cores). Comparison baseline: the 2026-10-22 abliterated IQ3_XXS run.

| Metric | default (all cores) | --pool-workers 11 | Delta |
| --- | ---: | ---: | ---: |
| short chat decode tok/s | 44.9 | 39.2 | -12.7 % |
| short chat TTFT (s) | 0.71 | 1.03 | +45.9 % |
| 4K prompt tok/s | 522.9 | 505.2 | -3.4 % |
| 16K prompt tok/s | 781.9 | 790.3 | +1.1 % |
| agent decode tok/s | 47.0 | 44.5 | -5.3 % |

Verdict: **keep the default** on this 2-socket machine. Strata's own setup hints at
`--pool-workers 11` for 2-socket Xeons, but that does not help here - decode is measurably worse.
Plausibly decode is not CPU-expert-worker-bound on this box (GPU expert cache + DDR3 bandwidth
roles), so pinning fewer workers only adds nothing. Recorded so the hint isn't re-applied
blindly; per-run data in results.json.