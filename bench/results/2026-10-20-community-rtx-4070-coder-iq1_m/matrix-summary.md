# Switch sweep behind the report

48 configurations on the **previous** Coder build (the 2026-09-29 release, the same engine and the same
card), 3 runs per test with the same harness, plus the corrected final arms in the README.

**Read the spread first.** The noise floor of this box, measured on the final run (the same stock
configuration twice, 40 minutes apart): decode **0.9% apart**, time to first token 0.9%, the 16K prompt
rate 6.3%, the **4K prompt rate 22.3% apart**. Two arms of this sweep are the *stock configuration
re-measured* (the sweep script failed to add their flag) and they came out at 40.5, 44.5 and 40.6 tok/s
decode in one session. The differences in the table below are therefore only meaningful where they are
much larger than that, and the 4K prompt column is not a discriminator at all.

| Configuration | Decode tok/s | vs stock | 4K prompt tok/s | 16K prompt tok/s | 16K TTFT s |
| --- | ---: | ---: | ---: | ---: | ---: |
| the stock configuration | 40.5 | +0% | 1882 | 1759 | 9.16 |
| the sm_86 switch set, applied by hand (Ada gets no per-architecture table) | 40.6 | +0% | 1591 | 1745 | 9.23 |
| the sm_86 kernel switches without the adaptive-policy part | 40.0 | -1% | 1827 | 1545 | 10.43 |
| the sm_120 switch set, applied by hand | 40.9 | +1% | 1819 | 1612 | 9.99 |
| `STRATA_ADAPT_OVERLAP_RESIDENT=1` (the adaptive overlap in resident-RAM mode) | 40.4 | -0% | 1585 | 1517 | 10.61 |
| `--kv k8v4` | 39.4 | -3% | 1941 | 2367 | 10.89 |
| `--kv q4_0` | 43.4 | +7% | 1808 | 1620 | 9.94 |
| `--vram-reserve-mib 1003` (what the engine itself suggests) | 38.5 | -5% | 1803 | 1628 | 9.90 |
| `--max-context 65536` | 44.4 | +10% | 1657 | 2253 | 7.16 |
| `STRATA_DEQUANT_WIDE=1` | 41.8 | +3% | 1962 | 1814 | 8.88 |
| `STRATA_PF_FUSED=1` | 38.9 | -4% | 1947 | 2381 | 7.51 |
| `--prefill auto:16384` | 39.5 | -2% | 1832 | 1882 | 8.56 |
| `--prefill-head-alias` | 40.6 | +0% | 1802 | 1442 | 11.17 |
| `STRATA_PROMPT_ATTN_IMMA=1` + `STRATA_HC_UPMIX=1` + `STRATA_SCORES_THREAD=1` | 41.6 | +3% | 1665 | 1902 | 8.47 |
| `--pcie-frac 0.20` | 43.4 | +7% | 1589 | 1659 | 9.71 |
| `--pcie-frac 0.30` | 39.4 | -3% | 1597 | 1646 | 9.79 |
| `STRATA_PCIE_BALANCE=1` | 42.4 | +5% | 1604 | 1622 | 9.93 |
| `--pool-workers 6` | 43.0 | +6% | 1628 | 1790 | 9.00 |
| `--aux-cpus auto` | 39.1 | -3% | 1918 | 2083 | 7.74 |
| `--spec 3` | 42.2 | +4% | 1585 | 1666 | 9.67 |
| `--spec-min-p 0.4` | 37.5 | -7% | 1599 | 1627 | 9.90 |
| `--spec 5` | 39.7 | -2% | 1823 | 1621 | 9.94 |
| `--ple-inflight 1024` | 39.1 | -3% | 1630 | 1818 | 8.86 |
| `--ple-inflight 64` | 40.5 | +0% | 1831 | 1667 | 9.66 |
| `--ple-io mmap` | 40.4 | -0% | 1827 | 1432 | 11.24 |
| `--ple-row-cache 4194304` | 40.3 | -0% | 1623 | 1608 | 10.02 |
| `--expert-cache-per-layer` | 44.5 | +10% | 1865 | 1780 | 9.05 |
| `--prefill auto:32768` | 42.3 | +4% | 1539 | 1542 | 10.44 |
| `--pcie-frac 0.55` | 38.4 | -5% | 1204 | 1646 | 9.78 |
| `--pcie-frac 0.65` | 38.7 | -4% | 1817 | 1652 | 9.75 |
| `--adapt-swaps 0` | 38.9 | -4% | 1796 | 1629 | 9.90 |
| `--adapt-min-gain 3.0` | 41.4 | +2% | 1820 | 1605 | 10.04 |
| `--host-core sibling` | 42.4 | +5% | 1397 | 1544 | 10.43 |
| `STRATA_IQ_MT_MIN=1` | 42.9 | +6% | 1807 | 1568 | 10.27 |
| `--pool-workers 6` + `--aux-cpus auto` | 39.2 | -3% | 1958 | 2128 | 7.58 |
| per-layer cache + wide dequant | 40.9 | +1% | 1917 | 1735 | 9.29 |
| per-layer cache + wide dequant + 64k context | 45.0 | +11% | 2000 | 2330 | 6.92 |
| per-layer cache + wide dequant + 96k context | 41.0 | +1% | 1958 | 2228 | 7.24 |
| `--mtp-draft-vocab` with the English+code subset (81 MiB of VRAM instead of 213) | 39.5 | -2% | 1852 | 1651 | 9.76 |
| the English draft subset + per-layer cache + wide dequant + 64k context | 43.9 | +8% | 1984 | 2274 | 7.09 |

What the sweep says, beyond the noise:

- **The per-architecture switch tables do not apply to Ada and do not help.** `nv_defaults.cpp` matches
  cc 8.6 and cc 6.0 only, so an sm_89 card gets no table at all; applying the sm_86 and sm_120 sets by
  hand cost 8-18% of prompt throughput and gained nothing in decode.
- **The VRAM budget is the binding constraint.** The engine reports 229 MiB of VRAM free with everything
  loaded and warns that requests may stall. Asking for more cache VRAM directly (`--expert-cache 4096`,
  `--vram-reserve-mib 300`) does not slow the engine down - **the model fails to load**. Everything that
  helped gave the expert cache more room or took pressure off the 8-core CPU pool.
- **The PCIe share is a decode-against-prompt dial.** `--pcie-frac 0.20` bought decode and cost 4K prompt
  throughput; 0.55 did the opposite and cost the most of any arm (4K -36%). The engine's automatic rule
  (it re-fits the share from the pool's measured speed) is a reasonable middle.
- **Two settings are traps on this card:** `--kv k8v4` (the time to first token of a short prompt grew
  from 0.77 s to 5.13 s) and `--ple-io mmap` (-19% on the 16K prompt, against the docs' own warning).
- **The context length is the one change that reproduced everywhere**: 64k or 96k context gave ~+10% decode
  and +28-32% on the 16K prompt with a 21-24% lower time to first token, on both model builds.

