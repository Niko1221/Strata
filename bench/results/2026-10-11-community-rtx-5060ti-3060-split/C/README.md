# Community benchmark C: `STRATA_ADAPT_OVERLAP=0` on a 2-GPU layer split (IQ3_S)

Measured on 2026-10-11 by @mmbeanz. Targets the live upstream debate:
**#2010** (Linux CUDA, -21% decode), **#1975** (RTX 5090, IQ3_S -26%), **#1964** (4070 Ti,
IQ3_XXS) and **PR #2011** (make the 0.1.43 default opt-in everywhere). Every regression
reported so far is on a **single GPU**; this run measures the same flag on a **two-card
layer split with `--pcie-frac 0.00`**.

**Headline.** On this topology the flag is **not a lever we can see**: the prompt path is
null to within run-to-run spread (**+0.14%** at 32K), and decode shows no significant
difference (permutation p = 0.39), which **rules out the -21%/-26% class of regressions
here** but cannot resolve effects smaller than roughly ±5-11% given per-request decode
variance. So the reported regressions do not generalize to unequal-card layer splits with
expert streaming disabled — the cost/benefit of `ADAPT_OVERLAP` is topology-dependent.

## Setup: identical to run A except one environment variable

Same daily-driver config as `../A/` (`--layer-split 40`, `--pcie-frac 0.00`, pinned
`data/expert-profile.bin`, no `--expert-profile-save`), same evening, cold boot, 0 requests
before the suite. Only difference: `STRATA_ADAPT_OVERLAP=0` (0.1.43's Linux CUDA default is
**on**). Engine facts confirm the layout is unchanged: `expert_slots 8842`,
`expert_slots_primary 4984`, `pcie_frac 0.00`, `pool_workers 11`, `vram_free_mib 324`.

**Provenance caveat, stated plainly:** 0.1.43 does **not** log the overlap state — PR #2011
adds logging "in both directions", which is exactly what would make this auditable. The env
var here is operator-asserted, not verifiable from the artifacts. Recommendation for future
runs (and worth saying in the thread): capture `env | grep STRATA_` at server start into the
results folder.

Hardware/software/model/method are as hand-written in `../A/README.md` (7900X,
64 GB DDR5-6400, 5060 Ti PCIe 5.0 x8 @ 28.2-28.7 GB/s H2D + 3060 PCIe 3.0 x4 @ 3.15-3.2
GB/s, nspawn on bare metal, Strata 0.1.43 from `7695859a` + `setup.py`-only local commit,
IQ3_S GSQ-RCO, 512K yarn 2.0). Bench driven remotely with
`strata_bench.py --url <server> --long --runs 3`; `GET /metrics?requests=all` before/after
(`C-before.json`, `C-after-full.json`). Suite ~205 s.

## Results — A (overlap ON, default) vs C (overlap OFF)

| metric | A | C | delta |
| --- | ---: | ---: | ---: |
| prompt 32K tok/s (median of 3) | 1,720.8 (range 1720.5-1721.7) | 1,723.2 (1723.0-1723.7) | **+0.14%** |
| prompt 16K tok/s | 1,412.7 | 1,413.3 | +0.04% |
| prompt 4K tok/s | 868.1 | 867.8 | -0.03% |
| TTFT 32K (s) | 18.66 | 18.64 | -0.1% |
| short-chat decode (median, 9 runs) | 74.4 | 75.3 | +1.2% |
| agent decode (median, 15 turns) | 80.7 | 83.6 | +3.6% |
| expert hit_rate (median, 40 req) | 0.859 | 0.863 | +0.004 |
| draft acceptance | 68.9% | 71.8% | +2.9 pts |
| `pcie_share` / `ram_blobs` / `file_blobs` | 0 / 0 / 0 | 0 / 0 / 0 | identical |

## Statistics (per-request values from `/metrics?requests=all`, n = 40 per run)

Two-sample permutation tests, 6,000 shuffles, two-sided:

| quantity | A | C | mean diff | p |
| --- | ---: | ---: | ---: | ---: |
| hit_rate | 0.8417 | 0.8394 | +0.0023 | **0.92** |
| decode tok/s (all requests) | 71.10 | 73.61 | -2.52 | **0.39** |
| decode at 32K prompt | 53.2 (med) | 58.3 (med) | -7.07 | 0.20 |
| decode at 16K prompt | 58.3 (med) | 56.4 (med) | +2.27 | 0.29 |
| decode at 4K prompt | 70.7 (med) | 71.6 (med) | -0.63 | 0.50 |
| prompt_ms at 32K | 18,624 ms | 18,599 ms | +25 ms | 0.10 |

Nothing is significant. With 95% confidence intervals rather than p-values, the honest
resolution of this suite is:

- **Prompt path: precisely null.** 32K prompt throughput differs by **+0.14%**, inside the
  run-to-run spread of a single config (A's own 3 runs span 1720.5-1721.7).
- **Decode: imprecise.** mean difference -3.4% with 95% CI **[-11.2%, +4.4%]**. That is
  enough to **rule out the -21% (#2010) and -26% (#1975) regressions on this topology**,
  but not enough to claim the flag is exactly neutral.
- **Hit rate: null.** +0.0023, 95% CI [-0.039, +0.044].

Per-request decode varies enormously within a run (48-104 tok/s depending on context and
cache state), which is what limits the decode bound; a soak or a larger `--runs` would
tighten it.

## Why a split might be immune (hypothesis, not measurement)

Two structural differences from the single-GPU reporters:

1. **Little to hide.** With `--pcie-frac 0.00` no expert is streamed on the miss path
   (telemetry: `pcie_share = 0` on all 40 requests in both runs). The only large copies left
   are the adaptive tier's own swaps: at the 0.1.43 defaults (96 swaps / 4 tokens, ~2 MB
   blobs) that is on the order of ~48 MB per verify window — ~1.7 ms at 28 GB/s against a
   ~43 ms decode window, i.e. ~4%.
2. **Already overlapped.** In a two-stage split, stage 0 works on the next window while
   stage 1 finishes the current one, so some of the hiding that `ADAPT_OVERLAP` provides on
   a single card is already happening structurally.

Both are consistent with "no measurable effect either way here", but neither is proven by
this run.

## What to post upstream

A comment on **#2010** (per the contributing rule "add to a thread that already covers your
point"): *ADAPT_OVERLAP=0 vs default on a 5060 Ti + 3060 layer split, IQ3_S,
`--pcie-frac 0.00`: prompt path null to +0.14%, decode difference not significant
(p=0.39, 95% CI -11%..+4%), hit rate identical — the single-GPU regressions do not
reproduce on this topology.* Plus the small ask: 0.1.43 does not log the flag state, which
makes A/B runs unauditable; PR #2011's logging would fix that.

## Limitations

- Speed only; no correctness/needle check (needle suite: `../needles/`).
- Decode resolution ±~8% at 95% confidence (see CI above); prompt resolution ~0.2%.
- Env-var provenance not verifiable from artifacts (see caveat).
- One suite per config (3 runs), same evening as A; A is the same-day baseline.
- Permutation p-values treat requests as independent; the adaptive tier correlates
  consecutive requests, so they are optimistic.
- Not tested: overlap on a split **with** `--pcie-frac > 0` (the case where copies are large
  and the flag might matter), concurrency, vision, mmap tier, Windows/HIP.
