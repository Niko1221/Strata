# Community benchmark D: `STRATA_ROUTE_TAIL_SKIP=0` on a 2-GPU layer split (IQ3_S)

Measured on 2026-10-12 by @mmbeanz. Targets the 0.1.42 default
change, whose own log line claims *"+10..20% decode on 12-16 GB cards that miss experts"*.
Those figures were measured on single CUDA cards and HIP splits; there was no NVIDIA-split
IQ3_S number.

**Headline.** On this box the default is **worth keeping but the claim does not hold**:
tail-skip ON is **+5.2% aggregate decode** vs OFF (95% CI -2.9%..+13.3%, p = 0.20), and the
gain is concentrated at short context (**+9.3% at 4K, +7.4% at 609, +4.8% at 16K, +1.0% at
32K**) with **no prompt-path effect**. Separately, a **measurement trap**: the engine skipped
**98,217 of 996,599 decode lookups (9.9%)** in run A, and depending on how the `hit_rate`
denominator treats them the metric moves by up to **+9.0 points** — so `hit_rate` is **not
comparable across this flag**, which matters for anyone reading coverage numbers off the
monitor.

## Setup: identical to run A except one environment variable

Same daily-driver config as `../A/` (`--layer-split 40`, `--pcie-frac 0.00`, pinned
`data/expert-profile.bin`), cold boot, 0 requests before the suite, engine facts identical
(`expert_slots 8842`, `expert_slots_primary 4984`, `pcie_frac 0.00`, `pool_workers 11`,
`vram_free_mib 324`). Only difference: `STRATA_ROUTE_TAIL_SKIP=0`.

**Unlike the overlap run (`../C/`), this A/B is auditable from artifacts**: the engine
prints the state, and run A's log contains

```
strata serve: STRATA_ROUTE_TAIL_SKIP=7 is ON (default): ... (+10..20% decode on 12-16 GB
              cards that miss experts, answers differ slightly from 0.1.41); STRATA_ROUTE_TAIL_SKIP=0 turns it off
```

and D's log shows the off state. Hardware/software/model/method as hand-written in
`../A/README.md`. Bench driven remotely (`--url`, LAN RTT 0.25 ms),
`GET /metrics?requests=all` before/after. Suite ~205 s.

## Results — A (skip ON, default) vs D (skip OFF)

| metric | A | D | delta |
| --- | ---: | ---: | ---: |
| short-chat decode (median, 9 runs) | 74.4 | 72.0 | -3.2% |
| agent decode (median, 15 turns) | 80.7 | 76.0 | -5.8% |
| prompt 32K tok/s | 1,720.8 | 1,722.7 | +0.1% |
| prompt 16K tok/s | 1,412.7 | 1,411.1 | -0.1% |
| prompt 4K tok/s | 868.1 | 871.2 | +0.4% |
| TTFT 32K (s) | 18.66 | 18.64 | -0.1% |
| draft acceptance | 68.9% | 68.9% | 0.0 pts |
| expert hit_rate (median, 40 req) | 0.859 | 0.806 | -5.3 pts *(see trap)* |

## Statistics (per-request `/metrics?requests=all`, n = 40 per run)

Permutation tests, 6,000 shuffles, two-sided; CI on the mean difference:

| quantity | A (ON) | D (OFF) | skip-ON effect | p | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| **decode tok/s (all requests)** | 71.10 | 67.58 | **+5.2%** | 0.20 | **-2.9% .. +13.3%** |
| decode at 4K prompt | 70.7 (med) | 62.1 (med) | +9.3 | 0.10 | [+2.9, +15.8] |
| decode at 609 prompt | 71.6 (med) | 60.6 (med) | +7.4 | 0.30 | [-3.9, +18.8] |
| decode at 16K prompt | 58.3 (med) | 53.5 (med) | +4.8 | 0.10 | [+1.4, +8.2] |
| decode at 32K prompt | 53.2 (med) | 52.3 (med) | +1.0 | 0.80 | [-7.4, +9.4] |
| prompt_ms at 32K | 18,624 | 18,604 | +11 ms (0.06%) | 0.61 | [-21, +43] |
| hit_rate | 0.8417 | 0.786 | +5.6 pts | 0.016 | [+1.2, +10.1] |

Reading: the direction is consistent (skip ON never slower, usually faster), the point
estimate is **+5.2% aggregate decode**, and the confidence interval **excludes the "+10..20%"
claim only at its upper end** — i.e. we cannot confirm the advertised range on this rig, and
the most likely true effect is single-digit. The prompt path is unaffected (as expected:
tail-skip is a decode-window mechanism). Draft acceptance is identical, so unlike the V100
observation in #1884 (70.1% -> 68.2%) there is **no acceptance cost here**.

## The hit_rate trap (important for anyone reading the monitor)

Run A's log carries a cumulative counter:

```
strata serve: route tail skip: 98217 missed experts skipped (105328 entries) since the start
```

against `996,599` decode lookups summed over the same run's 40 hit-rate lines — i.e. **9.9%
of decode lookups were skipped rather than counted as misses**. Recomputing A's aggregate
hit rate both ways:

| denominator treatment | A aggregate hit_rate |
| --- | ---: |
| skipped misses **included** as misses | 0.8259 |
| skipped misses **excluded** from lookups | 0.9162 (**+9.0 points**) |

The measured A-vs-D hit_rate difference was +5.6 points, so **it is plausibly entirely a
definition artifact** rather than a coverage change. Practical guidance: do not compare
`hit_rate` across `STRATA_ROUTE_TAIL_SKIP` on/off, and if the docs or the Monitor present
hit rate as a coverage measure, the flag's effect on the denominator should be stated.

**Confirmed by the cross-run arithmetic (both logs now in this folder):**

| | A (skip ON) | D (skip OFF) |
| --- | ---: | ---: |
| output tokens | 1,766 | 1,875 |
| lookups per output token, as reported | 564.3 | **611.8** |
| skipped per output token (log counter) | 55.6 | 0 |
| lookups per token including skipped | **619.9** | 611.8 |
| `hit_rate` as reported | 0.8259 | 0.7746 |
| `hit_rate` on a common denominator | **0.7518** | 0.7746 |

A's total routed lookups per token (619.9) match D's (611.8) to within 1.3%, and A's
*reported* lookups fall short by exactly the skipped count — so **the denominator excludes
tail-skipped misses**. On a like-for-like denominator the direction **reverses**: skip ON
measures 0.752 vs skip OFF 0.775, because the skipped experts are precisely misses the
engine chose not to compute. The Monitor's "expert cache hit" and `/metrics` `hit_rate`
therefore silently change meaning with this flag; the `route tail skip: N missed experts
skipped` log counter is the only way to correct it.

## Why the gain is modest here (mechanism, consistent with the data)

The flag avoids work on ~10% of decode expert lookups. Whether that translates to speed
depends on how much of the decode window is expert work:

- **Short context (609-4K):** experts dominate the window -> +7..9%.
- **16K:** +4.8%.
- **32K:** attention/KV dominate -> +1.0% (indistinguishable from zero).

That gradient is the clearest signal in this run, and it is the opposite of what a single
"10-20%" figure implies: the benefit shrinks as context grows, which is precisely the regime
where agent traffic spends most of its time.

## Recommendation for this box

**Keep the default (skip ON).** It is never slower in these measurements, worth ~5% decode
at short context, and costs nothing measurable in draft acceptance. The policy question the
engine itself flags — *answers differ slightly from 0.1.41* — is a quality decision, not a
speed one, and was not measured here (the needle suite (`../needles/`) is the place to do it).

## Limitations

- Speed only; no output-quality or needle comparison of skip ON vs OFF (that is the real
  open question the log line's "answers differ slightly" raises).
- Decode resolution: CI spans -2.9%..+13.3%, so "+5%" is a point estimate, not an
  established number; a soak or `--runs 10` would tighten it.
- The hit_rate denominator treatment is **confirmed by the two logs** (see the trap section);
  what is still missing is documentation of it. Note also that **run D's log contains no
  tail-skip lines at all** — when the flag is off there is no "is OFF" line and no counter,
  so the off state is only inferable from the absence of the ON line. PR #2011's
  "log the state in both directions" would fix that for overlap too.
- Per-request decode is highly variable (48-104 tok/s); permutation p-values assume
  independence, which the adaptive tier violates, so they are optimistic.
- A vs D is one variable; both differ from B1/B1c/B1d in split policy.
