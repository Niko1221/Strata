# Community benchmark: needle-in-a-haystack recall (daily driver, 0.1.43 defaults)

Measured on 2026-10-12 by @mmbeanz. The correctness counterweight
to the speed suite, on the **daily driver with defaults** (`--layer-split 40`,
`--pcie-frac 0.00`, route tail-skip ON, ADAPT_OVERLAP ON), cold boot, 0 requests before the
run. Tool: `tools/needle_bench.py --lengths 32k,128k,256k --depths 10,50,90 --timeout 900`
(haystack from the tool's `third_party/` text, code word hidden at a depth, greedy, thinking
off, normal `/v1/chat/completions` path).

**Headline: 9 of 9 found** — perfect recall at 32K, 128K and 256K, at depths 10/50/90%, on a
16+12 GB unequal GPU pair with a 12 GB card carrying 8 layers. This is the quality anchor for
the speed findings: the tuned split (and the tail-skip default that is on for these runs)
does not cost needle recall at any tested length.

## Results (`needles-final.json`)

| length | depth | prompt tokens | seconds | tok/s incl. generation | result |
| --- | ---: | ---: | ---: | ---: | --- |
| 32k | 10% | 32,804 | 21.7 | 1,512 | **FOUND** |
| 32k | 50% | 32,804 | 19.8 | 1,657 | **FOUND** |
| 32k | 90% | 32,805 | 19.8 | 1,657 | **FOUND** |
| 128k | 10% | 124,963 | 66.6 | 1,876 | **FOUND** |
| 128k | 50% | 124,961 | 66.6 | 1,876 | **FOUND** |
| 128k | 90% | 124,961 | 59.2 | 2,111 | **FOUND** |
| 256k | 10% | 253,177 | 129.9 | 1,949 | **FOUND** |
| 256k | 50% | 253,177 | 130.2 | 1,945 | **FOUND** |
| 256k | 90% | 253,178 | 130.1 | 1,946 | **FOUND** |

No misses, no hallucinated or near-miss answers, no position dependence (depth 10 vs 90
identical within run-to-run noise).

### Two side observations from the timings

1. **Prompt throughput rises with length, then plateaus:** ~1,600 tok/s at 32K, ~1,950 at
   128K and 256K (these include a short generation, so they slightly understate the engine's
   own prompt figure; the suite's 32K number was 1,720.8 tok/s).
2. **Cold vs warm matters at 128K.** An earlier pass with warm expert cache and OS page
   cache read the same 128K prompt in **50.1 s (2,494 tok/s)**; here, on a cold engine, the
   same size takes **66.6 s (1,876 tok/s)** — roughly a third slower cold. Worth stating
   whenever quoting prompt tok/s.

## Scope note: 256K is *within* the trained window, not beyond it

The model's native context is **262,144**; our config runs 524,288 via yarn factor 2.0. A
"256k" probe is **253,177 tokens ≈ 0.97x the trained window**, so these results show recall
holds *up to* the trained length — they do **not** test yarn extrapolation. Testing beyond it
needs ~393K-480K probes (the tool skips a length if `tokens + 200 > ctx`, and the answer
needs room too, so ~480K is about the practical ceiling at 524,288 context). That is a
follow-up row, not claimed here.

## Tool finding: the summary hides failures

`needle_bench.py`'s summary counts only rows carrying a `found` field
(`done = [r for r in rows if "found" in r]`), so errored probes vanish from the headline:
an earlier pass on this box printed **`4 of 4 found`** while 5 of 9 requests had actually
errored. For a correctness check that is the dangerous kind of silent truncation.
Suggested fix: print `completed / attempted`, or exit non-zero when any probe errored. The
published `needles-final.json` has no error rows: 9 attempted, 9 found.

Also noted: our config has the **conversation cache disabled** (`/metrics`:
`conversation_cache.enabled=false, budget_mib=0`), so the tool reusing one haystack across
the three depths of a length buys nothing — every probe is a cold read.

## Follow-ups suggested

1. **Beyond trained context:** ~393K and ~480K probes, to actually price yarn factor 2.0
   (the deep-dive warning is that static yarn can cost quality at long range).
2. **Tail-skip quality:** the same 9 probes with `STRATA_ROUTE_TAIL_SKIP=0`, to price the
   engine's own caveat *"answers differ slightly from 0.1.41"*. Our speed run (../D/)
   measured ~+5% decode for the default; this would measure what it costs.

## Limitations

- One needle per cell, single sample, no repeats: a strong smoke test, not a recall
  benchmark with error bars.
- Haystack is the tool's `third_party/` text, so domain-specific; greedy + thinking off.
- 256K is within the trained window (see scope note); nothing beyond it tested yet.
- Timings include a short generation, so they understate prompt-only throughput.
