# Community benchmark B1d: `--layer-split 42` — the as-given-frame equivalent of the engine's K=6 idea

Measured on 2026-10-11 by @mmbeanz. This is the **clean single-variable
test**: identical to run A (`../A/`, `--layer-split 40`, `--pcie-frac 0.00`) except
`40` -> `42`. It asks the question `auto` raised ("K=6 on the slow card, 49.2 vs 48.4 ms,
near tie — try `--layer-split 6`") in the correct card frame, since an explicit split does
not get the #1352 reorder (see `../B1c/README.md`).

**Headline.** K=42 is a **wash against K=40 on speed** (prompt 32K -2.2%, short-chat decode
+2.4%, decode 32K +11.7% on 3 samples, agent decode -0.5%) but its **expert hit rate is
significantly lower** (0.818 vs 0.859 median, permutation p = 0.024). Conclusion for this
pair: **K=40 stands as the tuned answer**; moving two more layers onto the 12 GB card costs
cache coverage on the 16 GB card without buying anything back.

## Layout (from `GET /metrics`, card identity per the A/B1c log convention)

| | CUDA0 = **5060 Ti** (16 GB, 0.78 ms/layer) | CUDA1 = **3060** (12 GB, 1.46 ms/layer) | total |
| --- | --- | --- | ---: |
| layers | 0-41 (42) | 42-47 (6) | 48 |
| expert slots | 4,874 (9,316 MiB) = 22.7% of its 21,504 pairs | 3,072 = **100%** of its 3,072 pairs | **7,946** (f = 0.323) |

Compare A: 5060 Ti 40 layers / 4,984 slots (24.3% of 20,480); 3060 8 layers / 3,858 slots
(**94%** of its 4,096). Predicted decode window: `42x0.78 + 6x1.46 ~ 41.6 ms` vs A's
`40x0.78 + 8x1.46 ~ 42.9 ms` — the engine's model says 42 should be marginally faster, and
the short-context measurements agree; the prompt path and hit rate do not.

## Hardware, software, model

Identical to `../A/README.md` (hand-written tables): 7900X 12c/24t, 64 GB DDR5-6400,
5060 Ti 16 GB on PCIe 5.0 x8 (H2D 28.2-28.7 GB/s) + 3060 12 GB on PCIe 3.0 x4 behind the
chipset (H2D 3.15-3.2 GB/s), 990 EVO Plus 4 TB DRAM-less, driver 615.71.09, CUDA 13.4,
systemd-nspawn on bare metal, Strata 0.1.43 from `7695859a` + `setup.py`-only local commit.
IQ3_S GSQ-RCO pack (sha256 in `../sha256sums.txt`), 512K yarn 2.0, pinned
`data/expert-profile.bin` (24,576 ranked pairs), `pcie_frac 0.00`, `vram_free_mib 324`,
`pool_workers 11`. Config snapshot: `config.json`.

## Method

As in A/B1/B1c: `strata_bench.py --url <server> --config config.json --long --runs 3`,
operator-started server, cold boot, 0 requests before the suite;
`GET /metrics?requests=all` before/after. Client on a separate LAN box (RTT 0.25 ms); the
`machine` block blends client/server fields (see `../A/README.md`, *Tool findings*).
Suite ~205 s.

## Results

| Configuration | Prompt tok/s (median, range) | Decode tok/s (median, range) | TTFT s (median, range) |
| --- | --- | --- | --- |
| short chat (9 runs) | n/a | 76.2 (70.2-93.1) | 0.40 (0.36-0.45) |
| 4K prompt, fresh | 888.8 (881.3-891.2) | 70.8* | 4.55 (4.54-4.59) |
| 16K prompt, fresh | 1397.3 (1395.9-1400.4) | 58.3* | 11.52 (11.50-11.54) |
| 32K prompt, fresh | 1683.4 (1682.4-1684.6) | 59.4* | 19.08 (19.06-19.09) |
| agent follow-up turns (15) | see `results.json` | 80.3 (49.5-101.3) | 4.16 (3.21-4.57) |

\* engine per-request `decode_tok_s` from `/metrics?requests=all` (24-token cap, short
window — indicative only).

### A (K=40) vs B1d (K=42) — one variable

| metric | A | B1d | delta |
| --- | ---: | ---: | ---: |
| prompt 32K tok/s | 1,720.8 | 1,683.4 | -2.2% |
| prompt 16K tok/s | 1,412.7 | 1,397.3 | -1.1% |
| prompt 4K tok/s | 868.1 | 888.8 | +2.4% |
| TTFT 32K (s) | 18.66 | 19.08 | +2.2% |
| decode 32K tok/s | 53.2 | 59.4 | +11.7% (n=3, 24-token windows) |
| decode 16K tok/s | 58.3 | 58.3 | 0.0% |
| decode 4K tok/s | 70.7 | 70.8 | +0.1% |
| decode 609 tok/s | 71.6 | 74.0 | +3.4% |
| short-chat decode | 74.4 | 76.2 | +2.4% |
| agent decode | 80.7 | 80.3 | -0.5% |
| expert hit_rate (median) | 0.859 | **0.818** | **-4.1 pts** |
| held experts (total) | 8,842 | 7,946 | -10% |
| `pcie_share` | 0 / 0 | 0 / 0 | identical policy |

Speed: within a few percent in both directions — a **wash**. Hit rate: significantly lower
(see below).

## Statistics on the hit-rate measure

Per-request decode expert-cache hit rate, n = 40 per run:

| run | held | f | median | mean | IQR | 10-90 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| A (K=40) | 8,842 | 0.360 | 0.859 | 0.842 | 0.774-0.915 | 0.711-0.945 |
| B1 (auto K=3) | 5,935 | 0.241 | 0.865 | 0.840 | 0.783-0.913 | 0.730-0.922 |
| B1c (mis-set K=6) | 5,611 | 0.228 | 0.669 | 0.652 | 0.532-0.758 | 0.485-0.793 |
| B1d (K=42) | 7,946 | 0.323 | 0.818 | 0.787 | 0.703-0.876 | 0.624-0.921 |

Two-sample permutation tests on the means (5,000 shuffles, two-sided):

| pair | mean diff | p | verdict |
| --- | ---: | ---: | --- |
| A vs B1 | +0.002 | 0.93 | **not significant** |
| A vs B1d | +0.055 | 0.024 | significant |
| B1 vs B1d | +0.053 | 0.030 | significant |
| A vs B1c | +0.189 | <0.0001 | significant |
| B1d vs B1c | +0.135 | <0.0001 | significant |

**A and B1 are statistically indistinguishable in hit rate despite a 33% difference in held
experts (8,842 vs 5,935).** That is the strongest evidence here against count-based coverage
curves: the old power law predicts 97.0% vs 95.6%, the new `H(f)` b=3 predicts 73.8% vs
56.4%, and the measurement says "no difference" at ~86%.

## What the four points do support: per-layer coverage

A stage can only cache **its own** layers' experts, so the quantity that behaves is
"what fraction of each layer's routed mass can the owning card hold":

| run | small card (3060) | big card (5060 Ti) | measured hit |
| --- | --- | --- | ---: |
| A | 8 layers, 3,858/4,096 pairs = **94%** | 40 layers, 4,984/20,480 = 24.3% | 0.859 |
| B1 | 3 layers, 1,536/1,536 = **100%** | 45 layers, 4,399/23,040 = 19.1% | 0.865 |
| B1d | 6 layers, 3,072/3,072 = **100%** | 42 layers, 4,874/21,504 = 22.7% | 0.818 |
| B1c | **42 layers**, 2,539/21,504 = 11.8% | 6 layers, 3,072/3,072 = 100% | 0.669 |

Reading A -> B1d: moving layers 40-41 from the 3060 (which held ~94% of their experts) to
the 5060 Ti (which holds ~23% of its pairs) costs those 2/48 = 4.2% of lookups roughly 70
points of coverage -> ~3 points of overall hit rate. Measured: 4.1 points. Same order of
magnitude, right direction. B1c is the same effect taken to the extreme (42 layers on the
card that can barely cache any of them).

Why A and B1 tie despite different totals is the open part: between 3 and 8 layers on the
small card the small card is essentially fully cached either way, and the big card's
fractional coverage (19.1% vs 24.3%) apparently does not move the aggregate — consistent
with the routing being concentrated enough that ~19% of pairs already captures most of the
mass on this model. That is a claim worth testing with a K sweep (36/40/44), not something
these four points can settle.

## Practical conclusion for this pair

**Keep `--layer-split 40`.** K=42 is not better (equal speed, significantly worse hit rate),
`auto`'s K=3 is clearly worse on speed (-25% prompt, -10..-19% short decode) even though its
hit rate matches, and the engine's literal `--layer-split 6` suggestion is catastrophic
because it inverts the card assignment.

## Limitations

- Speed only; no needle/quality check (needle suite: `../needles/`).
- Decode for the capped prompt tests is a 24-token window (n=3) — the +11.7% at 32K is not
  established, only suggestive.
- Hit-rate significance tests treat per-request values as independent samples; the adaptive
  tier makes consecutive requests correlated, so p-values are optimistic.
- A vs B1d is one variable; A/B1 vs B1c/B1d also differ in card order and PCIe share.
- No concurrency, no vision latency, no mmap/low-RAM tier, Windows/HIP untested.
