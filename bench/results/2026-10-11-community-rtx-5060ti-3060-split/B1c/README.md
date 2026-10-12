# Community benchmark B1c: `--layer-split 6` — the engine's suggestion, followed literally, is catastrophic

Measured on 2026-10-11 by @mmbeanz. Same box / pinned profile / suite as
`../A/` (K=40) and `../B1/` (auto -> K=3).

**Headline.** `auto` flagged K=6 as a near-tie and printed *"To try the other:
`--layer-split 6`"*. We did exactly that, and it is **-68% vs B1 and -76% vs A on the
prompt path**. The reason is a trap in how the suggestion reads: **`--layer-split N` counts
layers for CUDA0, and an explicit split does NOT get the #1352 card reorder that `auto`
applies.** In `auto`'s frame CUDA0 was the slow RTX 3060 (3 layers); pasted literally, CUDA0
is the fast RTX 5060 Ti, so `6` put **6 layers on the fast card and 42 on the slow one** —
the inverted, pathological assignment. The advice is correct in its own frame and
unusable in the copy-pasted one.

## Evidence: which card is which (from the engine logs)

| run | split arg | reorder? | CUDA0 | CUDA1 | layers on the slow card |
| --- | --- | --- | --- | --- | ---: |
| A | `40` (explicit) | no | **5060 Ti**, layers 0-39 | 3060, layers 40-47 | 8 |
| B1 | `auto` | **yes**, order 1,0 (#1352) | **3060**, layers 0-2 | 5060 Ti, layers 3-47 | 3 |
| B1c | `6` (explicit) | **no** | **5060 Ti**, layers 0-5 | 3060, layers 6-47 | **42** |

Card identity is unambiguous in the logs (total VRAM and link probe):

```
B1 : layer split, CUDA1: 9.21 GiB free of 15.57          -> CUDA1 = 5060 Ti (16 GB)
     layer split: CUDA1 PCIe probe 28.7 GB/s             -> the gen5 link, i.e. the 5060 Ti
     layer split: CUDA1 runs layers 3-47, expert cache 4399 slots
B1c: layer split, CUDA1: 5.68 GiB free of 11.68          -> CUDA1 = 3060 (12 GB)
     layer split: CUDA1 PCIe probe 3.2 GB/s              -> the gen3 x4 link, i.e. the 3060
     layer split: CUDA1 runs layers 6-47, expert cache 2539 slots
     WARNING: prompt chunk 7424 ... CUDA1 (NVIDIA GeForce RTX 3060) has 2539 expert-cache slots
```

That last WARNING line names the card outright: in B1c the **3060** carries layers 6-47.

## What the misassignment costs, per resource

**1. Layer time.** The 3060 is 1.9x slower per layer (engine's own figures: 1.46 ms/layer
vs 0.78 ms/layer). B1c puts 42 of 48 layers on it: `42 x 1.46 + 6 x 0.78 ~ 66 ms` per
window, versus `40 x 0.78 + 8 x 1.46 ~ 42.9 ms` in A. Decode collapses accordingly.

**2. Stranded VRAM on the fast card.** A stage can only cache **its own** layers' experts,
and 6 layers x 512 experts = 3,072 — so the 5060 Ti's cache is capped by layer count, not
memory:

```
B1c: expert cache auto: 13.28 GiB free, 700 MiB reserved -> 3072 slots   (5.52 GiB used)
```

**~7 GiB of the 16 GB card's free VRAM cannot be used for experts at all**, while the 3060
gets only 2,539 slots (4.90 GiB) for its 42 layers (21,504 pairs = 11.8%).

**3. Hit rate.** 0.669 median (min 0.309) vs 0.859 (A) / 0.865 (B1).

**4. Prompt path — and a correction to the B1 story.** B1c's chunk was **7,424** (only
lightly clamped), yet prompt throughput was 407.9 tok/s. So the prompt collapse here is
**not** the chunk clamp: it is prefill spending most of its layer work on the slow card.
B1's -25% prompt loss (chunk 3,328, fast card doing 45 layers) *is* the chunk clamp. The
two effects are separable, and B1c isolates the layer-assignment one.

## Results

| Configuration | Prompt tok/s (median) | Decode tok/s (median) | TTFT s (median) |
| --- | ---: | ---: | ---: |
| short chat (9 runs) | n/a | 41.4 (35.6-49.9) | 0.72 (0.60-0.78) |
| 4K prompt, fresh | 264.0 (240.6-264.7) | 38.7* | 15.26 (15.21-16.73) |
| 16K prompt, fresh | 346.2 (338.9-346.8) | ~50* | 46.42 (46.34-47.43) |
| 32K prompt, fresh | 407.9 (406.8-408.3) | 32.3* | 78.61 (78.54-78.82) |
| agent follow-up turns (15) | see `results.json` | 38.6 (27.1-52.2) | 15.38 (14.51-18.30) |

\* engine per-request `decode_tok_s` from `/metrics?requests=all` (24-token cap).

### Three-way comparison

| metric | A (K=40, fast card 40 layers) | B1 (auto K=3, fast card 45) | B1c (fast card 6, slow card 42) |
| --- | ---: | ---: | ---: |
| prompt 32K tok/s | **1,720.8** | 1,290.9 | **407.9** |
| prompt 4K tok/s | 868.1 | 835.6 | 264.0 |
| TTFT 32K (s) | 18.66 | 24.86 | 78.61 |
| short-chat decode | 74.4 | 66.6 | 41.4 |
| decode 32K | 53.2 | 54.9 | 32.3 |
| expert hit_rate (median / min) | 0.859 / 0.550 | 0.865 / 0.543 | **0.669 / 0.309** |
| held experts (total) | 8,842 | 5,935 | 5,611 |
| slots on the card doing most layers | 4,984 (40 layers) | 4,399 (45 layers) | 2,539 (42 layers) |
| prompt chunk (tokens) | 8,192 | 3,328 | 7,424 |
| `pcie_share` median / max | 0 / 0 | 0.08 / 0.247 | 0.004 / 0.019 |

## Coverage: what actually tracks hit rate

| run | held total | f | old `(r+1)^-1.2` | new `H(f)` b=3 | **measured hit_rate** | slots per layer on the busy card |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 8,842 | 0.360 | 97.0% | 73.8% | **85.9%** | 124.6 |
| B1 | 5,935 | 0.241 | 95.6% | 56.4% | **86.5%** | 97.8 |
| B1c | 5,611 | 0.228 | 95.4% | 54.0% | **66.9%** | 60.5 |

B1 and B1c hold nearly the same **total** number of experts (5,935 vs 5,611) and differ by
**20 points of hit rate**. A count-based coverage curve cannot express that: both curves
predict 95.6/95.4% (old) and 56.4/54.0% (new) for the two runs, i.e. essentially no
difference. What does track the measurement is **cache per layer on the card that runs most
of the layers** (124.6 -> 84%, 97.8 -> 86%, 60.5 -> 62%). Decomposing per stage under the
assumption that each layer gets a similar share of lookups:

| run | card doing most layers | its slots | implied coverage of its own routed mass |
| --- | --- | ---: | ---: |
| A | 5060 Ti, 40 layers (83% of lookups) | 4,984 | ~84% |
| B1 | 5060 Ti, 45 layers (94% of lookups) | 4,399 | ~86% |
| B1c | 3060, 42 layers (88% of lookups) | 2,539 | ~62% |

This qualifies the 0.1.43 re-fit comment *"hit rate follows the held COUNT and almost
nothing else"*: on this pair the total count is the one thing hit rate does **not** follow.
Placement across stages is what matters, because a stage can only cache its own layers.

> **Update from `../B1d/` (K=42):** the "slots per layer on the busy card" heuristic above
> (124.6 -> 84%, 97.8 -> 86%, 60.5 -> 62%) is **not monotonic** once a fourth point is added
> (K=42: 116.0 slots/layer but hit 0.818). The model that does hold is **fraction of each
> layer's own pairs that the owning card can cache** — see `../B1d/README.md`, which also
> adds permutation significance tests (A vs B1: no difference in hit rate despite a 33%
> difference in held experts).

Caveats: `hit_rate` is the engine's **decode-phase** expert-cache hit rate (the `/metrics`
per-request values match the log's `decode expert cache hit rate` lines exactly: min 55.0% /
max 96.9% in A); prefill routing is not covered. The per-stage implied coverage assumes
roughly uniform lookups per layer.

## Tool / docs findings from this run

1. **The `--layer-split N` suggestion from `auto` is not copy-pasteable.** `auto` reorders
   cards so the faster one is last (#1352); an explicit `--layer-split` does not reorder, so
   `N` counts layers for a *different* card than the search had in mind. Following the
   engine's own advice produced a 3-4x slowdown. Suggested fix: print the suggestion in the
   as-given frame (here `--layer-split 42`), or state the card order next to it.
2. A stage's expert cache is capped by `its own layer count x 512`, which can strand large
   amounts of VRAM on a card given few layers (7 GiB stranded on a 16 GB card here). Worth a
   log line, since the current one (`13.28 GiB free ... -> 3072 slots`) reads as if memory
   were the limit.
3. Previously reported (see `../A/README.md`): `gpu_slug()` names multi-GPU folders after
   the first card only; `collect_machine()` blends client/server fields under `--url`;
   `/metrics` defaults to 12 requests (`?requests=all` for the rest); the generated README
   cites `engine-log-excerpt.txt` in `--url` mode where none is written.

## Correct next test: B1d = `--layer-split 42`

To test the engine's intended "K=6 on the slow card" in the as-given card order, the
equivalent explicit split here is **`--layer-split 42`** (5060 Ti: layers 0-41; 3060:
layers 42-47, i.e. 6 layers, 3,072 slots, ~4.0 GiB, and the 5060 Ti keeps ~4,000 slots for
42 layers). Predicted window ~41.6 ms vs A's ~42.9 ms, with more of the 3060's VRAM used and
the fast card's cache roughly preserved. If that recovers A's prompt speed and beats its
decode, it is a concrete improvement to report; if not, K=40 stands as the tuned answer.

## Limitations

- Speed only; no needle/quality check (needle suite: `../needles/`).
- One suite per config (3 runs), same evening.
- A vs B1 vs B1c differ in split policy, card order and chunk size simultaneously; B1d is
  the clean single-variable test of the K=6 idea.
- No concurrency, no vision latency, no mmap/low-RAM tier, Windows/HIP untested.
