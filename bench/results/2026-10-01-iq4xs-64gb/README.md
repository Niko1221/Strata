# 2026-10-01: IQ4_XS on a 64 GB PC — pinned arena vs `experts.bin` mmap vs the RAM budget

**Question.** 60.94 GiB of routed experts on a machine with 64 GB of RAM. Which of the three ways
the engine can serve them is actually fastest here: the pinned arena (the default), `--mmap-experts`
over the pack's `experts.bin`, or the 0.1.31 path — read the GGUF in place and pin a
`--resident-budget-gib` of the hottest experts?

**Machine.** Ryzen 7 5700X3D (Zen 3, AVX2, no AVX-512), 64 GB DDR4-2666 dual channel, RTX 5060 Ti
16 GB at PCIe 4.0 x8 (the engine's own probe: 14.1 GB/s host→device), NVMe sequential read measured
2.41–2.46 GB/s, Windows 11.

**Model.** `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF`, IQ4_XS: 60.94 GiB of routed experts,
48 layers × 512 experts, 2.66 MB per expert blob, gate/up IQ4_XS + down IQ4_NL. Engine 0.1.31 and
0.1.32 built from source (with the AVX-2 IQ4_XS gate kernel).

**Method.** Offline `strata generate --tokens-file <frozen prompt> --max-new 300 --stats`, arms
interleaved, **arm order rotated one position per round**, one unmeasured warmup run. The primary
metric is **ms per draft round at equal draft-round counts**. tok/s alone is not comparable across
runs: the GPU hit path rounds differently from the CPU (the engine says so at start), so the draft
acceptance moves run to run — the same arm with the same arguments produced 112–115 rounds in five
runs and 133 in one, which alone moves tok/s by ~10%.

## The three paths

| path | cold | warm | expert bytes read per round |
|---|---:|---:|---:|
| pinned arena (no `--mmap-experts`) | 27.90 | 33.85 | — (60.94 GiB loaded at 1.88–2.19 GiB/s, every run) |
| `--mmap-experts` over `experts.bin`, setup's arguments | 14.06 | 16.88 | ~1.2 GB |
| same + `--kv-resident 20480` + `STRATA_IQ_PREFETCH=16384` | 33.3 | **37.5–39.0** | ~1.2 GB, 28 GB/s over the rows |
| GGUF read in place (pack without `experts.bin`) | 9.63 | 9.74 | **1325 MB** |
| GGUF in place + `--resident-budget-gib 40` | 26.7 | 26.0 | 42 MB |
| GGUF in place + budget 40 + the two knobs above | 26.7 | 27.5 | 34 MB |
| `experts.bin` + `--resident-budget-gib 40` | 24.7 | 25.4 | 0 (all from the RAM copy) |
| reference: the same model's IQ3_S, arena | 50.6 | 51.5 | ~0.94 GB |

So on this machine the order is **mmap over `experts.bin` > pinned arena > resident budget > GGUF in
place**, and the spread between the best and the worst way to run the same file is **4×**.

## What each number came from

**1. The arena does not pin, and at this size that flips the ranking.** The comment in
`generate.cpp` measures the arena at *1.79x better than the warm mmap and 3.7x better than the cold
one*, on a **34 GB `experts.bin` on a 63 GB machine**, and it already states that the arena is not
pinned there (`cudaHostRegister` on 31.64 GiB fails with "out of memory" - "you cannot pin 34 of
63 GB"). The same failure here, at a larger size:

```
expert arena: SetProcessWorkingSetSizeEx(14814 MiB) failed (error 1450); cudaHostRegister of the
whole arena FAILED (out of memory); 37 slices pinne[d]
```

(the line is cut at 165 columns by the harness, so the tail is not recorded). Large pages are refused
too, error 1450, for both the 61 GiB and the 50.3 GB arena. What changes at 60.94 GiB is the
consequence: the arena still runs - 33.85 tok/s - but it pays ~30 s to load 60.94 GiB at ~2 GiB/s on
**every** start, and it now **loses** to a warm mapped run (37.5-39.0). `cudaHostRegister` does
**succeed** for the IQ3_S arena (`cudaHostRegister PORTABLE ok` on 50,295,996,416 B), which is why
the arena is the right answer one quantization step down and the wrong one one step up.

**2. The RAM budget costs ~30% against plain mmap, and the two costs are visible in the log.**
`eb_sp` (mmap over `experts.bin`) 38.06 tok/s vs `eb_rb40sp` 27.45:

```
adaptive tier   2688 experts swapped into the VRAM tier (every 4 rounds, 7.580 ms/round)
```

7.6 ms/round of PCIe traffic that the mapped mode does not have, and:

| | GB/s over the rows |
|---|---:|
| mmap, page cache | 27.4 |
| resident budget, page-locked complement | **15.5** |

The pinned complement is good for the device and slower for the CPU pool that reads it.

**3. Page-locked RAM evicts everybody else's page cache.** Same arm, same arguments, two rounds,
different position in the round: `eb_sp` 38.06 tok/s / 27.4 GB/s in round 1, 18.59 / **10.6** in
round 2 — because two 40 GiB page-locked allocations ran in between and are not evictable. Any
comparison that mixes mapped and budgeted arms in one round is measuring the order, not the mode.

**4. `STRATA_IQ_PREFETCH` (the AVX-2 row prefetch) wants a bigger distance on the mapped path.**
Warm, ms/round at equal round counts:

| distance | ms/round (114 rounds) | tok/s |
|---|---:|---:|
| 2048 (default) | 73.4–73.9 | 35.3–35.5 |
| 8192 | 67.30 | 38.9–39.1 |
| 16384 | 67.55 | 39.0 |
| 24576 | 67.33 | 39.1 |
| 65536 | — | −0.6% over 16384 |

The default is tuned for ~1 KB rows on the arena path; an IQ4_XS row is 1360 B and on the mapped
path the prefetch also anticipates the page fault of the mapped page, so 8–24 KB is a plateau and
2048 B sits ~9% under it. On the arena path (IQ3_S) 16384 is not harmful either: 51.53/51.56 vs
50.56/51.26 with the default.

**5. Lookahead routing prefetch: no measurable effect at this size.** `--resident-budget-gib 40`
with and without `STRATA_LOOKAHEAD=0`: 26.74/25.94 vs 26.25/26.00 tok/s. With 42 MB per round of
file reads there is almost nothing left to warm.

**6. Two refusals reproduced.** `--resident-cpu-experts` on the mapped mode: *"resident complement
53.40 GiB exceeds available RAM minus the 8 GiB safety headroom"* — which is what
`--resident-budget-gib` exists for. And budget 50 hits #403 from the other side:

```
FileExpertSource: resident complement 46.70 GiB exceeds available RAM (50.70 GiB) minus the 4 GiB safety headroom
```

after a run at the same budget had already succeeded. 40 is the usable ceiling on 64 GB here.

**7. `--expert-cache-per-layer` does not start on a native IQ pack in mmap** (#369):
`ExpertCache::verify_slot: slot 0 differs from the arena at byte 0 (of 2662400)`.

**8. The AVX-2 multi-token kernel for IQ4_XS gate/up rows is worth ~3% here.** A/B with
`STRATA_NO_IQ256=1` (which drops that path; the IQ4_NL `down` rows have their own switch and stayed on
in both arms), four interleaved rounds, order rotated, compared as total time for the run: the kernel
wins **4 of 4 pairs**, median **−3.2%** (−2.6% with one outlier round dropped), gate/up phase −7.6%,
sustained row reads 28.6 against 25.6 GB/s. Small because both arms sit at the memory wall — the win
is walking each expert row once per verify window instead of once per token.

## What I would do with this

- The arena-vs-mapped ranking in the source comment is measured at 34 GB of experts on 63 GB, where
  the unpinned arena still wins. At 61 GiB on 64 GB it loses, and the load cost per start (~30 s)
  becomes visible. The second data point is worth putting next to the first.
- The prefetch default (2048 B) is right for the arena path and ~9% low for the mapped one; a
  distance that follows the row size (or 8192) would suit both.
- A warning when a page-locked budget is large enough to push out the file cache the same model is
  reading from would save the next person a confusing benchmark.

## Caveats

One machine, one model, offline mode (no server, no KV streaming across requests), 300 tokens per
run, greedy + `--spec 4`. The tok/s columns are reproducible to ±1.5% at fixed draft-round counts;
the cold columns depend on how much of the file the previous run left in RAM.
