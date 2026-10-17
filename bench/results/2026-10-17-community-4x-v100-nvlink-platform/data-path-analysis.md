# The data-transfer profile of Strata on 4x V100-SXM2 (layer-split auto)

```
Artifact:            data-path-analysis.md (consolidates what was measured earlier; no new measurements)
Date (UTC):          2026-10-17
Sources of numbers:  (1) raw/profiles/2026-10-17T004753Z/ - this machine's baseline profile:
                       baseline-points.csv (per-card pcie_rx_mb_*/pcie_tx_mb_*, dmon windows),
                       engine-log-delta.txt / engine-log-summary.txt ("… MB of expert reads",
                       "the OS read … MB (major faults)", hit rate);
                     (2) raw/ab/2026-10-17T061723Z-validation3/ - the A/B matrix: engine-request-counters.csv
                       (the same for auto and for the pinned 12,25,37), bench/baseline-points.csv
                       (per-card PCIe); the analysis is in ab-stand.md;
                     (3) PR #2525 interconnect-experiments/{baseline,control,help-on,help-forced,…}/phases.json,
                       uplink-saturation.json, nvlink-probe-phases.json - the phase profile and uplink saturation;
                     (4) p2p-results.md - bandwidth and latency of both inter-card paths.
Configuration:       Strata 0.1.44.1 (binary sha256 bdfec9f5…), config sha256 f539e51b…,
                     --layer-split auto / --expert-cache auto / --prefill auto, prompt chunk auto 8192 tokens,
                     context 262 144, KV int8, spec 4 (draft model), concurrency/serving = 1.
Units:               MB = 10^6 bytes (as in the engine's counter); GB = 1000 MB. PCIe numbers are
                     nvidia-smi dmon rxpci/txpci (cumulative, 1 s sampling) -> a lower bound.
Status labels:       measured · calculated · hypothesis · not possible in this build.
```

## 1. Method and what is actually measured

Three independent sources, none of which counts bytes on NVLink:

| Source | What it counts | Boundary |
|---|---|---|
| the engine's lines `for … MB of expert reads` + `the OS read … MB from storage (N major faults)` | the volume of expert-blob accesses per request, and the part that reached the OS | the counter's exact semantics (logical accesses vs bytes served from host) are not established from the code here; `major faults = 0` in every run means the stream comes from the page cache, not from disk |
| `nvidia-smi dmon` `rxpci`/`txpci` per card | bytes that actually crossed each card's PCIe | 1 s sampling -> short bursts are under-counted (a lower bound) |
| `interconnect-experiments/*/phases.json` (PR #2525) | the same dmon, split by phase (idle / prefill / decode) | taken on this machine in another session |

**The tool that does not exist:** NVLink byte counters. DCGM 4.7.0 without the profiling module; NVML Sampling & Reporting is absent in both driver 580.178.04 and 595.91.07 (checked against the symbol tables). So the routing conclusion is built by "counting PCIe in reverse": if a volume computed from code and slots exceeds what PCIe can carry, it cannot have gone over PCIe. **Not possible in this build:** direct confirmation that "N GB went over NVLink".

## 2. The measured scale: expert accesses and PCIe per request (this machine, `auto`)

The baseline profile, 3 repeats per point, output length fixed:

| case | `prompt_n` | `… MB of expert reads`, median per request | per-card PCIe `rxpci` per request, GPU0/1/2/3, MB | sum RX | sum TX | ER / sum RX |
|---|---|---|---|---|---|---|
| 4k | 3 971 | 17 871 | 1 763 / 4 939 / 4 456 / 4 392 | 15 549 | 2 395 | 1.15 |
| 16k | 15 667 | 46 593 | 12 526 / 7 763 / 13 448 / 9 915 | 43 651 | 7 280 | 1.07 |
| 32k | 31 307 | 78 412 | 19 068 / 29 815 / 22 220 / 32 802 | 103 905 | 15 984 | 0.75 |
| 64k | 62 519 | 142 592 | 48 422 / 59 119 / 55 186 / 69 852 | 232 579 | 36 941 | 0.61 |
| 128k | 125 011 | **269 455** | 60 406 / 65 037 / 62 355 / **86 148** | **273 945** | 56 459 | **0.98** |
| decode-only (163-token prompt, 1024 generated) | 163 | 2 722…3 552 | 26 / 87 / 228 / 281 | 623 | 266 | 4.49 |

**What follows (confirmed by direction and order of magnitude):**

1. **Both volumes grow with prompt length and are nearly linear in each other** at 32k…128k (ER/sum RX = 0.61…0.98). At 125 011 tokens the engine's counter (269.5 GB) and the measured PCIe received (273.9 GB) agree to 2 % - that is the evidence that the dominant long-prompt flow is **host -> card** and consists mostly of expert-blob accesses. Independent confirmation from PR #2525: 270.5 GB of `rxpci` for the same request.
2. **Received exceeds sent by 4.9x** (273.9 against 56.5 GB at 125k) - the asymmetry is the opposite of what NVLink would carry (inter-card hand-offs are symmetric and small).
3. **The cost per token falls with length** (36.7 GB per 8192-token chunk at 4k -> 17.7 GB at 125k): the access profile warms up and the share of accesses to unlearned experts falls. So "the cost of a byte" is only valid within one length.
4. **GPU3 receives the most** (86.1 GB against 60.4…65.0 GB on the others) - it holds the lowest profile coverage (5469 of 6144 profiled pairs, 89.0 %) and therefore streams more. The same number reproduces in the A/B matrix and falls with a pinned split (§4).
5. **The decode path is small:** 2.7…3.6 GB of accesses for 1024 generated tokens (about 2.7…3.5 MB/token), and PCIe received during the same run is 0.62 GB. At 100 tok/s that is <= 0.35 GB/s, so decode is not uplink-bound (consistent with the measured GPU utilisation of 35-38 % on pure decode). **The conclusion for a reader: the decode path is not worth touching** (see also `patch-notes.md` §2.3: 307 KB/token of hand-off is <= 0.5 % of ITL).

**An honest discrepancy (a FINDING, not explained).** At 32k and 64k the measured PCIe received **exceeds** the expert-access counter by 1.33x and 1.63x, while at 4k and 16k the counter is above the received traffic (1.15x and 1.07x). Possible reasons (hypotheses, none tested): (a) the draft model and the MTP path access their own experts and are not in that log line; (b) `rxpci` under-counting with 1 s sampling on short windows; (c) part of the expert accesses is served from the VRAM cache and never crosses PCIe, while PCIe includes other traffic (prompt-path activations, KV, loading dense-layer weights under a trim). What matters for a reader: **the `… MB of expert reads` counter is not the same as bytes over PCIe**, and accepting a change requires recording both, not one.

## 3. The phase profile (data from PR #2525; only partly repeated here)

| Phase | Observed | Status |
|---|---|---|
| idle | `rxpci = txpci = exactly 0` on every card | measured (PR) - which is why attributing the non-idle phases is reliable |
| prefill (125 351 tokens) | 61.9-78.4 GB RX and 6.9-11.9 GB TX per card | measured (PR); **independently reproduced on this machine** at 125 011: 60.4-86.1 GB RX and 12.9-18.8 GB TX per card (§2) |
| decode | GPU utilisation 35-38 %, traffic <= 0.35 GB/s | measured (this machine) |
| saturation of the shared x8 uplink | median 5.25 GB/s, p95 7.67, worst second 9.33 against 7.88 nominal; 33 of 75 seconds at or above 6 GB/s | measured (PR); on this machine the per-request sums in §2 give 2.50 / 4.01 / 6.30 / 5.49 / 5.97 GB/s of average uplink load across the 4k…128k runs |
| `STRATA_PREFILL_HELP` / `…_FRAC` on/forced/control | speed does not change | measured (PR, two sessions + a control) |

## 4. Comparing two placement configurations (the A/B matrix, this machine)

A 122 087-token request, `raw/ab/2026-10-17T061723Z-validation3/` (three repeats per cell, two independent pairs of cells):

| Measure | `auto` (A0/A1/A0r) | `12,25,37` (B0/B1) | delta |
|---|---|---|---|
| `… MB of expert reads` per request | 253 528 / 253 618 / 253 624 | 240 442 / 240 440 | **-13 086 MB (-5.2 %)** |
| `the OS read … (major faults)` | 2.8…13.9 MB, 0 faults | 3.8…6.6 MB, 0 faults | streaming from the page cache in both configurations |
| sum `rxpci` over 3 requests at 122k | 776 960 / 776 949 / 790 666 MB | 744 681 / 755 592 MB | **-32 GB (-4.1 %)** |
| of which GPU3 | 236 050 / 224 377 / 247 495 MB | 197 593 / 209 835 MB | **-12 %** |
| hit rate at 122k | 99.5…99.8 % | 100.0 % | - |
| CUDA0 expert-cache slots / prompt-path borrowing | 5632 / 5178 | **6144** / 5259 | **+512 slots / +81 (not reduced)** |
| prompt tok/s | 2428.0 / 2426.1 / 2426.5 | **2589.8 / 2589.6** | **+6.66 %** |

**This is so far the only case where a placement change made three independent observations agree at once:** fewer expert accesses (-5.2 %), less PCIe received (-4.1 %), higher speed (+6.66 %). It also yields the empirical constant for "the price of a host byte" (about 3.9 GB/s, `patch-notes.md` §3.4) and shows that the gain was bought with **more cache and more coverage**, not with refusing to borrow slots.

## 5. How this profile connects to the code

The map of every inter-card and host path with volumes and file/line references is in **`patch-notes.md` §2-§3** (the prompt hand-off between stages, 32.2 GB per request from `prefill.cpp:4161/2189/1106`, `D = n_embd·HC = 10 240`, 16 chunks x 3 boundaries x 2 directions; the stage helper, 0 in the working config, `prefill.cpp:1591` `p2p = false`; the decode/verify hand-off `generate.cpp:7829`, `verify.hpp:177`; expert streaming `prefill.cpp:2380-2460`, `native_mmvq.cu`). This document is only the measurement half: which volumes are actually observed.

The key conclusion for the decision "move it or don't": **the dominant flow cannot be moved onto NVLink on this hardware** - 4x16 GB of VRAM against 50.29 GB of experts, the caches filled exactly, and no card holds copies of the experts another card needs (`patch-notes.md` §2.4). The movable volume is the 32.2 GB of inter-stage hand-off (0.12 % of the 269 GB stream), and it is small exactly to the degree that it is not the bottleneck.

## 6. Limits of this document

1. There are no NVLink byte counters, so nothing here says "X GB went over NVLink".
2. `rxpci`/`txpci` are a lower bound (1 s sampling); at short windows (4k, 16k) the under-count is noticeable.
3. The semantics of the `… MB of expert reads` counter are not established from the code; the 1.3-1.6x discrepancy against `rxpci` at 32-64k is unexplained (§2, FINDING).
4. Everything was taken at `concurrency/serving = 1`. Concurrent load was not studied.
5. The full phase profile was taken in PR #2525 (another session); here the per-card prefill sums and the decode profile were independently reproduced.
6. The hand-off volume (32.2 GB) is a calculation from the code that matched the PR's measured TX within 9 %; it was not measured directly (a direct measurement would need a kernel tracer).
