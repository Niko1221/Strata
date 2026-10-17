# CUDA P2P / NVLink: capability matrix, latency, bandwidth, correctness, pair competition

```
Artifacts:            p2p-matrix.csv, p2p-benchmark.csv, p2p-correctness.json, p2p-results.md (this file)
Primary data:         raw/p2p/2026-10-17T001200Z/  (phase B wrapper: before.csv, after-stop.csv, stop.log, probe.log, points-to.txt)
                      raw/p2p/2026-10-17T001226Z/  (the probe itself: conditions.txt, build.log, stdout.log, p2p-*.csv/json)
                      raw/p2p/2026-10-16T235659Z/  (phase A: the same 4 KiB…8 MiB ladder with the engine running)
                      raw/logs/2026-10-17T001623Z/ (confirmation that Strata returned to service)
Program:              probes/p2p-probe-ext.cu  sha256 0b2c7485085838abbb9861ba50497dd6bb9684d89964ace26c471830d3168247 (the version these data were taken with)
Binary:               build/p2p-probe-ext          sha256 968561c50e68a086ecc996cffdaa3a927a55b6171947afd2bb50b9869f4b8859
After the run:        the source was then fixed for the contention_pair rows (§9.1); the fixed version
                      sha256 d92b1fbbd401ace2b318e7024f0ea5871d023c287a5876b3a71089b266d68a54 was only compile-checked
                      (build/p2p-probe-ext.check, sha256 618250f956ca93bbcd317f641f6b3a755d48d34e7654791865d7a9329242890f),
                      the measurement was NOT re-run; the measuring binary above was not rebuilt.
Build:                private nvcc cuda_12.8.r12.8/compiler.35583870_0, -ccbin g++-14 (Ubuntu 14.3.0), -arch=sm_70 -O2
                      (no system packages installed; artifacts only inside this folder)
Phase B launch:       scripts/run-p2p-full-detached.sh -> scripts/run-p2p.sh --max-size 256M
                      probe arguments: --max-size 256M
Time (UTC):           service stopped 2026-10-17T00:12:00Z; probe 00:12:28Z-00:12:34Z; service back up 00:12:34Z,
                      fully checked and served requests at 00:16:23Z (raw/logs/2026-10-17T001623Z/)
Conditions:           Strata stopped; VRAM free on all four cards: 16138 MiB each (after-stop.csv);
                      36-37 °C, SM clock 135 MHz idle -> 1530 MHz during the probe, 38-40 W -> 62-67 W
Units:                GB/s = 1e9 B/s; GiB/s = 2^30 B/s; latency in microseconds. Tables are in GB/s.
Method:               3 warm-up copies, then 5 measurement blocks; for sizes <= 1 MiB every copy is timed
                      separately (cudaEventRecord before/after a single cudaMemcpyPeerAsync), for > 1 MiB a block of
                      max(3, 64 MiB / size) copies. Median/min/max over blocks and the within-row spread are reported.
Status:               measured wherever there is a CSV row to point at; arithmetic and conclusions are labelled
                      separately (calculated, hypothesis, recommendation). The comparison with PR #2525 is §8.
```

---

## 1. What this document covers

| Item | Covered by |
|---|---|
| capability matrix | §2 + `p2p-matrix.csv` (6 pairs, all 4 attributes CUDA 12.8 defines, the price of peer access) |
| the 4 KiB…256 MiB size ladder, latency and bandwidth | §3-§5 + `p2p-benchmark.csv` (211 rows: 17 sizes x 6 pairs x 2 directions, no gaps) |
| several pairs at once | §6 (six pairs simultaneously) |
| correctness of the exchange | §7 (`p2p-correctness.json`, 37 tests, 0 failures) |
| the artifacts `p2p-matrix.csv`, `p2p-benchmark.csv`, `p2p-correctness.json` | files in this folder |
| the open part: the bandwidth of **direct kernel access** to a peer's memory (only `cudaMemcpyPeerAsync` had been measured) | **§13** (taken 2026-10-17T05:13Z, `raw/peer-read/2026-10-17T051351Z/`) - closed without stopping the service, except for pairs involving GPU3 |

Why phase B needed Strata stopped: with the model loaded, GPU3 had 71 MiB of VRAM free, so its pairs could not be measured in phase A (36 rows skipped there, the reason written into each CSV row). In phase B, after the stop, 16138 MiB is free on every card - **no gaps at all**.

## 2. The capability matrix (measured)

All 6 pairs, both directions, `p2p-matrix.csv`:

| pair | A->B / B->A canAccessPeer | AccessSupported | NativeAtomicSupported | PerformanceRank | CudaArrayAccessSupported | price of enabling peer access | status |
|---|---|---|---|---|---|---|---|
| GPU0<->GPU1 | 1 / 1 | 1 | 1 | 0 | 1 | 0 MiB / 0 MiB | OK |
| GPU0<->GPU2 | 1 / 1 | 1 | 1 | 0 | 1 | 0 MiB / 0 MiB | OK |
| GPU0<->GPU3 | 1 / 1 | 1 | 1 | 0 | 1 | 0 MiB / 0 MiB | OK |
| GPU1<->GPU2 | 1 / 1 | 1 | 1 | 0 | 1 | 0 MiB / 0 MiB | OK |
| GPU1<->GPU3 | 1 / 1 | 1 | 1 | 0 | 1 | 0 MiB / 0 MiB | OK |
| GPU2<->GPU3 | 1 / 1 | 1 | 1 | 0 | 1 | 0 MiB / 0 MiB | OK |

`cudaDeviceEnablePeerAccess` returns "no error" in both directions on every pair; free VRAM before and after it is identical (15826 MiB per card at the time of the probe), i.e. **enabling peer access costs not one byte**. Each pair is described by UUID and PCI bus (0000:03:00.0 … 0000:06:00.0), matching the inventory snapshot `raw/topology/2026-10-16T232834Z/`.

A method note: in CUDA 12.8 `enum cudaDevP2PAttr` contains **exactly four** values; there is no `ReadPoolSupported` / `LaunchableKernel` in this toolkit. `CudaArrayAccessSupported = 1` (whether a kernel may directly address an array on another card) is a value PR #2525 did not query; it is queried **for the first time here**, and it is 1 on every pair.

The NVML view of the same path (`nvmlDeviceGetP2PStatus`) was taken in PR #2525: `NVLINK=OK` and `PCI=OK` on all 6 pairs - reused, not repeated.

## 3. Latency of small transfers (measured, `kind=latency`)

Median over all 6 pairs and both directions; the spread between pairs is at most 0.06 us (the fabric is uniform).

| size | latency, us (median) | min/max over pairs | bandwidth, GB/s | % of 38.77 GB/s | "clean" wire time at 38.77 GB/s, us | fixed overhead, us |
|---|---|---|---|---|---|---|
| 4 KiB | 6.85 | 6.82 / 6.88 | 0.60 | 1.5 % | 0.11 | 6.74 |
| 8 KiB | 6.98 | 6.94 / 7.01 | 1.17 | 3.0 % | 0.21 | 6.76 |
| 16 KiB | 6.94 | 6.91 / 6.98 | 2.36 | 6.1 % | 0.42 | 6.52 |
| 32 KiB | 7.34 | 7.30 / 7.39 | 4.46 | 11.5 % | 0.85 | 6.50 |
| 64 KiB | 8.21 | 8.13 / 8.32 | 7.98 | 20.6 % | 1.69 | 6.52 |
| 128 KiB | 9.98 | 9.89 / 10.05 | 13.13 | 33.9 % | 3.38 | 6.60 |
| **256 KiB** | **13.44** | 13.38 / 13.50 | **19.50** | **50.3 %** | 6.76 | 6.68 |
| 512 KiB | 20.19 | 20.13 / 20.26 | 25.97 | 67.0 % | 13.52 | 6.67 |
| 1 MiB | 33.70 | 33.66 / 33.79 | 31.12 | 80.3 % | 27.05 | 6.65 |

**The latency conclusion (confirmed by a 9-point ladder):** one peer copy is described by
`T = 6.7 us + size / 38.8 GB/s`
with a fixed cost of 6.5-6.8 us that **does not depend on size** up to 1 MiB. That is call overhead (launch + path traversal + event synchronisation), not time on the links: the clean wire time for 4 KiB is 0.11 us, sixty times smaller than what is measured.

Practical thresholds (calculated from the measured curve):
- **256 KiB** - 50 % of peak bandwidth;
- **1 MiB** - 80 %;
- **8 MiB** - 96 %;
- **>= 32 MiB** - >= 99 %.

So any peer transfer below about 256 KiB spends more time on overhead than on moving bytes: **NVLink pays off when small exchanges are batched into blocks of >= 1 MiB**, not through many small copies.

## 4. Bandwidth: the full 2 MiB…256 MiB ladder (measured, `kind=bandwidth`)

| size | bandwidth, GB/s (median over all pairs and both directions) | min/max over rows | within-row spread (max-min)/min, % | GiB/s |
|---|---|---|---|---|
| 2 MiB | 32.59 | 32.44 / 32.62 | 0.59 | 30.35 |
| 4 MiB | 35.40 | 35.40 / 35.43 | 0.26 | 32.97 |
| 8 MiB | 37.28 | 37.19 / 37.31 | 0.46 | 34.72 |
| 16 MiB | 38.07 | 38.04 / 38.09 | 0.19 | 35.46 |
| 32 MiB | 38.45 | 38.44 / 38.46 | 0.09 | 35.81 |
| 64 MiB | 38.64 | 38.63 / 38.64 | 0.05 | 35.99 |
| 128 MiB | 38.73 | 38.72 / 38.73 | 0.02 | 36.07 |
| **256 MiB** | **38.77** | 38.77 / 38.77 | 0.01 | 36.11 |

The curve reaches a plateau: 128 MiB to 256 MiB adds 0.1 %. **The ceiling of a single peer copy on this board is 38.77 GB/s.**

## 5. Pair uniformity and direction symmetry (measured)

| pair | 32 MiB GPUa->GPUb | 32 MiB GPUb->GPUa | 256 MiB GPUa->GPUb | 256 MiB GPUb->GPUa | direction difference |
|---|---|---|---|---|---|
| GPU0<->GPU1 | 38.45 | 38.46 | 38.77 | 38.77 | 0.00 % |
| GPU0<->GPU2 | 38.45 | 38.46 | 38.77 | 38.77 | 0.00 % |
| GPU0<->GPU3 | 38.46 | 38.46 | 38.77 | 38.77 | 0.00 % |
| GPU1<->GPU2 | 38.45 | 38.46 | 38.77 | 38.77 | 0.00 % |
| GPU1<->GPU3 | 38.45 | 38.44 | 38.77 | 38.77 | 0.00 % |
| GPU2<->GPU3 | 38.46 | 38.45 | 38.77 | 38.77 | 0.00 % |

All 12 directions at 256 MiB give 38.77 GB/s with a 0.01 % within-row spread and 0.0 % between pairs and directions. **The fabric is uniform: no fast or slow pairs, no directional skew** - an independent confirmation of PR #2525's finding that every pair is wired the same way (2 NV2 links).

For reference (calculated against the rated figures from `nvidia-smi nvlink -s`, 25.781 GB/s per link):
- a pair has 2 links = 51.56 GB/s rated -> measured **75.2 %** of the pair's link capacity;
- a card has 6 links = 154.69 GB/s rated -> measured **25.1 %**.

The gap between the rated link figure and the measured copy is discussed in §9 (hypothesis: the limit is on the device side, not the links).

## 6. Six pairs competing at once (measured, new result)

Method: one buffer per card (256 MiB), all 6 pairs started at the same moment in one direction, 3 blocks x 6 copies; the measurement is the total time from an event on GPU0 with all cards synchronised. The row is `contention,6,-1,all_pairs,268435456,…` in `p2p-benchmark.csv`.

| mode | total bandwidth of all 6 pairs, GB/s | per pair, GB/s | block spread |
|---|---|---|---|
| 6 pairs at once | **77.50** (min 77.48 / max 77.51) | **12.92** | 0.03 % |
| pairs one after another (isolated) | - | **38.77** (median; recorded in the note on the row) | - |
| **ratio** | - | **a 3.00x fall per pair** | - |

What follows (the arithmetic is confirmed by measurement, the interpretation is labelled):

1. With all six pairs running the system moves **77.5 GB/s** card<->card. That is **not** 6 x 38.8 = 232 GB/s: what scales is not the whole fabric but the path to 2 x 38.8.
2. In this pattern every card is the **source** of three copies (0->1, 0->2, 0->3 and so on) and sends 3 x 12.92 = **38.75 GB/s** - exactly the same ceiling as one isolated copy. **The ceiling sits at the device (~38.8 GB/s), not at the pair or the link.** *Refinement (taken 2026-10-17T05:13Z, §13):* this ceiling belongs to the **copy-engine** path; with direct kernel access one card sustains 32 GB/s in each direction (64 GB/s in total). For budgeting patches that use copies, the conclusion stands.
3. Therefore "add one more parallel peer copy" buys nothing: while a card is already sending ~38.8 GB/s, any further copy from that card only divides its bandwidth.

*Hypothesis (not proven on this machine):* the limiter is the GPU-side copy path (copy engines / SMs involved in `cudaMemcpyPeerAsync`), not the NVLink links. Indirect signs: 75 % of a pair's link capacity, and exactly a threefold fall with three competitors per card. There is no direct evidence: this machine has no NVLink byte counters (DCGM 4.x without the profiling module, NVML Sampling & Reporting absent), and `nsys` is not installed.

*Not measured:* opposing (half-duplex) competition A<->B on all pairs at once; mixed NVLink + host-path load; the bandwidth of direct kernel access (which `CudaArrayAccessSupported = 1` makes possible, but only `cudaMemcpyPeerAsync` was measured here) - closed in §13.

## 7. Correctness of the exchange (measured)

`p2p-correctness.json`: **37 tests, 0 failures** (`"tests_failed": 0`). All 6 pairs are covered (6 tests per pair) plus an out-of-memory handling test.

| test | what it checks | result |
|---|---|---|
| `deterministic_pattern_checksum` | fill a deterministic pattern on A, copy to B, two 64-bit sums on B against a host-computed reference | OK on 6/6 pairs |
| `element_by_element_compare` | element-wise comparison on the GPU after the copy | OK, 0 differences |
| `repeat_cycles_10_random_patterns` | 10 cycles with changing patterns, each verified | OK, all cycles |
| `no_partial_write_beyond_copy_length` | copying half a buffer: the first half matches the source, the second is **unchanged** | OK (no write past the copy length) |
| `two_streams_concurrent_copies` | two halves copied by two streams at the same time, whole buffer verified after synchronising both devices | OK |
| `cross_device_event_dependency` | an operation on B depending on an event on A | OK, no deadlock |
| `allocation_failure_handled` | a 64 GiB request -> `cudaErrorMemoryAllocation`, then a 16 MiB allocation succeeds | OK (the error does not break the context) |

Separately: **the absence of errors is not proof of the route taken.** What was verified is the content of the copies and the absence of side writes; what was not verified is which physical path each copy took (NVLink or the PCIe switch) - see §9.

## 8. Comparison with PR #2525 (reused, not re-measured)

| Quantity | PR #2525 | our phase B | Comment |
|---|---|---|---|
| `canAccessPeer`, all pairs | 1/1 | 1/1 | independently reproduced |
| price of enabling peer access | 0 MiB | 0 MiB | independently reproduced |
| card<->card bandwidth, 256 MiB | **38.8 GB/s** (12 repeats) | **38.77 GB/s** (5 blocks x 3 copies x 2 directions) | 0.08 % apart, within the method |
| bandwidth, 32 MiB, model unloaded | 38.4 GB/s | 38.45 GB/s | matches |
| bandwidth, 32 MiB, during a 31375-token prefill | 29.4-29.9 GB/s | not measured (not repeated) | see below |
| host path, one card | H2D 6.29, D2H 6.57-6.59 GB/s | not measured (not repeated) | - |
| host path, four cards at once | 1.56 / 1.65 GB/s per card | not measured (not repeated) | - |
| P2P pair competition | **not measured** | **77.50 GB/s total, 12.92 per pair** | new result |
| latency and the size ladder | **not measured** | 4 KiB…256 MiB | new result |
| correctness of the exchange | **not tested** | 37/37 | new result |
| `CudaArrayAccessSupported` | not queried | 1 on every pair | new result |

**Against the host path (calculated from our data and the PR's):**
- one NVLink pair against one host stream per card (1.56 GB/s with four running at once): **24.9x**;
- the whole fabric against all four host streams (4 x 1.56 = 6.24 GB/s): **12.4x**.
The second number is the one an engine cares about: it is the maximum available gain from moving **all** inter-card traffic from the host path onto NVLink under fully parallel load.

**An important clarification about the PR's 29.4-29.9 GB/s "during prefill".** Our phase A (the same ladder with the model **resident but idle**) gave 38.45 -> 37.28 GB/s at 32-8 MiB, and phase B (model unloaded) gave the same 38.45 -> 37.28 (differences -1.7 %…+0.2 %, inside the method's noise; the comparison table is in the run log). So the fall to 29.4-29.9 GB/s in the PR is caused **not by occupied VRAM** but by the engine actively working (the SMs and DRAM traffic of prefill compete with the copy). *That causal wording is a hypothesis consistent with two independent observations; we did not test it deliberately.*

## 9. Method limits and artifact defects (please read before using the numbers)

1. **A defect of the probe (ours, not the hardware's), fixed in the source after the run:** in the `contention_pair` rows the `gbps_median` column is 0.00 - the `fprintf` argument order was wrong, so the per-pair isolated values were not written into those rows. **The measurement was not re-run** (re-measuring an unchanged factor is not allowed): the isolated 256 MiB values per pair are present in the size-ladder rows (38.77 for every pair and direction), and the isolated overall median (38.77) is in the note on the `contention` row, so the "isolated vs simultaneous" comparison is complete. The fixed code writes each pair's isolated median/min/max; it was compile-checked (`build/p2p-probe-ext.check`) and is what the A/B stand uses.
2. **The latency figure includes call overhead.** Timing a single `cudaMemcpyPeerAsync` on stream 0 with `cudaEventRecord` includes submitting the operation and synchronising the event. 6.5-6.8 us is an **upper bound** on the true NVLink latency, not its value. At small sizes this is the dominant effect, which is why the conclusion is worded as "call overhead" rather than "link latency".
3. **Individual small copies vary a lot:** `spread_pct` in the latency rows reaches 265-4700 % (some 4 KiB copies took 6.7 us, others 24-26 us). All tables use medians; min/max are given separately.
4. **Competition was measured in one pattern only:** one copy per pair, all six pairs in one direction, 256 MiB only. Opposing flows, uneven patterns (for example three pairs in a star around GPU3) and mixing with host traffic were not tested.
5. **The route of any individual copy is not confirmed by counters.** There are no NVLink byte counters on this machine (DCGM 4.7.0 without the profiling module; NVML Sampling & Reporting absent in both 580.178.04 and 595.91.07). Every "this goes over NVLink" statement rests on the impossibility of running that fast over PCIe (38.77 GB/s against a 6.29 GB/s maximum single host stream) and on `nvidia-smi nvlink` reporting 2 links per pair.
6. **Only the `cudaMemcpyPeerAsync` path was measured.** Direct kernel access to a peer's array was not tested as a transport - a separate and potentially different path. Closed in §13.
7. **Phase B conditions:** the engine was stopped, so what was measured is **the ceiling of an idle fabric**. The working configuration (model resident, prefill running) gives different numbers - those are in PR #2525 and were not repeated.

## 10. Conclusions

### Confirmed by measurement (this machine, phase B)

1. P2P and P2P atomics are available on all 6 pairs; direct kernel access to a peer array (`CudaArrayAccessSupported`) is 1 as well; enabling peer access costs 0 MiB.
2. The ceiling of a single peer copy is **38.77 GB/s**, the same on every pair and in both directions (spread 0.0 %); the plateau is reached by 128-256 MiB.
3. The fixed cost of one peer copy is **6.5-6.8 us**, independent of size up to 1 MiB; 50 % of bandwidth at 256 KiB, 80 % at 1 MiB, >= 99 % at >= 32 MiB.
4. With all six pairs running at once the total is **77.50 GB/s**, 12.92 GB/s per pair (3.00x below isolated). Each card carries exactly ~38.8 GB/s of outgoing traffic in that pattern, i.e. **the limit is set at the device, not at the pair or the link**.
5. The exchanges are correct: 37/37 tests, including no write past the copy length, two concurrent copies on two streams, a cross-device event dependency and correct handling of an allocation failure.

### Hypotheses (need a separate check; do not treat as proven)

1. The 38.8 GB/s limiter is on the GPU side (the copy path), not in the NVLink links (indirect signs: 75 % of a pair's link capacity, exactly a threefold fall with three competitors per card).
2. The path "a kernel reads/writes a peer's memory" might be faster than `cudaMemcpyPeerAsync` - measured later, see §13.
3. The bandwidth fall during prefill (29.4-29.9 GB/s in the PR) is explained by SM/DRAM competition rather than by occupied VRAM - consistent with our phases A/B, but not deliberately tested.

### What this implies for a change (recommendation)

- **Budget bandwidth per device (~38.8 GB/s), not per pair.** A change that "turns on NVLink" and starts several peer copies from one card at once will not get 6 x 38.8: the fabric ceiling under a fully parallel pattern is 77.5 GB/s, which is itself 38.8 GB/s per card.
- **Batch.** Because of the fixed 6.7 us, a 4 KiB exchange costs as much time as 260 KiB through the host path at 1.56 GB/s; a 1 MiB peer exchange takes 33.7 us against about 670 us for the same volume through the host path (calculated). Blocks of >= 1 MiB win, frequent small copies do not.
- **But "turn on P2P" does not by itself solve the engine's problem:** the dominant long-prompt flow is host->card (61.9-78.4 GB per card, PR #2525), and NVLink does not carry it. These numbers set **the ceiling of the inter-card part** (77.5 GB/s per system, 38.8 per card) and the price of small exchanges; they do not contradict the PR's finding that forcing the stage helper on produced no measurable speed change.

## 11. What was deliberately not done here (the data already exists in PR #2525)

The full prompt/decode benchmark sweep, the `canAccessPeer` matrix as a standalone check, card<->card and host-path bandwidth, the per-phase PCIe profile, uplink saturation, the A/B of `STRATA_PREFILL_HELP` / `…_FRAC`. Reused without repetition: the NVML matrix `NVLINK=OK/PCI=OK`, the host path 6.29 -> 1.56 GB/s per card, the 0.1.44.1 / 0.1.43 baselines, the per-phase traffic profile.

## 12. How to reproduce

```bash
cd <HOME>/strata185/'Strata+4xV100+NVlink'
./scripts/run-p2p-full-detached.sh          # stops Strata -> probe --max-size 256M -> service back
# or just the probe, if VRAM is free:
./scripts/run-p2p.sh --max-size 256M --note "…"
```

Built with the private `<HOME>/strata185/cuda-12.8/bin/nvcc` and `<HOME>/strata185/gcc185/usr/bin/g++-14`, `-arch=sm_70`; the binary and the results stay inside the project folder.

---

## 13. Direct kernel access to a peer's memory (the open part closed; taken 2026-10-17T05:13Z)

**Why.** `CudaArrayAccessSupported = 1` was confirmed in §2, but transport through direct kernel access had not been measured: only `cudaMemcpyPeerAsync` was. Without this number one cannot choose between P2a (hand-off buffers in VRAM + a peer copy) and P2b (a kernel reading the peer's VRAM) in `patch-notes.md` §4.

**Method.** `probes/peer-read-probe.cu` + `scripts/run-peer-read.sh` (private nvcc 12.8 + g++-14, `-arch=sm_70`). Cards 0-2 (on this machine an external CUDA context cannot attach to GPU3 while the service runs - see `ab-stand.md` §7). A 1 MiB…128 MiB ladder with a x2 step, 5 measurement blocks of `max(3, 64 MiB / size)` operations, 2 warm-up blocks, a separate cold run. The method is deliberately the same as in §3-§5 (a single asynchronous operation is not timed with events). Correctness: a checksum of what was read (the expected value computed on the host) plus a comparison of the buffer contents and of a poisoned zone beyond the range - **263 rows, 0 correctness errors**. SM clocks recorded in parallel: 1530 MHz (13 of 20 samples), 1447-1312 MHz at idle - the runs were not at reduced clocks. Data: `raw/peer-read/2026-10-17T051351Z/` (`peer-read-benchmark.csv`, `peer-read-summary.json`, `peer-read-correctness.json`, `conditions.txt`, `clocks.csv`).

**Result** (median over three pairs; spread between pairs <= 0.3 % at >= 16 MiB; spread between blocks <= 2.2 %, median 0.10 %). GB/s:

| size | `local_read` (own memory) | `peer_read` (kernel reads a peer) | `peer_read_write` (copy inside a kernel) | `peer_write` (kernel writes into a peer) | `memcpy_peer` (control) |
|---|---|---|---|---|---|
| 1 MiB | 11.15 | 10.49 | 31.35 | 31.95 | 28.72 |
| 2 MiB | 11.35 | 9.89 | 33.84 | 34.17 | 32.92 |
| 4 MiB | 18.18 | 13.62 | 35.16 | 35.36 | 35.46 |
| 8 MiB | 35.22 | 20.49 | 35.82 | 35.97 | 36.99 |
| 16 MiB | 67.77 | 27.34 | 36.19 | 36.29 | 37.83 |
| 32 MiB | 126.52 | 32.17 | 36.41 | 36.46 | 38.38 |
| 64 MiB | 222.91 | 34.41 | 36.53 | 36.56 | 38.58 |
| **128 MiB** | **361.74** | **35.64** | **36.60** | **36.62** | **38.68** |

Opposing load on one pair (simultaneous, start synchronised by an event):

| pattern | direction 1 | direction 2 | pair total |
|---|---|---|---|
| `peer_read_pair` - a kernel on A reads B **and** a kernel on B reads A | 31.99 | 31.99 | **63.98** |
| `peer_read_vs_memcpy` - a kernel on A reads B **and** a copy A->B | 23.89 (read) | 26.94 (copy) | **50.83** |

**Conclusions (measured).**

1. **Direct kernel access to a peer's VRAM nearly matches a copy:** at >= 32 MiB `peer_read` = 35.64 GB/s against 38.68 GB/s for `cudaMemcpyPeerAsync` (**0.92**), a copy performed inside a kernel = 36.60 (**0.95**), writing into a peer = 36.62 (**0.95**).
2. **A control on the conditions:** `memcpy_peer` = 38.68 GB/s reproduces the plateau of §4 (38.77, delta 0.2 %) - the conditions are comparable with the earlier phase.
3. **The size thresholds for kernel access are noticeably higher than for copies:** 50 % of the plateau around 8 MiB, 90 % at 32 MiB, >= 97 % at 64 MiB (for copies: 50 % at 256 KiB, 80 % at 1 MiB, §3). The engine's hand-off payload is 335.5 MiB per chunk per boundary (`patch-notes.md` §3.2), an order of magnitude above the threshold.
4. **A refinement of §6:** the ~38.8 GB/s per-device ceiling belongs to the **copy-engine** path, not to the device as such or to NVLink in general. Under kernel traffic one card sustains 32 GB/s in each direction = **64 GB/s in total**, with each direction falling only 10 % (35.64 -> 31.99): the fabric works half-duplex across directions without loss. *Boundary:* one pair in two directions was measured; simultaneous kernel load on several pairs was not.
5. **Mixing kernel and copy-engine traffic on one pair is a bad idea:** 23.89 + 26.94 = 50.8 GB/s against 64.0 GB/s for two kernel directions. A change that uses kernel peer reads must not compete with copies on the same link.
6. **Cold access (first touch of a freshly allocated peer buffer) is expensive in relative terms only at small sizes:** +14 % at 1 MiB, +1.8 % at 8 MiB, +1.1 % at 128 MiB (for copies: +44 % / +4.7 % / +0.2 %). There is no separate page-table cost at the useful size.
7. **The kernel is not the limiter:** the same kernel on its own memory gives 361.7 GB/s at 128 MiB. The low `local_read`/`peer_read` values at <= 4 MiB are a consequence of the reduction kernel's occupancy, not a property of the channel. (Note: 361.7 GB/s is above V100 HBM2's 300 GB/s, so this column is used only as "the kernel is not the limiter" and never as a result.)

**What this means for choosing the variant.** The P2b transport (a kernel reading a peer's VRAM) is usable: 92-95 % of copy bandwidth at the relevant sizes. Together with the byte arithmetic (`patch-notes.md` §3.3: P2a is net 0, P2b is net -16.1 GB per request) this settles the choice in favour of **P2b**. Status: *the transport is measured*; the end-to-end effect of such a change is still not measured (no change was applied).

**Boundaries of this section.** Pairs involving GPU3 were not measured (with the service running, GPU3 has ~79 MiB free) - for copies all six pairs matched within 0.0 % (§5), and the topology's symmetry is confirmed by NVML (every pair NV2). The run was made with the service running (895/545/955 MiB free on GPU0-2), so the top of the ladder is 128 MiB rather than 256 MiB; the plateau is reached by 128 MiB (64 -> 128 MiB adds +0.7 %).
