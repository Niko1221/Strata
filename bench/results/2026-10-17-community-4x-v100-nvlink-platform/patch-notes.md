# Notes on the code: what the measurements say about NVLink/P2P in Strata 0.1.44.1 on 4x V100-SXM2 + NVLink 2.0

```
Artifact:            patch-notes.md (this file). Nothing was changed: the tree was not modified, the measuring
                     binary (sha256 bdfec9f5…) and the working config (sha256 f539e51b…) were never touched.
Date (UTC):          2026-10-17
Method:              reading the sources of the working build (v0.1.44.1 = 296b59d2f73c8f04a805df29c2fa97655adace78)
                     + arithmetic on numbers ALREADY MEASURED on this machine. No new measurements, the service was not stopped.
Bases:               all from this machine unless marked "PR":
  • p2p-results.md            - 38.77 GB/s, 77.50 GB/s over 6 pairs, 38.8 GB/s per device, 6.7 us fixed cost,
                                thresholds 256 KiB / 1 MiB / 32 MiB, 37/37 correctness tests, peer access costs 0 MiB;
  • baseline-profile.md       - the baseline line (prompt tok/s, TTFT, VRAM, PCIe by phase); §8 is the attribution experiment;
  • sm70-compatibility.md     - SM70 gates, the inventory of 2084 kernels, the map of routing points;
  • correctness-report.md     - the acceptance criteria (bit determinism, 1e-4·K tolerances, hit rate as an indicator);
  • PR #2525 nvlink-platform-dossier.md - the per-phase PCIe profile (RX 61.9-78.4 GB per card, TX 6.9-11.9),
                                uplink saturation (median 5.25, p95 7.67, worst second 9.33 GB/s against 7.88 nominal),
                                the A/B of STRATA_PREFILL_HELP (no speed change), the host path 6.29 -> 1.56 GB/s per card;
  • the working engine log and raw/restart-attribution/2026-10-17T015058Z/ - the split-search lines, the engine's
                                counters ("… MB of expert reads"), the cache-initialisation and slot-borrowing lines.
Status labels:       measured (this machine) · calculated (arithmetic on measured numbers) ·
                     hypothesis (not tested) · proposal (a suggested change).
```

---

## 0. The short answer

**The point of failure is confirmed, and it is wider than "one line with `p2p = false`".** In 0.1.44.1 a layer-split machine performs no GPU->GPU transfer at all, and at the same time **the dominant long-prompt flow (host->card, 269 GB per request) cannot be moved onto NVLink under any condition on this hardware** - because no card holds copies of the experts another card needs (VRAM is filled exactly, §2.4). So these notes split into two parts: what genuinely should become GPU->GPU (a small but mandatory volume), and what would have to change for long prompts to get faster (that is not the transport, it is the placement decisions).

| # | Element | Level | File / function | Expected effect on THIS machine | Priority |
|---|---|---|---|---|---|
| **P5** | The `--layer-split auto` search criterion: does not see the cost of the LM head, of the prefill pipeline, or of residency on an individual card | C (the primary object) | `src/program/generate.cpp:3832, 3907-3946, 3957-3968, 4131-4199` | **Replicates the measured +4.5 % at 31 307 and +7.3 % at 125 011 without pinning anything by hand** (confirmed by the experiment in baseline-profile.md §8.2) | **1** |
| **P5.5** | `stage_trim` is not applied to CUDA0 under `auto` | A | `src/program/generate.cpp:3082` | Measured: +2.3 GiB of free VRAM on CUDA0 and +512 cache slots with an explicit split; -> fewer borrowed slots -> less expert streaming. Upper bound -19 % TTFT at 125k (calculated, §6) | **2** |
| **P2** | Prompt hand-off between stages -> GPU->GPU | B | `src/prefill/prefill.cpp:4161` (D2H), `:2189` (H2D), `:1106` (allocating host buffers) | -32.2 GB of PCIe per request at 125k; **but** buffers in VRAM cost exactly the same number of stream bytes (§3.3) -> variant P2b: **net -16.1 GB, up to +8 % prompt tok/s at 125k**; variant P2a: **net 0** | 3 |
| **P1** | Enabling peer access for the stage set (today: only for `--peer-device`) | A | `src/core/peer_experts.cpp:97-110` -> called from the split setup | A mandatory precondition of P2/P3. Costs 0 MiB (measured) | 3 (with P2) |
| **P3** | The stage-helper route: `p2p = false` -> selectable | B | `src/prefill/prefill.cpp:1591` | **0 % on this machine** (the A/B of `STRATA_PREFILL_HELP` changed nothing, PR §8.3). Value: correctness and testability of that path on machines where the hand-off saturates the link | 4 |
| **P4** | Allow `--peer-device` / the remote cache together with `--layer-split`; peer copy in `copy_cached_blobs` | A/B | `src/program/generate.cpp:2716-2720`; `src/core/remote_experts.cpp:349-375` | **0 % on this machine** (no free VRAM on any card, §2.4). Value: heterogeneous rigs | 5 |
| — | Decode/verify hand-off (mapped host) | — | `src/program/generate.cpp:7829`, `include/strata/core/verify.hpp:177` | **Do not touch:** 307 KB per token is <= 0.5 % of ITL (calculated, §2.3) | — |

**The one number for judging any change (calculated from two measured configurations, §3.4):** on this machine, at 125 011 tokens, reducing host traffic by 13 967 MB gave -3.59 s of TTFT -> **the marginal price of a host byte is about 3.9 GB/s**, i.e. **1 GiB of traffic is about 0.27 s of TTFT, about 0.5 % at 125k**. Any proposal to "speed up prefill" should be re-priced through this coefficient, with the regime boundary stated explicitly (§6.3).

## 1. Method and its limits

1. Nothing was measured again. All numbers come from the artifacts of the earlier stages and from PR #2525; each source is named in the line.
2. The code reading was done on the tree of the working build, tied to the binary sha256 `bdfec9f5…` (the git SHA of tag `v0.1.44.1` is `296b59d2…`; the tree itself has no git - see §10).
3. Method boundary: **there are no NVLink byte counters on this machine** (DCGM 4.7.0 without the profiling module, NVML Sampling & Reporting absent in both 580.178.04 and 595.91.07). Every statement about routing rests on (a) the code, (b) measured volumes matching `rxpci`/`txpci`, (c) the engine's own counters ("… MB of expert reads"). (b) and (c) agree for two independent sessions - §2.1, §3.1; that is strong but still indirect evidence.
4. Direct kernel access to a peer's memory (`CudaArrayAccessSupported = 1`) **as a means of transport was not measured** when these notes were first written; it was measured later (2026-10-17T05:13Z) and is in `p2p-results.md` §13, because variant P2b depends on it.

## 2. The map of inter-card paths in 0.1.44.1

Model geometry from the pack (confirmed: `packs/iq3_s/index.txt`, `native_experts.txt`, `src/prefill/prefill.cpp:118`): `n_embd N = 2560`, `hyper_connection HC = 4`, residual width `D = N·HC = 10 240`, `n_expert = 512`, `experts/token K = 10`, 48 layers, an expert blob of about 2.05 MB (at most 2.176 MB), 24 576 pairs in total, `experts.bin` = 50.29 GB. Service configuration: `--layer-split auto`, prompt chunk `auto: 8192 tokens` (log line 33858), context 262 144, KV int8.

| Flow | Code | Route now | Volume per 125 011-token request | Could it become GPU->GPU? |
|---|---|---|---|---|
| **A. Prompt hand-off between stages** (mandatory, part of the pipeline) | `prefill.cpp:4161` D2H into `m.hand[]`; `:2189` H2D at the next stage; buffers are `cudaHostAlloc(…Portable)` at `:1106`, 2 per boundary | card -> **host** -> card (2 PCIe crossings per boundary) | **32.2 GB** (calculated: 16 chunks x 335.5 MB x 3 boundaries x 2 directions) | **Yes** - this is exactly what P2 is about |
| **B. Stage helper** (`STRATA_PREFILL_HELP`, `PeerPrefill`) | `prefill.cpp:1591` `pp->p2p = false`; the `p2p = true` branches already exist: `:3508, 3513, 3672` | card -> host -> card | 0 in the working config (the variable is unset); with it forced on, +15…20 GB per card (PR §8.3) | Yes, but useless on this machine (A/B: 0 %) -> P3 |
| **C. Decode/verify hand-off** | `generate.cpp:7829` `cudaHostAlloc(Mapped\|Portable)`, `verify.hpp:177 handoff_floats = hc·N + N + hc = 12 804 float` | the kernel writes/reads mapped host (2 PCIe crossings per boundary) | **0.08 GB per 256 tokens** (calculated: 3 x 2 x 51 216 B x 256) | Technically yes, practically no reason (§2.3) |
| **D. Expert streaming in the prompt path** (dominant) | `prefill.cpp:2394-2417` (a list per chunk), `:2432-2452` `cudaMemcpyAsync(…, HostToDevice)`; threshold `stream_all_min() = 1024` (`:164`) | host page cache -> **card** (PCIe) | **269 472 MB** (measured: the engine's counter, log line 33755) | **Not on this machine** (§2.4, §3.5) |
| **E. Peer tier / remote expert cache** | `peer_experts.cpp:97-110` (P2P exists), `remote_experts.cpp:349-375` (D2H + H2D only) | `--peer-device` is **refused** with `--layer-split` (`generate.cpp:2716-2720`); the remote cache goes through host | 0 (not in use) | Yes (P4), but there is nothing to pay with here |

### 2.1 Flow D: why it is expert streaming and not activation staging (closing the open question in PR #2525 §9.1)

The PR asked: are the 61.9-78.4 GB per card the expert ring or activation staging? "The code could answer this in one line." The answer from code and counters:

1. Code: for `T >= 1024` the prompt path builds, **for every chunk**, the list "all non-resident experts of all the stage's layers" (`prefill.cpp:2394`, skipping only those with `m.host_res[…] >= 0`, i.e. resident in the cache) and loads them into the ring `m.stage_dev[issued % ring]` (`:2432-2452`). There is no check "is this expert already in a slot from the previous chunk".
2. The engine's counter for a 125 011-token request: `file tier I/O this request: … for 269471.8 MB of expert reads` (269.5 GB) with `the OS read 4.3 MB from storage (0 major faults)` - so 269.5 GB came from the page cache and were pushed to the devices.
3. An independent measurement from the PR: a total `RX 270.5 GB` across the four cards for the same shape of request (125 351 tokens) and `TX 35.2 GB`.
4. A hand check of the volume: with an 8192-token chunk and K=10 per stage layer, essentially all 512 experts are involved; what makes them non-resident is **the borrowing of cache slots for the prompt path's buffers** - the log says `the prompt path borrows 1954 CUDA0 cache slots (3.41 GiB)` (and 1631-1841 slots on CUDA1-3). The CUDA0 cache is 5632 slots = exactly 11x512, so after borrowing exactly ~1954 experts are non-resident -> 1954 x 2.05 MB = about 4.0 GB per chunk x 16 chunks = about 64 GB per card. Measured: 61.9-78.4 GB per card, and 269.5/16/4 = 4.21 GB per card per chunk. The agreement is within the uncertainty (blob sizes differ by layer).

**Conclusion (confirmed by three independent sources agreeing):** the dominant flow is expert streaming; activation staging (flow A) is 32.2 GB of about 305 GB, i.e. 10.5 % of the request's traffic and **almost all of the measured TX** (32.2 GB calculated against 35.2 GB measured - a 9 % match).

### 2.2 Why it repeats on every chunk (and why "a ring for the whole prompt" does not fix it)

The 3.41 GiB borrowed are the prompt path's buffers, not a cache for experts. There is nowhere to keep 1954 experts (4.0 GB) in VRAM between chunks: a stage occupies 11-13 layers x 512 experts x 2.05 MB = 11.5-13.4 GB, and its cache is filled exactly (5632/5632, 6656/6656, 6144/6144, 5469/6144 in the initialisation log). A larger chunk reduces the number of stream repeats but needs buffers proportional to `T` - the auto-scan (`prompt chunk auto: 8192`) already considers that trade-off, and a change should not duplicate it (the hypothesis that a better optimum exists was not tested).

### 2.3 The decode hand-off is not worth touching (calculated)

`handoff_floats = 12 804 float = 51 216 B` per window row; per token, 3 boundaries x (write + read) = 307 KB. At an ITL of 9.12-10.40 ms (baseline-profile.md §3) that is 30.7 KB/ms = **<= 0.5 % of ITL** even on the worst host path. Nothing is gained here, and the FP operation order would be put at risk for nothing.

### 2.4 Why flow D cannot go over NVLink on this machine

For NVLink to carry an expert, **another** card must be holding it. Under a layer split a stage may only hold the rows for its own layers (`generate.cpp:7300`), and all four caches are filled exactly to the VRAM limit:

| Config | CUDA0 | CUDA1 | CUDA2 | CUDA3 |
|---|---|---|---|---|
| `auto` (K=11,24,36) | 5632/5632 (100 %) | 6656/6656 (100 %) | 6144/6144 (100 %) | **5469/6144 (89.0 %)** |
| `12,25,37` | 6144/6144 | 6656/6656 | 6144/6144 | **5443/5632 (96.6 %)** |

There are no free slots in either configuration, and 4 x 16 GB of VRAM cannot duplicate 50.3 GB of experts. **Therefore NVLink cannot reduce flow D on this rig under any variant of a change.** That is the formal answer to "which path should become GPU->GPU": A (the mandatory hand-off) and B (the helper, optionally) - but not D.

## 3. The traffic budget of a long prompt

### 3.1 The measured base

| Quantity | Value | Source |
|---|---|---|
| Host->card, all four cards, 125 011-token request | 269 472 MB (`auto`), 255 505 MB (`12,25,37`) | the engine's counter, log line 33755; `raw/restart-attribution/…/engine-split12.log` |
| PCIe RX/TX per card, 125 351-token request | RX 61.9-78.4 GB per card, TX 6.9-11.9 GB per card; totals RX 270.5 GB, TX 35.2 GB | PR §3, §8.3 |
| The Gen3 x8 uplink (7.88 GB/s per direction) | median 5.25, p95 7.67, worst second 9.33 GB/s; 33 of 75 s at or above 6 GB/s | PR §8.2 |
| The host path | 6.29 GB/s per card alone, **1.56 GB/s per card with four at once** | PR |
| P2P | 38.77 GB/s isolated; 77.50 GB/s over 6 pairs; **38.8 GB/s per device**; a fixed copy cost of 6.7 us; >= 1 MiB -> 80 % of bandwidth, >= 32 MiB -> >= 99 % | `p2p-results.md` §3-§6 |
| **Direct kernel access to a peer's VRAM** (taken 2026-10-17T05:13Z) | at >= 32 MiB: a kernel reading a peer **35.64 GB/s** (0.92 of a copy), a copy inside a kernel **36.60** (0.95), writing into a peer **36.62** (0.95); thresholds: 50 % of the plateau ~8 MiB, 90 % at 32 MiB; opposing kernel reads on one pair **64.0 GB/s** (-10 % per direction); kernel traffic plus a copy on the same pair only 50.8 GB/s; cold access +1.1 % at 128 MiB | `p2p-results.md` §13; `raw/peer-read/2026-10-17T051351Z/` |
| TTFT / prompt tok/s | 52.69 s / 2372.4 (`auto`), 49.10 s / 2546.2 (`12,25,37`) | `baseline-profile.md` §8.2 |
| Method noise | prompt tok/s +-0.19…0.29 % (1.31 % at 4k) | `baseline-profile.md` §2 |

### 3.2 Calculated volumes per 125 011-token request (chunk 8192 -> 16 chunks)

| Flow | Formula | Volume |
|---|---|---|
| A. hand-off between stages | 8192 x 10 240 x 4 B = 335.5 MB per chunk per boundary; x 16 chunks x 3 boundaries x 2 directions | **32.2 GB** (16.1 D2H + 16.1 H2D) |
| B. the helper with it forced on | the PR's measurements | +15…20 GB per card |
| C. decode hand-off | 3 x 2 x 51 216 B x number of tokens | 0.08 GB per 256 tokens |
| D. expert streaming | the engine's counter | **269.5 GB** |
| Total over host | | about 302 GB (matching the measured 270.5 RX + 35.2 TX = 305.7) |

### 3.3 The key relation: VRAM and host bytes (calculated, structural)

Every cache slot (about 2.05 MB) given to something else becomes a non-resident expert and **is loaded again on every chunk**. So for a prompt of length `T`:

> **1 MiB taken from the expert cache = (T / 8192) MiB of extra host traffic.**
> at 125k, x16; at 62k, x8; at 31k, x4; at 16k, x2; at 4k, x1.

Consequences that any decision has to respect:

* Moving the hand-off buffers into VRAM (2 x 335.5 MB = 671 MB on the receiving card) costs 671 x 16 = **10.7 GB of extra stream per card** and saves exactly 10.7 GB of hand-off per card. **This is an identity, not a coincidence** (2 buffers x 16 chunks = 2 directions x 16 chunks): variant P2a on a machine with a full cache gives **net 0**.
* Variant P2b (a kernel reading the peer's VRAM, with no receiving buffer) needs only a second `m.R` on the source card (335.5 MB) -> +5.4 GB of stream per card against -10.7 GB of hand-off -> **net -5.4 GB per card, -21.4 GB for the system**.
* P5.5 (trimming CUDA0's weights under `auto`) works the other way: it **frees** VRAM. 2.3 GiB free (measured: 11.08 -> 13.38 GiB) means about 1200 slots need not be borrowed -> -2.3 GiB of stream per chunk x 16 = **-36.8 GiB (about -39.5 GB) per request**.

### 3.4 The marginal price of a host byte (calculated from one pair of configurations)

The difference between `auto` and `12,25,37`: host traffic -13 967 MB, TTFT -3.59 s -> **about 3.9 GB/s**, i.e. **1 GiB is about 0.27 s of TTFT, about 0.5 % at 125k**. This is an empirical constant from one pair of points in a narrow +-5 % neighbourhood; for reductions of tens of percent it overstates the effect (the engine then runs into compute, §6.3).

---

## 4. The elements

Format: (1) file and function; (2) current behaviour; (3) the confirmed problem; (4) the proposed change; (5) expected benefit; (6) risks; (7) correctness test; (8) the benchmark that would prove the benefit; (9) when not to enable it.

### P1. Level A. Peer access for the layer-split stage set

1. **Where:** `src/core/peer_experts.cpp:97-110` (already has `cudaDeviceCanAccessPeer` x2, `STRATA_PEER_NO_P2P`, `cudaDeviceEnablePeerAccess` swallowing `cudaErrorPeerAccessAlreadyEnabled`) - call it from the stage setup: `src/program/generate.cpp:2346-2470` (the layer-split block) / `:7822-7840` (allocating the hand-offs).
2. **Now:** peer access is enabled only for the pair "CUDA0 <-> `--peer-device`". Under a layer split there is none at all, so `cudaMemcpyPeerAsync` between stages would return `cudaErrorInvalidDevice`.
3. **Problem (measured):** all 6 pairs are P2P-capable (`p2p-matrix.csv`: `canAccessPeer 1/1`, `AccessSupported 1`, `NativeAtomicSupported 1`, `CudaArrayAccessSupported 1`), enabling it costs **0 MiB** (measured twice), and the engine does not use any of it on a layer-split machine.
4. **Change:** one helper `enable_stage_peer_access(std::vector<int> devs, std::string& err)` that enables access across all **directed** pairs of the pipeline (0->1, 1->2, 2->3 and the reverse - needed for events and for P2b), stores the result in a `peer_ok[i][j]` table and prints it to the log one line per pair (as `peer_experts` already does). A `STRATA_STAGE_PEER_NO_P2P=1` switch in the existing style.
5. **Benefit:** the precondition for P2/P3; by itself 0 %.
6. **Risks:** (a) `cudaDeviceEnablePeerAccess` requires the current device - the call order is already established in `peer_experts.cpp`; (b) on some platforms enabling P2P changes the allocator - measured 0 MiB here, but other machines need a VRAM check before/after; (c) 12 directed enablements at start-up - fractions of a millisecond.
7. **Test:** after start, `cudaDeviceCanAccessPeer`/`GetP2PAttribute` from the process itself, printed to the log; plus the existing cross-card kernel-read check (`build/sm70-tests`, 6/6 pairs).
8. **Benchmark:** not needed (the transport is not enabled).
9. **Do not enable if:** any pair has `AccessSupported = 0`; `STRATA_PEER_NO_P2P`/`STRATA_STAGE_PEER_NO_P2P` is set; the pair's CC differs and the build was not built for the lower one; the user asked for no P2P.

### P2. Level B. Prompt hand-off between stages -> GPU->GPU (the main subject of level B)

1. **Where:** `src/prefill/prefill.cpp:4161` (`cudaMemcpyAsync(h, m.R, T*D*4, DeviceToHost)` + `cudaStreamSynchronize(m.cs)`), `:2189` (`cudaMemcpyAsync(m.R, hand_in_ + c0*D, …, HostToDevice)`), buffer allocation at `:1106` (`cudaHostAlloc(chunk*D*4, Portable)`, 2 per boundary, `hand_buf_ ^= 1`); the control flow is the `on_stage_chunk` callback and `std::async` at `:4173`.
2. **Now:** every stage boundary is two PCIe crossings (D2H at the predecessor, H2D at the successor) plus a **full synchronisation of the compute stream** at `:4161` before the slot is handed over.
3. **Problem (measured):** 32.2 GB of PCIe per request at 125k = 10.5 % of the request's total traffic and about 92 % of the measured TX (35.2 GB). The synchronisation at every boundary (48 per request) breaks the chunk pipeline.
4. **Change - two variants behind one API boundary (a `set_stage_helper`-style route choice):**
   * **P2a - a peer copy.** Allocate `hand[]` with `cudaMalloc` on the successor's device, transfer with `cudaMemcpyPeerAsync(dst_dev, …, src_dev, …, s_cp)`, and replace the synchronisation at `:4161` with `cudaEventRecord` + `cudaStreamWaitEvent` at the successor. The `p2p = true` branches in `PeerPrefill` (`:3508, 3513, 3672`) show the idiom; `p2p` already defaults to `true` (`:647`).
   * **P2b - a kernel reading the peer's VRAM (the preferred one).** The predecessor's `m.R` stays in its own VRAM; the successor reads it directly (`CudaArrayAccessSupported = 1` confirmed on every pair). This needs a second `m.R` on the source (a double buffer for the residual) and replacing the host synchronisation with events.
5. **Expected benefit (arithmetic only, §3.3, §3.4):**
   * P2a: -32.2 GB of hand-off, **+32.2 GB of expert stream** (671 MB x 16 chunks) -> **net 0**. On a machine with a full cache the variant is pointless.
   * P2b: -32.2 GB of hand-off, +16.1 GB of stream (335.5 MB x 16 chunks on the three source cards) -> **net -16.1 GB -> -4.1 s of TTFT at 125k (-7.8 % TTFT, about +8.4 % prompt tok/s) - an upper bound**. At 31k the volumes are 4x smaller -> **-1…-2 %**. Plus removing 48 host synchronisations per request (not measured, hypothesis).
   * Below 8k tokens the effect of both variants is under 1 % and indistinguishable from noise.
6. **Risks:** (a) P2a can make things **worse** on a machine with a full cache - that is exactly the risk calculated in §3.3; (b) **closed 2026-10-17T05:13Z:** the bandwidth of a direct kernel read was measured - 35.64 GB/s at >= 32 MiB (0.92 of `cudaMemcpyPeerAsync`), a copy inside a kernel 36.60 (0.95), the 90 % plateau threshold is 32 MiB, an order of magnitude below the hand-off payload (335.5 MiB). The P2b transport is usable; the end-to-end effect is still not measured (`p2p-results.md` §13). Measured there as well: opposing kernel reads give 64.0 GB/s per pair, but mixing kernel traffic with copy-engine traffic on one link is a bad idea (50.8 GB/s) - a P2b implementation must not run parallel copies on the same pairs; (c) changing the synchronisation order must not change the bits (§8, bit acceptance); (d) buffer lifetime: the slot must live until the successor's event, not until the host function returns.
7. **Correctness test:** bit-identical `text_sha` on stable prompts (long context, fact extraction) after warm-up - a transport has no right to change bits; the `cross_device_event_dependency` test and "two concurrent copies on two streams" as the lower level; a separate check of `no_partial_write_beyond_copy_length` for the residual buffer.
8. **Benchmark:** 32k and 125k, 3 repeats, medians, with EOS banned; a mandatory control "same binary, same config"; recording `MB of expert reads` per request (the counter in the serve log) - that is what separates "less PCIe" from "more streaming".
9. **Do not enable if:** (checked at start) the receiving/source card does not have `2 x chunk x D x 4` bytes beyond what the cache already reserved - on this rig that holds only for P2b, and only if the second `m.R` does not come out of the cache; there is no P2P path between the stages; `--pipeline-windows` (its own set of hand-offs at `:7891`) was not adapted.

### P3. Level B. The stage-helper route: `p2p = false` -> selectable

1. **Where:** `src/prefill/prefill.cpp:1591` (`pp->p2p = false; // the mapped host route, whether or not the cards could reach each other (measured there)`).
2. **Now:** a hard constant; the comment says the measurement behind it was made on 2x RTX 3090 (PCIe 4.0 x16, no P2P).
3. **Problem (measured):** on a machine with P2P at 38.8 GB/s against 1.56 GB/s per card, the route is chosen from someone else's measurements. Turning the helper on here **changed speed at no length** (PR §8.3, six sessions, spread inside the +-0.9 % control).
4. **Change:** `pp->p2p = peer_pair_ok(m.device, h.device) && !env_off;` - one line plus a `STRATA_PREFILL_HELP_P2P=0/1` switch (default "auto": on when the pair supports P2P and the hand-off is >= 1 MiB).
5. **Benefit on this machine: 0 %** (measured). Elsewhere: where the hand-off saturates the link, the difference is 25x in bandwidth and 6.7 us of fixed cost instead of two PCIe crossings.
6. **Risks:** minimal (the `p2p = true` branches already exist and are used by `--peer-device`).
7. **Test:** the 37/37 `p2p-probe-ext` tests + an integration test "the same text with `…_HELP_P2P=0` and `=1`".
8. **Benchmark:** an A/B of `STRATA_PREFILL_HELP=1` with `…_HELP_P2P=0/1` at 32k/125k - here 0 +- 0.3 % is expected; that is **a check of a null result**, not proof of benefit.
9. **Do not enable if:** the pair has no P2P; the helper is not armed (`STRATA_PREFILL_HELP` unset in the working config); the hand-off is smaller than 1 MiB (80 % of bandwidth only from 1 MiB, §3.1).

### P4. Level A/B. `--peer-device` and the remote cache together with a layer split

1. **Where:** `src/program/generate.cpp:2716-2720` (the parser's refusal: `--peer-device cannot be combined with --layer-split (use one or the other)` - confirmed by running it in the PR, `interconnect-experiments/peer-device-refusal.txt`); `src/core/remote_experts.cpp:349-375` (`copy_cached_blobs` - `cudaMemcpyAsync(…, DeviceToHost)` only).
2. **Now:** the combination is forbidden; the remote cache reads a peer's VRAM through host.
3. **Problem (measured):** on a machine that has to be split by layers, the only mechanism for "a neighbour holds the expert" is unavailable.
4. **Change:** (a) drop the refusal when the peer device is not in the stage set, **or** is in it and has free VRAM by the measured `stage_left`; (b) in `copy_cached_blobs`, use `cudaMemcpyPeerAsync` when `peer_ok`, keeping the existing fallback and its error handling (it already resets `hit[]` correctly on failure).
5. **Benefit on this machine: 0 %** - there is no free VRAM (§2.4), so a peer tier could not hold a single extra expert. On heterogeneous rigs (32+16 GB): a cache miss served at 38.8 GB/s instead of 1.56…6.29 GB/s.
6. **Risks:** two "second instances" of the engine on one card (exactly what the code comment at `:2714-2715` guards against); doubled VRAM on the peer.
7. **Test:** the existing `--peer-device` path (it already works with P2P) + a check that `peer->has(l,e)` (`prefill.cpp:2396`) does not change the answer's bits.
8. **Benchmark:** only on a machine with free VRAM; here it should not be run (a measurement would prove nothing).
9. **Do not enable if:** the stages' total free VRAM is below a reserve; the peer falls into the same layer range as a stage.

### P5. Level C (the primary object). The `--layer-split auto` search criterion

This is the only item for which **a measured effect already exists**: pinning `12,25,37` instead of what `auto` chose gave **+4.48 % prompt tok/s at 31 307 and +7.33 % at 125 011**, TTFT -4.2 % and -6.8 % (baseline-profile.md §8.2). Pinning it by hand is not something to ship - so the search itself is what would have to get better.

**The current model (code):** `generate.cpp:4183-4185`

```
ms = miss_ms · (1 − held_mass/total_mass) + Σ_i (layers_i · layer_ms[i]),
layer_ms[i] = 0.33 · (84.0 · 2.617) / (SM_i · GHz_i)      (:3832)
miss_ms = 190 ms, scaled by the measured link_scale[i]      (:3856, :3941)
held_mass — increments of the curve H(f) = 1 − (1 − f)^3    (:3877-3893)
enumerating all C(45,3) = 15 180 layouts                    (:4131-4199)
on a tie — the layout with the smallest max_i(layers_i · layer_ms[i]) wins  (:3957-3968)
```

**The defects and what to change.**

| # | Defect | Evidence | Change (proposal) |
|---|---|---|---|
| **P5.1** | The cost of the LM head / output projections of the last stage is in neither `layer_ms` nor `stage_max` | 30 search lines in the working log: **4 different layouts at the same score** - `K=12,25,37` x20, `11,24,36` x5, `11,24,37` x4, `11,25,37` x1, all "predicted 28.4 ms" (one 28.5). The choice moves between runs of one binary and config; the measured gain from moving one layer **off** the stage that has the head is +4.5…+7.3 % | Add a head term to `layer_ms`/`stage_max` (`output.weight` 2560x248320 + `output_hc_*`), estimated from the same SMxGHz; then the tie-break stops overloading the last stage by itself |
| **P5.2** | The cost model is a decode window only; prefill is a pipeline of chunks (`prefill.cpp:4161-4177`) whose speed is the slowest stage, not a sum | `predicted 28.4 ms` = 48 x 0.59 ms - a sum of layers, identical for every layout; the `stage_max` tie-break does not see the head (P5.1) or the stream (P5.3) | Add a second objective `prefill_ms = max_i(layers_i · layer_ms[i] + head_i + stream_i)` and choose on the tuple `(decode_ms, prefill_ms)` with an explicit precedence rule; print both scores in the `layer split auto:` line |
| **P5.3** | The cost of expert streaming (dominant here) is not in the estimate at all | `stage_left(i,lb,le)` (`:3820`) subtracts the session and the weights but not the prompt-path buffers; borrowing 1954 slots -> 4.0 GB of stream per chunk (confirmed by the counter) | `stream_i = (n_pairs(range_i) − slots_i + borrowed_bytes_i/blob) · blob · n_chunks(T_ref) / link_gbps_i`, where `n_chunks(T_ref)` is the number of chunks of a reference length (say 32k). Every quantity is already in the search |
| **P5.4** | The coverage curve saturates: `H(f)=1−(1−f)³` reports "~100.0 % of the routed mass" where CUDA3 actually holds 89.0 % of its pairs; the profile file carries a ranking without frequencies (the code says so: `tools/make_profile.py` has them and drops them) | Log: `the caches hold 24263 of 24576 profiled pairs (~100.0% of the routed mass)` alongside `5469 of its 6144 profiled pairs`; the authors themselves note the regime where the model is wrong ("below f ~ 0.5 the two curves diverge") | Carry counters into the profile (a format with frequencies) and compute `held_mass` from them; until then, do not use `held_mass` as the only discriminator, and add P5.3 and P5.5 |
| **P5.5** | `stage_trim = trim_asked && multi_gpu && !split_auto && !split_at.empty()` - under `auto` CUDA0 is never trimmed, even though `STRATA_STAGE_TRIM=1` is set in the config | Measured: CUDA0 free VRAM **11.08 GiB under `auto` against 13.38 GiB with an explicit split**; GPU0 VRAM 15246 against 13744 MiB; cache slots 5632 against 6144. The code itself names the reason: "CUDA0'S ARENA IS BUILT BEFORE THE SPLIT IS KNOWN" | After the search (the `late_weights` branch at `:4357` already loads late-stage weights later) **rebuild the CUDA0 arena** and apply the trim to it; or build the CUDA0 arena after the search. This one is a clean gain: VRAM is returned, not moved |
| **P5.6** | There is no per-card residency criterion: the tie-break does not prefer layouts where every stage holds 100 % of its own pairs | `auto`: CUDA3 89.0 % (5469/6144); `12,25,37`: 96.6 % (5443/5632); resident experts 23901 (97 %) against 24387 (99 %) | In the tie-break (after `stage_max`) add the maximum of the minimum `held_i/required_i` over stages |

**Expected benefit of P5 (arithmetic).** The upper bound is replicating the measured gain of the pinned layout: **+4.5 % at 31 307 and +7.3 % at 125 011 prompt tok/s**, TTFT -4…-7 %, decode and ITL unchanged (measured: decode does not depend on the split). P5.5 additionally removes -39.5 GB of host traffic per request at 125k -> by the constant in §3.4 another **-10.1 s of TTFT (-19 %)** as an upper bound; realistically less because the regime changes (§6.3). In total P5 is the only element of these notes with a measured confirmation of its mechanism.

**Risks of P5.** (a) a different choice changes the set of resident experts and therefore the answer's text (the hit rate 99.1 % against 99.7-99.8 %) - that is **not a regression**, but `text_sha` acceptance does not apply here; a common-prefix criterion is needed instead; (b) enumerating 15 180 layouts with new terms costs more - `range_hold` is already memoised, adding `stream_i` to it is linear; (c) the risk of "improving for this machine" - every new term is built from quantities already measured (SMxGHz, `link_gbps`, `stage_left`, blob size); no new constants should be introduced; (d) on heterogeneous cards `layer_ms` is not degenerate and the new terms may change the choice there - that is the point, but it needs an A/B on another machine.

**Test for P5:** a unit test of the scoring function (input: this machine's profile and card measurements; expected: `12,25,37` ranks above `11,24,36` on the prefill component); a log regression test - the `layer split auto:` line must print both scores and per-stage residency.

**Benchmark for P5:** the same A/B already done by hand (`comparisons/auto-vs-pinned-split.csv`): `auto` of the new search against `auto` of the old, 3 repeats, 4k/32k/125k. The target signature: the new `auto` picks a layout whose measured speeds match the pinned `12,25,37` within noise (+-0.3 %).

**When not to apply:** one card (`--layer-split auto` is off anyway, `:2388`); fewer than 4 stages (the 0.1.39 curve is kept there, `:3885-3890` - do not touch without a separate measurement); no profile (`--expert-profile` unset) - then `held_mass` cannot be computed and the new terms must rely on P5.5 only.

---

## 5. Checks a GPU->GPU path has to include

Requirements on code that enables a peer route (all from this machine's measured behaviour):

1. **Architecture compatibility.** Do not assume equal CC: check `cudaDeviceGetAttribute(cudaDevAttrComputeCapabilityMajor/Minor)` on both sides and do not enable the path if the device code was built for a higher CC than the lower card (the working build: all 62 cubins `sm_70`).
2. **Pair capability.** `cudaDeviceCanAccessPeer` in both directions **and** `cudaDeviceGetP2PAttribute(cudaDevP2PAttrAccessSupported)`; require `NativeAtomicSupported` only if the path uses atomics; `CudaArrayAccessSupported` is a separate flag for P2b (here 1 on every pair). Keep a test switch like `STRATA_PEER_NO_P2P` (already present).
3. **Enabling.** `cudaDeviceEnablePeerAccess` with the current device set in turn, `cudaErrorPeerAccessAlreadyEnabled` -> `cudaGetLastError()` and success (the idiom at `peer_experts.cpp:104-110`). Check free VRAM before/after: here the price is 0 MiB, but on other platforms that is a disable condition.
4. **Sizes and thresholds.** The fixed cost of a copy is 6.5-6.8 us, independent of size up to 1 MiB; 50 % of bandwidth at 256 KiB, 80 % at 1 MiB, >= 99 % at >= 32 MiB. The rule: **a peer copy only for sizes >= 1 MiB**; smaller exchanges must be merged into one buffer (for the hand-off this holds: 335.5 MB per chunk; for the routing tables `src`/`bounds` it does not, and they stay on the host path).
5. **A per-device budget, not per pair.** A scheduler must account for six concurrent copies giving 12.92 GB/s per pair at 38.8 GB/s per card: do not start more than one large peer copy from one card at a time, or the second only divides the bandwidth.
6. **Streams/events and lifetime.** The copy on a separate non-blocking stream (`PeerPrefill` already has `s`, `s_out`, `s_cp`), dependencies through `cudaEventRecord`/`cudaStreamWaitEvent` (a cross-device event is covered by the `cross_device_event_dependency` test), and the buffer lives until the consumer's event. Replacing `cudaStreamSynchronize` at `:4161` with events should be a separate change and tested separately - it changes order, not route.
7. **Fallback and failure.** Any error during setup (allocation, enabling access) -> a full return to the existing host route with an unchanged result; a flag to force it off. The error must be logged as a line (following `prefill peer: the two GPUs cannot access each other (no P2P)`, `:1693`).
8. **Alignment.** A peer copy imposes no extra requirement (4 KiB copies worked), but residual buffers should be multiples of 128 B to avoid segmentation losses; `m.R` from `cudaMalloc` is a multiple of 256 B - assert it rather than assume it.
9. **What must not change.** The numeric path: the transport must be bit-transparent (§8). No FP16/BF16 compression of the hand-off "to save bandwidth" in the same change - that changes bits and needs its own decision.

---

## 6. Expected effect - arithmetic only

### 6.1 Per element (125 011-token request, `auto`)

| Element | Bytes removed from the host path | Bytes added to the stream | Net host | Delta TTFT at 3.9 GB/s | Upper bound |
|---|---|---|---|---|---|
| P2a (peer copy + buffers in VRAM) | -32.2 GB | +32.2 GB | **0** | 0 | 0 % |
| P2b (a kernel reading a peer's VRAM) | -32.2 GB | +16.1 GB | **-16.1 GB** | -4.1 s | -7.8 % TTFT, -3 % prompt tok/s at 125k; 4x smaller at 31k |
| P3 (the helper route) | 0 (the helper is off in the working config) | 0 | 0 | 0 | **0 % - measured** |
| P4 (peer tier) | 0 (nothing to pay with) | 0 | 0 | 0 | 0 % |
| P5.1-P5.4 (the criterion) | -14.0 GB (moving to a layout holding 96.6 % instead of 89.0 %) | 0 | -14.0 GB | -3.6 s | **+4.5…+7.3 % prompt tok/s - measured with a pinned layout; reproduced a second time 2026-10-17: +4.48 % at 31 307 and +6.66 % at 122 087, delta expert reads -13.1 GB per request (`ab-stand.md` §2-§3)** |
| P5.5 (trimming CUDA0 under auto) | 0 | -39.5 GB (less borrowing) | -39.5 GB | -10.1 s | -19 % TTFT, about +24 % prompt tok/s (upper bound, §6.3). Note: the mechanism was refined by measurement on 2026-10-17: with an explicit split CUDA0 gets **+512 cache slots (5632 -> 6144)** and **total borrowing does not fall** (5178 -> 5259 slots), so the gain comes through cache size and a lower required coverage on CUDA3 (6144 -> 5632 profiled pairs), not through refusing to borrow (§7.2, §11.4) |

### 6.2 What these numbers do not promise

* No element does anything at short prompts: the volumes scale with the number of chunks `T/8192` (§3.3), so at 4k tokens even P5.5 is x1, not x16.
* The method's resolution is +-0.3 % (1.31 % at 4k). Effects under 1 % on this machine **cannot be shown**; they can be neither promised nor ruled out. **Calibrated on the stand 2026-10-17:** session noise 0.06…0.14 % at 32k/125k and 0.42 % at 4k (`ab-stand.md` §4).
* The 3.9 GB/s constant comes from one pair of configurations in a +-5 % neighbourhood.

### 6.3 The regime change (why the upper bounds do not add up)

Today the host path is occupied to its limit (median 5.25, p95 7.67 GB/s against 7.88 nominal). If 50…70 GB of 302 were removed from the uplink in total, the system would stop being host-bandwidth-limited and would run into compute; the linear extrapolation of §3.4 stops working. So the combined effect of P5 + P5.5 should be stated as "**+5…+15 % prompt tok/s at 125k, not measured**", not as the sum of the columns.

---

## 7. Separately: what `--layer-split auto -> 12,25,37` gives, and whether it stands in for a change's benefit

1. **Measured** (baseline-profile.md §8.2, `comparisons/auto-vs-pinned-split.csv`): 31 307 -> 1978.0 tok/s (**+4.48 %**), TTFT 16.57 -> 15.87 s (-4.2 %); 125 011 -> 2546.2 tok/s (**+7.33 %**), TTFT 52.80 -> 49.19 s (-6.8 %); decode and ITL unchanged; GPU0 VRAM lower (13 744 against 15 246 MiB).
2. **The mechanism is not the transport.** It splits into parts, and all are visible in the data:
   * *less streaming:* CUDA3 holds 96.6 % of its pairs instead of 89.0 % -> the engine's counter falls from 269 472 to 255 505 MB per request (**-5.2 %**), and the TTFT change (-3.59 s) relates to the byte change (-13.97 GB) as 3.9 GB/s - i.e. **the linear part of the gain is exactly the reduction of host traffic**; **reproduced a second time 2026-10-17** (`ab-stand.md` §3): 253 528…253 624 -> 240 440…240 442 MB per 122 087-token request (-13.1 GB, -5.2 %), with an independent confirmation from `rxpci` (-32 GB over 3 requests, -4.1 %, of which -12 % on GPU3), residency 97 % -> 99 %, hit rate 99.5…99.8 % -> 100.0 %;
   * *more cache on CUDA0 (measured 2026-10-17):* with an explicit split `stage_trim` applies to CUDA0 -> **+512 slots (5632 -> 6144) and -1394 MiB of used VRAM on CUDA0**, while total borrowing by the prompt path **did not fall** (5178 -> 5259). So P5.5 works through cache size, not through refusing to borrow;
   * *pipeline balance:* the stage with the LM head gets 11 layers instead of 12 - this is not reflected in bytes and probably accounts for the remainder (hypothesis; testable by fixing `MB of expert reads` + `stage_max` in a new search log).
3. **It does not stand in for a change; it partly competes with one.** A change alters **the cost of moving a byte** (host 1.56…6.29 GB/s against 38.8 GB/s per device); a placement change alters **the number of bytes that need moving at all**. They multiply: the better the placement, the fewer bytes remain for which the transport matters. Therefore:
   * they must be measured **separately, as a 2x2 matrix** (`auto`/pinned x change/no change) - otherwise a change's effect gets attributed to placement or the reverse (§9.2);
   * if P5 comes first (the search picks a good layout by itself), the expected effect of P2b **shrinks**, because host traffic is already lower; that is the correct result rather than a bad one, and the protocol must record it;
   * the working configuration stays `auto`: a pinned layout is only a diagnostic probe, and in before/after comparisons it is a separate row, not the baseline.

---

## 8. Acceptance criteria for a change (ready, from the correctness work)

1. **Bit acceptance for a transport (P1, P2, P3, P4).** Moving data does not change content: after warm-up, on stable prompts (long context, fact extraction) `text_sha` must **match** the baseline. A difference means the change altered computation, not routing -> acceptance failed. This is stricter than for P5.
2. **For P5 (the resident expert set changes, the text may change):** a common prefix of at least 95 % on stable prompts; do not use a "matched / did not match" comparison (differences are born in the tail: a common prefix of 109-592 characters out of about 630-668 on unstable short prompts).
   Note, from a measured fact (2026-10-17T06:30Z, `ab-stand.md` §5.2): the 95 % threshold **is not met by the real difference between the two layouts** `auto` / `12,25,37`: on a fixed 10 012-token prompt the common prefix is 32 characters out of 647 (4.95 %), while within each configuration the text is bit-reproducible across sessions. Conclusion: bit acceptance does not apply to P5 by construction, and **any change that alters the auto decisions must pass an output-quality check** (a fixed prompt set with an answer evaluation, or comparing logits at the first diverging token), not only a speed check.
3. **Numeric tolerances** for any new GEMM/reduction (there are none in P2-P4, but if summation appears on the receiving side): `1e-4 · K` against a CPU reference, as in `build/sm70-tests`.
4. **Fixing the output length:** `logit_bias = -100` for the EOS ids 248044/248046, with `gen_n == max_tokens` and `finish_reason = "length"` checked in **every** run; without this the decode column is invalid.
5. **Warm-up:** the first request after a start / config change / split change is not a measurement; comparisons use repeats 2…n only.
6. **Resources:** VRAM on the four cards before/after (for P2 the line with the number of cache slots lost to buffers is mandatory: `expert cache … slots` and `the prompt path borrows …`); no CUDA errors in the log growth; no TTFT growth at 4k.
7. **Counters mandatory in the protocol:** `MB of expert reads` and `the OS read … MB (major faults)` per request - they separate "less PCIe" from "more streaming"; `decode expert cache hit rate`; the `layer split auto:` line with both scores (after P5).
8. **Statistics:** >= 3 repeats, medians, within-row spread; a result counts as changed at |delta| > 1 % prompt tok/s; decode at |delta| > 5 %.
   Note, calibrated on the stand (2026-10-17, `ab-stand.md` §4): the session-to-session noise of prompt tok/s is **0.06…0.14 % at 32k/125k and 0.42 % at 4k** (within-row spread up to 1.19 % at 4k) -> the 1 % threshold is right only for lengths >= 32k; at 4k 2 % is needed. The decode column: session spread **up to 7.8 %** with an unchanged config -> a 5 % threshold does not work; use decode only as a "must not get worse" control with a 10 % threshold and with draft acceptance and hit rate recorded.

---

## 9. The A/B protocol (before/after a change)

**9.1 A folder and build identity.** `raw/ab/<UTC stamp>/` with a mandatory `manifest.json`: sha256 of the binary, the tree (a git SHA - the machine has no git), the config and every script; `engine.log`, `nvidia-smi` before/after, the pid file. Nothing is overwritten.

**9.2 The 2x2 matrix (the difference from the existing runs).**

| | base binary | changed binary |
|---|---|---|
| `--layer-split auto` | A0 (base) | A1 (the effect in the working configuration) |
| `--layer-split 12,25,37` | B0 (diagnostic probe) | B1 (the change on top of a good layout) |

Order: A0 -> A1 -> B0 -> B1 -> **A0 again** (a control for session drift; the difference between the controls sets the run's noise threshold). Each cell: warm-up + 3 repeats at 4k/32k/125k with a fixed output length.

Checked in practice (2026-10-17T05:43…06:38Z, no change applied, all 5 cells on the base binary): `raw/ab/2026-10-17T061723Z-validation3/`, analysed in `ab-stand.md`. Result: 5/5 cells, a run noise threshold of 0.06…0.42 % prompt tok/s, the known `auto` / `12,25,37` difference reproduced in two independent pairs of cells (+4.48 % at 31 307, +6.66 % at 122 087, TTFT -4.2 % / -6.2 %), and the repeat of the base stayed under the significance threshold. Two protocol corrections came out of it: warm up until a stable pair of `text_sha`, and use a "the model is loaded" threshold rather than an "auto layout" check.

**9.3 Transport acceptance before any benchmark.** Before any speed comparison: bit identity on A0 and A1 (`scripts/determinism-probe.py` + `scripts/correctness-integration.py`), comparing `text_sha`. If they differ, no benchmark is run - the change is not correct.

Note after the stand validation: the reference `text_sha` is taken **in the same placement configuration** and **after a stable pair of repeats**. Measured: the text is reproducible across service restarts with one config (A0 = A1 = A0r bit-for-bit), but changes when `--layer-split` changes (F-8) - so acceptance for P1-P4 is only valid with a fixed layout, and P5 needs an output-quality criterion (§8.2).

**9.4 The low-level part.** `build/kernel-load-audit` (loading every kernel of a new build), `build/sm70-tests` (13 tests, run on dev0/dev1/dev2 only), `probes/p2p-probe-ext.cu` with the **fixed** `contention_pair` output (binary `build/p2p-probe-ext.check`, sha `618250f9…`; in the original run that column was 0.00 - a defect of the probe, not of the hardware).

**9.5 The kernel-access probe (needed to choose between P2a and P2b) - taken 2026-10-17T05:13Z.** `probes/peer-read-probe.cu` + `scripts/run-peer-read.sh`, data in `raw/peer-read/2026-10-17T051351Z/` (a 1…128 MiB ladder, three GPU0-GPU2 pairs, 263 rows, 0 correctness errors), analysed in `p2p-results.md` §13. Result: at >= 32 MiB a direct kernel read gives **35.64 GB/s (0.92 of a copy)**, a copy inside a kernel **36.60 (0.95)**, writing into a peer **36.62 (0.95)**; the 90 % plateau threshold is 32 MiB (the hand-off payload is 335.5 MiB, above it); opposing kernel reads on one pair total 64.0 GB/s. **The choice between P2a and P2b is settled in favour of P2b** (the transport is usable, and by bytes P2a is net 0). Not taken: pairs with GPU3 (only in a stopped window) and the 256 MiB point (not enough free VRAM with the service running); the plateau was reached (64 -> 128 MiB added 0.7 %).

**9.6 The phase profile.** `nvlink-probe.py` (2.5 min) before/after: the sign of a working peer path is **`rxpci` falling on the cards while prompt tok/s rises**, with idle traffic remaining exactly 0 (which is what makes the attribution reliable).

**9.7 Returning the service.** Three readiness signs (processes alive, VRAM "model loaded", a real chat completion returning `choices`); update the pid file to the actual wrapper pid first (`ss -ltnp | grep ':8080 ' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2`; note mawk on this system - a three-argument `match()` does not work); start any service variant only through the normal script with an absolute python path.

**9.8 What not to do in the stand.** Do not re-measure: the full benchmark sweep, the P2P matrix, card<->card and host-path bandwidth, the per-phase PCIe profile, the A/B of `STRATA_PREFILL_HELP`, the size ladder and the 6-pair competition, the kernel inventory and the SM70 gate table, the baseline profile and the idle-machine control point, the attribution experiment, the SM70 and multi-card correctness checks, the recall needles.

---

## 10. What would still need asking or checking

| Item | Why | Status |
|---|---|---|
| **Q1** (git/archives) | Bind these notes to a tree SHA rather than only a binary sha256; needed for naming stand folders by SHA | open |
| **Q4** (`ncu` in the private toolkit, no `nsys`) | Direct proof of the copy route is not required: code + the engine's counters + the match with `rxpci` close the question. `ncu` would only be needed if, after P2, one had to explain why a kernel peer read is slower | deferred by the rule "only if the code cannot answer" - the code answered |
| **E2** (the same model on three cards with `--gpu 0,1,2`) | Shrink the search space to C(45,2)=990 and get into the `f < 0.5` regime where the coverage model is known to be wrong - a direct test of P5.4 | needs a ~2 minute window, only with agreement |
| **E1** (a "cold topic", `auto` against `configs/strata-185-iq3_s.split12.json`, x3 in one window) | Tests the risk behind staying on `auto`; its second part (changing the profile -> a new auto choice) tests whether the search adapts after P5 | needs a window, only with agreement |
| The kernel peer-read probe (§9.5) | Choosing between P2a and P2b | **closed 2026-10-17T05:13Z** without a stop window (GPU0-GPU2): P2b is usable - 0.92…0.95 of copy bandwidth at >= 32 MiB (`p2p-results.md` §13). Pairs with GPU3 remain unmeasured (only in a window) |

---

## 11. Limits: what this document does not show

1. None of the P2/P3/P4 effects was measured on this machine - they are recomputed from measured bandwidths and volumes. "Up to -7.8 % TTFT" is an upper bound, not a forecast.
2. The 3.9 GB/s constant (§3.4) comes from one pair of configurations; for larger reductions it overstates the effect (§6.3).
3. The attribution of flow D rests on the code and on matching volumes (the engine's counter, `rxpci`, the slot arithmetic), not on per-path counters: there are no NVLink byte counters on this machine.
4. The difference in CUDA0 free VRAM (2.3 GiB) between `auto` and an explicit split was measured, but the two logs differ in **both** the trim **and** the length of the layer range; attributing the clean effect of P5.5 needs an `auto` run with the condition fixed - that would be the first measurement after such a change, not a ready number. **Added 2026-10-17:** the second matrix run confirmed the effect itself (+512 cache slots on CUDA0, -13.1 GB of stream per request, +4.48 % / +6.66 % prompt tok/s) but did **not separate** the trim's contribution from the shorter CUDA3 range's - they change together (`ab-stand.md` §3).
5. The hypothesis that part of the pinned layout's gain comes from pipeline balance (rather than only -5.2 % of bytes) is untested: it needs a log of a new search with `stage_max` and a prefill score.
6. Direct kernel access to a peer's memory as a transport **is measured** (2026-10-17T05:13Z and 05:45Z, all 12 ordered pairs, 1…256 MiB, 0 correctness errors): a kernel reading a peer = 0.92…0.94 of a peer copy at hand-off payloads, so the transport for P2b is usable. But **the end-to-end effect of P2b inside the engine is not measured** - that is the first measurement after such a change.
7. These notes do not propose applying anything to a working tree: by the rules of this investigation the author decides what to merge, and source edits made here could only have been compile-checked.
8. **A new limit found on the stand (F-8, `ab-stand.md` §5.2):** changing `--layer-split` changes the output of greedy generation on a long prompt (a common prefix of 32 characters out of 647) while the text is fully reproducible within each configuration; the mechanism is not established. Consequence: changes to the auto decisions cannot be accepted on speed alone (§8.2).
