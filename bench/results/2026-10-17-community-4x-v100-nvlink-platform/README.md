# 4x Tesla V100-SXM2 + NVLink: platform dossier

```
Artifact:            README.md (index and short engineering report; the full package is listed in §10)
Date (UTC):          2026-10-17
Object:              Strata 0.1.44.1 (tag v0.1.44.1 = 296b59d2f73c8f04a805df29c2fa97655adace78),
                     binary <HOME>/strata185/Strata/engine-cuda12/strata sha256 bdfec9f553d07c2b… (build_info "Strata 0.1.44.1")
Model:               Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S, 48 layers, MoE 512 experts x 10/token, IQ3_S,
                     context 262 144, KV int8, speculative decoding (draft, spec 4)
Hardware:            4x Tesla V100-SXM2-16GB (CC 7.0, ECC on), NVLink 2.0 (2 links per pair, NV2),
                     PCIe Gen3 x8 per card behind one shared x8 uplink (Minisforum MS-01, carrier board
                     V100-SXM2-NVLINK_D V2.10, Broadcom PEX8748 switch) - details in system-inventory.md
Service config:      <HOME>/strata185/Strata/strata-185-iq3_s.json sha256 f539e51b…, --layer-split auto,
                     --expert-cache auto, --prefill auto (all automatic modes)
Project folder:      <HOME>/strata185/Strata+4xV100+NVlink/ (raw data: raw/<type>/<UTC stamp>/, nothing overwritten)
Status labels:       measured (this machine) · calculated (arithmetic on measured numbers) ·
                     hypothesis (not tested) · proposal (a suggestion) · not possible in this build (with the reason)
No secrets in any file (the api_key is not published).
```

**Who this is for.** Written for a reader with no access to the correspondence behind it: every statement
carries a link to an artifact and says whether it was measured or computed. Assumptions are never presented
as facts.

**What this folder is not.** It is not a request to support this hardware. The support matrix starting at
sm_75 and Volta being a community-tested target ([docs/NVIDIA_V100.md](../../../docs/NVIDIA_V100.md)) is the
arrangement this machine lives with anyway: it builds its own engine because CUDA 13 dropped Volta. Everything
here is measurement. The notes on the code in `patch-notes.md` are notes on what the measurements say, not a
patch to merge - no engine file was touched, and the measuring binary and the working config were never changed.

---

## 1. What is confirmed

| Statement | Status | Where |
|---|---|---|
| NVLink 2.0 on this machine is fully healthy: 24 links, each of the 6 pairs carrying 2 links at 25.781 GB/s; P2P and P2P atomics available on every pair; `CudaArrayAccessSupported = 1`; enabling peer access costs **0 MiB** | measured | `p2p-results.md` §1-§4, `p2p-matrix.csv` |
| Peer copy bandwidth: a **38.77 GB/s plateau**, the same on all 6 pairs in both directions (256 MiB); 38.45 at 32 MiB; reproduced independently from PR #2525's 38.8 with a 0.08 % difference | measured | `p2p-results.md` §5, `p2p-benchmark.csv` |
| Peer copy latency: a fixed cost of **6.5-6.8 us**; `T = 6.7 us + size / 38.8 GB/s`; 50 % of the plateau at 256 KiB, 80 % at 1 MiB, >= 99 % at >= 32 MiB | measured | `p2p-results.md` §5 |
| **The ~38.8 GB/s ceiling is set at the device, not at the pair or the link:** 6 pairs running at once give 77.50 GB/s total, 12.92 GB/s per pair (a 3.00x fall) | measured (the nature of the limiter is a **hypothesis**) | `p2p-results.md` §9 |
| **Direct kernel access to a peer's VRAM is a usable transport:** a kernel reading a peer 32.13 -> 36.25 GB/s at 32 -> 256 MiB = **0.838 -> 0.936 of a peer copy**; copy performed inside a kernel 36.63; writing into a peer 36.64; two kernels reading at each other on one pair **65.1 GB/s**; 0 correctness errors (576 rows, all 12 ordered pairs) | measured | `p2p-results.md` §13, `ab-stand.md` §7, `raw/peer-read/2026-10-17T051351Z/`, `…T054509Z/` |
| The dominant flow of a long prompt is **host->card (MoE expert streaming)**, not activation staging: 269 455 MB for a 125 011-token request by the engine's own counter (a different session logs 269 472 MB), which matches the measured 270.5-273.9 GB of `rxpci` | measured | `data-path-analysis.md` §2, `patch-notes.md` §3.1 |
| **In 0.1.44.1 a layer-split machine issues no GPU->GPU transfer at all:** `--peer-device` with `--layer-split` is refused by the parser; the stage-helper route is hard-wired `p2p = false` (`prefill.cpp:1591`) | measured (code) | `sm70-compatibility.md`, `patch-notes.md` §2 |
| **This dominant flow cannot be moved onto NVLink on this hardware:** 4 x 16 GB of VRAM against 50.29 GB of experts, the caches filled exactly, no card holds another card's experts | measured (arithmetic on measured geometry) | `patch-notes.md` §2.4, §3.3 |
| Structural relation: **1 MiB of VRAM taken from the expert cache = (T/8192) MiB of extra host traffic** per request of length T | calculated from the code and the measured slots | `patch-notes.md` §3.3 |
| The marginal price of a host byte on this machine is about **3.9 GB/s** (delta -13 967 MB of host traffic against delta -3.59 s of TTFT) | calculated from **one** pair of configurations | `patch-notes.md` §3.4 |
| The difference between `--layer-split auto` and `12,25,37`: **+4.48 % prompt tok/s at 31 307** and **+6.66…+7.33 % at 122-125k**, TTFT -4.2…-6.8 %; **measured twice, in two independent pairs of cells** | measured | `baseline-profile.md` §8.2, `ab-stand.md` §2-§3 |
| The mechanism of that gain: **+512 cache slots on CUDA0 (5632 -> 6144)** and profile coverage on the narrow CUDA3 rising 89.0 % -> 96.6 %, residency 97 % -> 99 %, -13.1 GB of stream per request; **the total slot borrowing by the prompt path does not fall** (5178 -> 5259) | measured (the "trim" and the "shorter CUDA3 range" contributions are not separated) | `ab-stand.md` §3 |
| The `auto` search criterion is **degenerate**: 30 search lines in the working log give 4 different splits at the same score, "predicted 28.4 ms" | measured (working log) | `patch-notes.md` §5 |
| The build is fully compatible with V100: **2084 kernels from 62 cubin/PTX objects load without one error** (`sm_70`, no JIT involved), resource limits are not exceeded (max 255 registers, 41 104 B shared), the SM70 paths are on by default, and the >= sm_75 requirements are covered by fallback paths | measured | `sm70-compatibility.md`, `correctness-report.md` §2 |
| The SM70 numeric primitives (HMMA, fp32/fp16 GEMM) match a CPU reference within `1e-4 * K`; P2P copy and cross-card synchronisation are correct **37/37** | measured | `correctness-report.md`, `p2p-correctness.json` |
| Greedy generation is **bit-reproducible within one placement configuration - and across service restarts** (checked at 1259 and 10 012 prompt tokens, and at 80 999) | measured | `correctness-report.md` §5, `ab-stand.md` §5.1 |
| **The first request after any change to engine state returns different text**; the state warms up over **2…3 requests** (the expert-cache hit rate rises 98.8 -> 99.8 %) | measured | `correctness-report.md` §5, `ab-stand.md` §7.1 |
| **Changing `--layer-split` changes the output** on a long prompt: at 10 012 tokens the `auto` and `12,25,37` texts share a **32-character prefix out of 647**; at 1259 tokens they are identical | measured as an observation (F-8); the mechanism is a **hypothesis** | `ab-stand.md` §5.2 |
| The noise floor of the method: prompt tok/s **0.06…0.14 % at 32k/125k**, **0.42 % session-to-session and up to 1.19 % within a row at 4k**; decode **up to 7.8 %** with an unchanged config | measured | `ab-stand.md` §4 |

## 2. What does not work (negative results, all measured)

1. **The stage helper adds nothing.** A/B of `STRATA_PREFILL_HELP` / `…_FRAC` in two sessions: no change in
   speed. The reason: the helper is off in the working config, and with a layer split it has nothing to carry.
   So P3 on this machine is 0 %. (`patch-notes.md` §4, data from PR #2525.)
2. **`--peer-device` / the remote cache cannot be switched on:** there is no free VRAM on any card (the caches
   are filled exactly). So P4 on this machine is 0 %. (`patch-notes.md` §2.4.)
3. **Moving the hand-off buffers into VRAM (P2a) is net 0:** the 32.2 GB of PCIe removed comes straight back
   as lost expert-stream bytes. The variant that works is P2b (a kernel reading the peer's VRAM), net -16.1 GB.
   (`patch-notes.md` §3.3.)
4. **The decode column is not a measure on this machine:** up to 7.8 % session-to-session spread with an
   unchanged config; the reason is visible in the counters - draft acceptance itself moves by 3-7 percentage
   points between sessions. (`ab-stand.md` §2, §4.3.)
5. **Short prompts (4k) do not separate configurations:** the volumes scale with the number of chunks,
   `T/8192`, so at 4k even P5.5 is x1 rather than x16; the measured B0/B1 differences at 4k
   (+0.58…+1.41 %) sit at the noise boundary of the method. (`ab-stand.md` §2, `patch-notes.md` §6.2.)
6. **GPU3 is not available to an external CUDA context while the service runs** (`cudaSetDevice(3)` -> out of
   memory, ~328 MiB free). With the service stopped, GPU3 behaves like an ordinary card and its pairs are
   indistinguishable from GPU0-GPU2 (<= 0.06 GB/s). (`correctness-report.md` §2.5, `ab-stand.md` §7.)
7. **There is no instrumental confirmation of NVLink routing:** NVLink byte counters do not exist on this
   machine (DCGM 4.7.0 without the profiling module; NVML Sampling & Reporting absent in both 580.178.04 and
   595.91.07). Every statement about routing is indirect (code + matching volumes against `rxpci`/`txpci` + the
   engine's own counters). (`p2p-results.md` §12, `patch-notes.md` §1.)
8. **CUDA 13 does not build sm_70**; the build needs a private CUDA 12.8 and `-DSTRATA_EXPERIMENTAL_SM60=ON`.
   (`sm70-compatibility.md`.)

## 3. What was already measured in PR #2525 (and therefore not re-measured here)

The full list is in the audit of existing materials (kept out of this folder, see `TRIMMED.md`). Used here as
ready data:

* the 0.1.44.1 and 0.1.43 baselines, a length sweep to 260 331 tokens, recall needle 6/6;
* `canAccessPeer = 1` on all 6 pairs, NVML `NVLINK=OK`, 38.8 GB/s on every pair (256 MiB, 12 repeats), the
  price of peer access 0 MiB, the host path 6.29 -> 1.56 GB/s per card, 29.4-29.9 GB/s during an active prefill;
* the per-phase PCIe profile (RX 61.9-78.4 GB per card during prefill, TX 6.9-11.9), uplink saturation
  (median 5.25, p95 7.67, worst second 9.33 GB/s against 7.88 nominal), the A/B of `STRATA_PREFILL_HELP`;
* the open question in PR #2525 §9.1 (which flow dominates) - **closed** by the data in this folder.

**One discrepancy established here rather than in that PR:** this machine's profile sits 1.5-5.7 % below
PR #2525's in prompt tok/s **with an identical binary (sha256 `bdfec9f5…`) and config (`f539e51b…`)**. The
cause was found by stopping the service: **the `--layer-split auto` choice** (`K=11,24,36` instead of
`K=12,25,37`) - not the version, not the hardware, not background load, not warm-up. Any comparison from this
machine should therefore use `baseline-profile.csv` as its baseline, not the numbers in that PR.
(`baseline-profile.md` §8.)

## 4. What is new here

| What | Artifact | Key numbers |
|---|---|---|
| Hardware and software inventory, topology | `system-inventory.md` + `.json`, `raw/{inventory,topology}/2026-10-16T232834Z/` | 4x V100-SXM2-16GB, ECC 0 errors, x8 where x16 is possible, 24 NVLink links, PEX8748, x8 uplink at 4:1 |
| P2P matrix (all four capability attributes, UUID/bus, price of peer access) | `p2p-matrix.csv`, `raw/p2p/2026-10-16T235659Z/` | every pair available, 19/19 correctness tests |
| The full P2P size ladder + 6-pair competition | `p2p-benchmark.csv`, `raw/p2p/2026-10-17T001226Z/` | 211 rows, 17 sizes 4 KiB…256 MiB x 6 pairs x 2 directions, no gaps; 38.77 GB/s; 77.50 GB/s over 6 pairs |
| Bandwidth of direct kernel access to a peer's VRAM (including pairs with GPU3, to 256 MiB) | `raw/peer-read/2026-10-17T051351Z/`, `…T054509Z/`, `p2p-results.md` §13 | 576 rows, 0 errors; 0.838…0.936 of a copy; 65.1 GB/s for opposing kernel reads |
| A baseline profile of Strata with a **valid** decode column (EOS banned) | `baseline-profile.md`, `baseline-profile.csv` | prompt tok/s 868.7 at 3971 … 2372.8 at 125011; decode 96.2-109.6; ITL 9.12-10.40 ms; TTFT 4.59…52.78 s; VRAM peak 15246/15600/15188/16068 MiB; service start 66 s |
| Attribution of the delta against the PR (the pinned-split experiment) | `baseline-profile.md` §8, `comparisons/auto-vs-pinned-split.csv` | +4.48 % at 31 307, +7.33 % at 125 011 |
| Kernel inventory and the SM70 gate table | `sm70-compatibility.md`, `raw/build-audit/…` | 2084 kernels, 56 TUs (2028 attributed), all sm_70; 20 rows "component -> requirement -> check" |
| Correctness matrix (41 rows) and 13 integration checks through the API | `correctness-matrix.csv`, `correctness-report.md` | 27 PASS, 4 LIMITATION, 3 FINDING, 7 GAP; 2084/2084 kernels load; tolerance `1e-4·K` |
| The inter-card path map with volumes and the notes on the code | `patch-notes.md` | hand-off 32.2 GB per request (calculated from the code, matching the PR's TX within 9 %); streaming 269 472 MB; P1-P5 |
| The A/B stand and its validation (2x2 matrix + a repeat of the base) | `ab-stand.md`, `test-plan.md`, `raw/ab/2026-10-17T061723Z-validation3/` | noise 0.06…0.42 %; reproduction of +4.48 % / +6.66 %; F-8 |

## 5. The confirmed bottleneck

**There are two, and both are confirmed.**

1. **The external one:** the shared PCIe x8 uplink (four x8 cards behind one x8 = 4:1 oversubscription) -
   6.29 -> 1.56 GB/s per card; during prefill a median of 5.25 GB/s, p95 7.67, a worst second of 9.33 against
   7.88 nominal; 33 of 75 seconds at or above 6 GB/s. The host path is occupied to its limit.
2. **The internal one (the larger):** **the number of bytes the engine decides to re-read from host**, not the
   speed at which they cross. For a 125 011-token request that is 269 455 MB of expert stream, caused by the
   prompt path borrowing expert-cache slots for its buffers (1954/1841/1706/1631) while the caches are filled
   exactly. The evidence: the engine's counter (`for 269454.7 MB of expert reads`, with 1.3 MB of OS reads and
   0 major faults) against the measured 273.9 GB of `rxpci` (`data-path-analysis.md` §2) against the slot
   arithmetic; and an independent reproduction in the A/B matrix (-13.1 GB of stream per request against
   -4.1 % PCIe received and +4.48…+6.66 % prompt tok/s).

**What this means for anyone reading it:** on this rig NVLink does not solve "move 269 GB" - there is nowhere
to take them from. It does solve "move the 32.2 GB of hand-off between stages", and above all the real task,
"stream less", which is decided by the placement search rather than by the transport. The price of a byte here
is about 3.9 GB/s (one calibrated pair of configurations): 1 GiB of removed traffic is about 0.27 s of TTFT,
about 0.5 % at 125k.

## 6. Where NVLink use is proven, and where it is not

| Statement | Status |
|---|---|
| The physical path is healthy and P2P is available on every pair | **proven** (24 links, the capability matrix, 37/37 tests) |
| The bandwidth and latency of a peer copy are known with the spread controlled | **proven** (38.77 GB/s, 6.5-6.8 us, size thresholds) |
| A kernel reading a peer's VRAM directly is a usable transport (0.94 of a copy at the relevant sizes) | **proven** - this is the new result, and it is what selects P2b over P2a |
| Strata **uses** NVLink in the working layer-split mode | **not proven - and rejected by the code:** no path performs a GPU->GPU copy |
| Part of the long-prompt traffic goes over NVLink | **not proven instrumentally** (no counters); by code and by volumes, no: that traffic is host->card |
| Patch P2b would speed up prefill on this machine | **not measured.** The transport's capability and the byte budget (net -16.1 GB) are proven; the end-to-end effect is the first measurement after such a change |
| Patch P5 (the `auto` criterion) reproduces the measured gain of a pinned layout | **partly proven:** the gain itself was measured twice; what exactly produces it (the CUDA0 trim against the shorter CUDA3 range) is not separated |

## 7. The minimal change to try first

The order follows `patch-notes.md` §0 (each item names the file and function, the expected effect and the
condition for not enabling it). In short:

1. **P5 - the `--layer-split auto` search criterion** (priority 1). Object: `src/program/generate.cpp:3832,
   3907-3946, 3957-3968, 4131-4199`. The criterion does not see: the cost of the LM head on the last stage,
   the cost of the prefill pipeline, residency on an individual (narrow) card, the cost of streaming.
   Evidence of degeneracy: 4 different splits at the same score "predicted 28.4 ms". Expected: a replication
   of the measured +4.5 % at 31 307 and +6.7…7.3 % at 122-125k **without pinning anything by hand**.
2. **P5.5 - `stage_trim` is not applied to CUDA0 under `auto`** (priority 2). Object:
   `src/program/generate.cpp:3082` (`const bool stage_trim = trim_asked && multi_gpu && !split_auto`).
   Measured with an explicit split: +2.3 GiB of free VRAM on CUDA0 and **+512 cache slots** -> -13.1 GB of
   expert stream per request. The upper bound under `auto`: -19 % TTFT at 125k (calculated).
3. **P1 + P2b - peer access for the stage set, and inter-stage hand-off by a kernel reading a peer's VRAM**
   (priority 3). Objects: `src/core/peer_experts.cpp:97-110` (enabling peer access, price 0 MiB),
   `src/prefill/prefill.cpp:4161` (D2H), `:2189` (H2D), `:1106` (host buffers). The mandatory condition:
   **do not allocate VRAM for the buffers** (P2a is net 0). Expected: net -16.1 GB -> up to +8 % prompt tok/s
   at 125k (calculated, not measured).
4. **P3 (the helper route, `prefill.cpp:1591`) and P4 (`--peer-device` with a split)** - 0 % on this machine;
   their value is the correctness and testability of that path on other rigs (priorities 4-5).
5. **Do not touch:** the decode/verify hand-off - 307 KB per token is <= 0.5 % of ITL.

The checks a peer route has to satisfy, from the code and from these measurements: CC and `canAccessPeer` plus
the P2P attributes; sizes >= 1 MiB (the 50 %/90 % thresholds of the plateau are 8/32 MiB for kernel reads and
256 KiB/1 MiB for copies); the 38.8 GB/s per-device budget and the pair competition; streams/events and buffer
lifetime; a fallback to the host path; alignment.

## 8. The test such a change has to pass

A ready protocol, **already used**: `test-plan.md` §1-§9, implemented in `scripts/ab-run.sh` (plus 6 helper
scripts); a worked example and its analysis in `raw/ab/2026-10-17T061723Z-validation3/` and `ab-stand.md`.

```bash
cd <HOME>/strata185/'Strata+4xV100+NVlink'
setsid nohup scripts/ab-run.sh --patched-binary <path to the changed binary> --label patch1 \
  > raw/ab/run-$(date -u +%H%M).log 2>&1 &
tail -f raw/ab/run-*.log
```

What the stand does by itself:

1. **A 2x2 matrix plus a control:** `A0 auto:base -> A1 auto:patched -> B0 split12:base -> B1 split12:patched
   -> A0r auto:base`. The repeat of the base sets the noise threshold for that run; without it a change cannot
   be told from noise.
2. **A manifest:** sha256 of the binary, the config and every script, plus `build_info` from `/props` (there is
   no git on the machine - Q1).
3. **Readiness of a cell by three signs:** the processes are alive, VRAM says "the model is loaded" (a sum of
   >= 57 000 MiB and every card >= 13 000 MiB), and a real `/v1/chat/completions` returned `choices`.
4. **Bit-identity before the benchmark:** a request is repeated until two consecutive `text_sha` match (at most
   6 attempts; the state warms over 2…3 requests). For P1-P4 (transport) a mismatch with the reference means the
   acceptance fails and no benchmark is run. **For P5 bit-identity does not apply by construction** (a placement
   change changes the output, F-8), so an output-quality check is required, not only a speed check.
5. **The benchmark:** lengths `4k, 32k, 131072`, >= 3 repeats, medians, a fixed output length
   (`logit_bias = -100` for the EOS ids 248044/248046, with `gen_n == max_tokens` and
   `finish_reason = "length"` checked).
6. **The mandatory counters (extracted automatically):** `… MB of expert reads`, `the OS read … MB (major
   faults)`, `decode expert cache hit rate`, `expert cache … slots`, `the prompt path borrows …`,
   `layer split auto:` (after P5, with both scores), any CUDA error lines, and per-card PCIe `rxpci`/`txpci`.
7. **A summary:** `summary.{csv,json,md}` with the delta against the base cell, the delta against
   `baseline-profile.csv`, and verdicts.

**The significance thresholds calibrated on this stand:** prompt tok/s |delta| > 1 % at lengths >= 32k and
|delta| > 2 % at 4k (the measured noise is 0.06…0.42 %); decode only as a "must not get worse" control with a
10 % threshold (the session-to-session spread reaches 7.8 %); any new numeric reduction to a tolerance of
`1e-4·K` against a CPU reference.

**What such a change must not make worse:** TTFT at 4k; VRAM (the line with the number of cache slots lost to
buffers); no CUDA errors in the log growth; no growth in `the OS read … (major faults)` - the sign that a
"speed-up" was bought with extra streaming rather than with fewer bytes.

## 9. What still needs work

1. **The end-to-end effect of P2b inside the engine** - not measured. It would be the first measurement after
   such a change: compare `MB of expert reads` and `rxpci` before and after at 32k and 125k, not only TTFT.
2. **Separating the contribution of P5.5** (the CUDA0 trim) from the shorter CUDA3 range - they change
   together. It needs an `auto` run with only the trim condition changed (one changed variable).
3. **The pipeline-balance hypothesis** (part of the pinned-layout gain is not in bytes): testable by fixing
   `stage_max` and the prefill score in a new search log.
4. **F-8: why a placement change changes the output.** The check: compare logits at the first diverging token
   and find out whether experts served through a ring slot give bit-identical results to the same experts from
   a cache slot (the same blob, different slots).
5. **The stability of the `auto` decision under a different access profile** (experiment E1): a "cold topic",
   `auto` against a pinned layout x3, recording the sha256 of the expert profile and re-running `auto`. This is
   a direct test of the risk of staying on `auto`.
6. **The regime where experts do not fit the cache at all** (E2: the same model on three cards with `--gpu
   0,1,2`, f < 0.5) - a direct test of P5.4 (the price of streaming inside the criterion).
7. **Kernel load on several pairs at once** (half-duplex and the 38.8 GB/s per-device split under kernel
   traffic) was not measured.
8. **Instrumental control of NVLink:** there are no NVLink byte counters on this machine; the minimum action is
   DCGM with the profiling module, or `ncu` (present in the private CUDA 12.8) with the service stopped (Q4).
9. **Binding the folders to a tree commit SHA** (Q1: git/archives). Today the primary identity is the sha256 of
   the binary plus `build_info` from `/props`.

## 10. The full package

| Item | File | State |
|---|---|---|
| hardware and software inventory | `system-inventory.md`, `system-inventory.json`, `raw/inventory/2026-10-16T232834Z/` | present |
| PCIe/NVLink topology | `system-inventory.md` (topology), `raw/topology/2026-10-16T232834Z/` | present |
| P2P matrix | `p2p-matrix.csv`, `raw/p2p/2026-10-16T235659Z/` | present (all 6 pairs) |
| bandwidth and latency tests | `p2p-benchmark.csv`, `p2p-results.md` §5, `raw/p2p/2026-10-17T001226Z/` (211 rows); `raw/peer-read/2026-10-17T051351Z/`, `…T054509Z/` (576 rows) | present |
| probe source | `probes/p2p-probe-ext.cu`, `probes/peer-read-probe.cu`, `probes/cuda-tests/*` | present |
| baseline benchmarks | `baseline-profile.md`, `baseline-profile.csv`, `raw/profiles/2026-10-17T004753Z/`, `…T005858Z/` | present |
| data-path profiles | `data-path-analysis.md`, `patch-notes.md` §2-§3, and the interconnect experiments in PR #2525 | present (no NVLink counters - see §2 item 7) |
| correctness tests | `correctness-report.md` §2-§3, `probes/cuda-tests/*`, `raw/correctness/2026-10-17T023315Z-cuda/` | present |
| SM70 compatibility matrix | `sm70-compatibility.md` (20 rows "requirement -> check"), `raw/build-audit/…` | present |
| integration test results | `correctness-matrix.csv` (41 rows), `raw/correctness/2026-10-17T022637Z/`, `…T063813Z/` | present |
| service restart confirmation | `raw/ab/*/ready.json` for every cell, `raw/logs/2026-10-17T054344Z/` (before the stop), `raw/logs/2026-10-17T130536Z-service-check/` (final state check) | present |
| architectural limits | `sm70-compatibility.md`, `patch-notes.md` §1-§2 | present |
| implementation notes | `patch-notes.md` §0, §4-§5, §7-§10 | present |
| known limits and open questions | `test-plan.md` §8, `ab-stand.md` §6.5, `patch-notes.md` §11 | present |
| the method for testing a change and its measured noise | `ab-stand.md`, `test-plan.md` | present |
| repeatable scripts | `scripts/` (sha256 in the run manifests) | present |
| what was left out of this folder, renames, and the anonymization applied | `TRIMMED.md` | present |

Two practical notes about the files themselves. **The reports are in English; the raw artifacts and the tools
are not translated on purpose** - the comments and printed messages inside `scripts/` and `probes/`, and the
notes some of them write into raw logs and CSV columns, are in Russian, while every number, identifier, engine
log line and API exchange is in English. And the tools are byte-identical to the ones that produced the numbers
(20 of 21 files; the 21st differs only by an anonymized path), so the SHA-256 values quoted in the reports and
in the run manifests identify these files. `TRIMMED.md` lists the five anonymization substitutions and how to
verify that they are the only edits.

**Suggested reading order:** `patch-notes.md` -> `p2p-results.md` -> `baseline-profile.md` (§8 separately) ->
`correctness-report.md` -> `sm70-compatibility.md` -> `ab-stand.md` -> `test-plan.md` -> `system-inventory.md`
-> this file.

## 11. How to reproduce the key results

```bash
cd <HOME>/strata185/'Strata+4xV100+NVlink'

# 1. Inventory and topology (no service stop)
scripts/collect-inventory.sh

# 2. P2P: the capability matrix and the size ladder with the service running (GPU0-GPU2)
scripts/run-p2p.sh
#    the full ladder 4 KiB…256 MiB x 6 pairs x 2 directions + pair competition needs a stopped-service window:
scripts/run-p2p-full-detached.sh

# 3. Direct kernel access to a peer's VRAM (all 4 cards - only with the service stopped)
scripts/run-peer-read.sh --devices 0,1,2,3 --max-size 256M --blocks 5

# 4. Baseline profile (no stop; a fixed output length is mandatory)
#    the "decode-only" point (120-token prompt, 1024 generated) is added by the script automatically
scripts/baseline-profile.py --lengths 4k,16k,32k,64k,128k --reps 3

# 5. Kernel inventory and SM70 gates
python3 scripts/audit-kernel-inventory.py

# 6. Correctness matrix and integration checks
python3 scripts/correctness-integration.py && python3 scripts/build-correctness-matrix.py

# 7. The A/B stand for a change (a ~40 minute stop window; record the actual wrapper pid before stopping)
ss -ltnp | grep ':8080 ' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2   # -> echo <PID> > ~/strata185/run/server-iq3_s.pid
scripts/ab-run.sh --patched-binary <path> --label patch1
```

The probes build only with the private toolchain: `<HOME>/strata185/cuda-12.8` (nvcc 12.8; CUDA 13 does not
build sm_70) and `<HOME>/strata185/gcc185/usr/bin/g++-14`, `-arch=sm_70`. Artifacts are written only inside the
project folder; every new measurement gets its own `raw/<type>/<UTC stamp>/`.

## 12. Limits of this report (what cannot be claimed from these data)

1. The effects of P2/P3/P4 are **upper bounds from arithmetic, not measurements**. "Up to +8 %" is not a
   forecast.
2. The 3.9 GB/s constant comes from **one** pair of configurations and overstates the effect of larger traffic
   reductions (the system stops being host-bandwidth-limited).
3. Routing is established by the code and by matching volumes (the engine's counter against `rxpci` against the
   slot arithmetic), but **not by per-path counters**: NVLink byte counters do not exist on this machine.
4. The contributions of the CUDA0 trim and of the shorter CUDA3 range to the pinned-layout gain are **not
   separated**.
5. The hypothesis about the pipeline-balance contribution (outside bytes) is untested.
6. F-8 (a `--layer-split` change alters the output) is an observation; the mechanism is not established.
7. The `local_read` column in the probe data (362…519 GB/s) is physically unreachable for V100 HBM2 (300 GB/s):
   it is usable only as "the kernel is not the limiter" and is never part of a conclusion.
8. The length ladder in the A/B stand reaches `prompt_n = 122 087` rather than 125 011 (different wording of
   the target); comparisons are valid inside a run, not across runs with different targets (fixed in
   `test-plan.md` §2).
9. Everything was measured at `concurrency/serving = 1`. Behaviour under concurrent load was not studied.
10. No change has been implemented or measured: this folder describes the **object**, the **budget** and the
   **method**, not the result of applying anything.
