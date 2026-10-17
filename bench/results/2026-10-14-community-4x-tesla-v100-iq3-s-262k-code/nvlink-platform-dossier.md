# Platform dossier: what an NVLink / P2P route would need on this machine

This appendix exists so that the multi-GPU side of Strata can be developed against this
hardware without anyone here having to guess what the code does, and without the author
having to own an SXM2 carrier board. Everything below is measured on this machine on
2026-10-15, except the code map, which is read from the 0.1.44.1 tree.

The premise: the layer-split design is written for hardware without NVLink
(`docs/MULTI_GPU.md`: "No NVLink or peer-to-peer access is needed"), and the prompt
path's host-staged route is justified in the source by measurements on 2x RTX 3090 -
PCIe 4.0 x16, no P2P (`src/prefill/prefill.cpp:1560-1591`). This board is the first
platform we know of where that branch could be turned on. So the useful contribution
from here is not a question but a dossier.

## 1. P2P is available on every pair - nothing gates it

`p2p-probe.cu` (in this directory), run with the model loaded:

| Pair | `cudaDeviceCanAccessPeer` | `AccessSupported` | `NativeAtomicSupported` | `PerformanceRank` |
|---|---|---|---|---|
| GPU0-1, GPU0-2, GPU0-3, GPU1-2, GPU1-3, GPU2-3 | 1 / 1 | 1 | 1 | 0 |

NVML agrees and says which fabric it would use - `nvmlDeviceGetP2PStatus`
(`p2p-nvml.py`): `NVLINK=OK` and `PCI=OK` for all six pairs.

**Enabling peer access costs 0 MiB of device memory**, measured as a `cudaMemGetInfo`
delta on GPU1/GPU2 before any allocation. This matters here specifically because the
free VRAM with the model loaded is small: 256 MiB on CUDA3, 404 MiB on CUDA0, 998 MiB
on CUDA1, 1350 MiB on CUDA2. A P2P route that needs no extra per-pair staging buffers
fits on this machine; one that needs a buffer per pair would not.

## 2. What the P2P path is worth, measured

Same program, 32 MiB `cudaMemcpyPeerAsync`, 20 repetitions, GPU1 <-> GPU2 (the only
pair with enough free VRAM while the model is loaded):

| Condition | GPU1 -> GPU2 | GPU2 -> GPU1 |
|---|---|---|
| idle (model loaded, no request in flight) | 38.4 GB/s | 38.4 GB/s |
| during a real 31375-token prefill | 29.9 GB/s | 29.4 GB/s |

The same measurement was taken while the server processed the prompt, with the PCIe
counters captured in parallel to prove the load was real (`p2p-under-load.sh`); the
per-card PCIe traffic during those seconds was 0-5.5 GB/s.

For comparison, the host path on this board: each card has a Gen3 x8 downlink
(~7.9 GB/s), and all four share one Gen3 x8 uplink through the PEX8748 switch, i.e.
4:1 oversubscription, ~2 GB/s per card effective. Measured during a 125351-token
prefill: 1.2-1.6 GB/s average per card, 5-6 GB/s peaks. So the peer path is roughly
**15-20x the effective host bandwidth under load**, and ~5x even the unshared per-card
link.

Two things follow from the 38.4 GB/s number that are worth stating explicitly:

- It is **not reachable over this PCIe fabric** (x8 Gen3 is 7.9 GB/s per card, and the
  shared uplink is below that), and it is consistent with two bonded NVLink 2 links
  (2 x 25.781 GB/s = 51.6 GB/s theoretical, 74% achieved). So this is indirect evidence
  that when P2P is enabled, the traffic really does go over NVLink on this board - the
  fact the DCGM counters could not give us here.
- `NativeAtomicSupported = 1`, so a P2P route on this platform does not have to avoid
  peer atomics.

A full 6-pair matrix needs the server stopped (each context needs ~400 MiB); that is a
10-minute run and can be done on request.

## 3. What the prompt path actually moves

From the phase probe ([nvlink-probe-phases.json](nvlink-probe-phases.json),
`nvidia-smi dmon -s t`, 1 s resolution):

| Phase | RX per card | TX per card |
|---|---|---|
| idle | 0 | 0 |
| generation, 85 -> 60 tokens | 0-706 MB | 0-67 MB |
| prefill 31375 tokens | 13.7-23.5 GB | 1.5-2.4 GB |
| prefill 125351 tokens | 61.9-78.4 GB | 6.9-11.9 GB |

Reference arithmetic for reading those numbers:

- the activation stream for a 125351-token prompt is `125351 x 47 layers x 2560 x 4 B`
  = **60.2 GB** in total;
- the model's expert blob is 50.3 GB, of which ~12.6 GB per card is resident (98% of
  the expert cache);
- measured per-card RX (61.9-78.4 GB) is of the same order as the whole activation
  stream, and 5-8x the measured TX.

So the dominant flow is **into** the cards, not between them. A P2P route would directly
change only the between-cards part (the 7-12 GB of TX) unless the engine can also serve
the dominant flow from a peer's resident copy instead of from host memory. Which of the
two candidate compositions those bytes are - activation staging, or the prompt-path
expert ring - is one line of code to check and is not derivable from PCIe counters.

## 4. Where the route is decided today (0.1.44.1)

- `src/prefill/prefill.cpp:1591` - `pp->p2p = false; // the mapped host route, whether
  or not the cards could reach each other (measured there)`. The comment above it
  (lines ~1560-1570) records that the measurements behind this were taken on 2x RTX
  3090, PCIe 4.0 x16, no P2P.
- P2P branches already exist in the same file (lines ~3508, 3673, 3736, 3755), and
  `src/core/peer_experts.cpp:97-110` does `canAccessPeer` / `EnablePeerAccess`, with
  `STRATA_PEER_NO_P2P=1` to pretend the pair has none.
- CLI: `--peer-device N` ("a second GPU as an adaptive expert-cache tier (rows over
  P2P...)"), `--peer-reserve-mib`, `--peer-slots`, `--peer-adapt-swaps`,
  `--peer-prefill-rows`. None of these is in this machine's launch command, and
  `STRATA_PREFILL_HELP` is unset in the working config - which is why the measured
  traffic is all PCIe.
- `set_peer_portable(o.peer_device >= 1)` (`src/program/generate.cpp:2310`) ties the
  Portable flag on mapped host buffers to a peer device being configured.

## 5. How to validate a P2P route here without DCGM

The DCGM NVLink byte counters are not obtainable on this host (no profiling module in
the installed DCGM 4.7.0, and neither the installed driver 580.178.04 nor the newest
driver in the archives, 595.91.07, exposes the NVML Sampling & Reporting entry points
that DCGM 4.x uses for them - checked by reading the symbol tables of both). The
PCIe counters are a sufficient substitute:

1. Run `nvlink-probe.py` (2.5 min, no privileges). If a P2P route is active, per-card
   `rxpci` during prefill must fall and prompt tok/s must rise. Idle must stay exactly
   zero, which is what makes the phase attribution trustworthy.
2. Preflight with `p2p-probe.cu` - capability matrix plus achievable peer bandwidth,
   runs with the model loaded, needs ~400 MiB free per card.
3. A short A/B (32K + 128K, 3 runs each plus a same-binary control) is ~25 minutes;
   the full 8-length community sweep is ~2 hours.

## 6. Building and testing on this machine

No root, no system CUDA: private CUDA 12.8 (V12.8.93) and private gcc-14 (14.3.0),
`-arch=sm_70`, `STRATA_EXPERIMENTAL_SM60=ON`. A build from a clean tag takes ~20
minutes; install plus rollback ~2 minutes; the previous tree is kept next to the
working one (`Strata-0.1.43.bak`). **A branch is enough for this platform - no
binaries need to be produced for it.**

## 7. Where the P2P route can and cannot be reached in 0.1.44.1

Peer copies in the engine are issued from one object only, `PeerPrefill`, and there are
exactly two places that construct it:

| Site | What it is | `p2p` |
|---|---|---|
| `prefill.cpp:1687-1702` `set_peer()` | the `--peer-device` feature (a second card as an expert tier) | `p2p = peer->p2p()` - P2P is used |
| `prefill.cpp:1582-1591` `set_stage_helper()` | the layer-split stage helper (`STRATA_PREFILL_HELP`) | **`p2p = false`, hard-coded** |

`--peer-device` is refused together with `--layer-split` - measured, not read from the
source: started with `--peer-device 1 --peer-reserve-mib 1024` on the working config the
engine exits at argument parsing with `strata generate: --peer-device cannot be combined
with --layer-split (use one or the other)`
([interconnect-experiments/peer-device-refusal.txt](interconnect-experiments/peer-device-refusal.txt)).
The remote expert caches (`--expert-cache-device1..3`, `--remote-expert-opt`) do not use
P2P either: `src/core/remote_experts.cpp` reads and writes through pinned host buffers
(`cudaMemcpyAsync … HostToDevice`).

So on a machine whose model has to be spread by layer-split - as this one does, 50.3 GB
of experts against 4x16 GB - **no combination of flags or environment variables in
0.1.44.1 makes the engine issue a single GPU→GPU copy.** Enabling NVLink here is a code
change, not a configuration: either allow `--peer-device` alongside a layer split, or
make the stage helper's route selectable. The second is small: the `p2p` field already
exists, defaults to `true` (`prefill.cpp:647`), and the copy code at
`prefill.cpp:3508, 3673, 3736, 3755` already handles `p2p = true`.

## 8. Follow-up run: ceilings, the shared uplink, and the two prompt-path knobs

Measured 2026-10-15 in a separate detached run
([nvlink-full-185.sh](nvlink-full-185.sh), outputs in
[interconnect-experiments/](interconnect-experiments/)), same engine build, same working
config. Each configuration was restarted, warmed up with a full pass over the length
set, then measured with three runs, with the PCIe phase probe run for each configuration
(`interconnect-experiments/<name>/phases.json`) and the engine's own start log captured
(`<name>/engine-log.txt`).

### 8.1 Ceilings, measured with the model unloaded

[p2p-full-probe.cu](p2p-full-probe.cu), 256 MiB buffers, 12 repetitions, model stopped -
the only way to measure all six pairs, because with the model loaded a context plus a
buffer needs ~400 MiB free and three cards have less:

| Path | One card at a time | All four at once |
|---|---|---|
| card → card (P2P), all six pairs, either direction | **38.8 GB/s** | — |
| host → card (pinned H2D) | 6.29 GB/s | **1.56 GB/s per card** |
| card → host (pinned D2H) | 6.58 GB/s | 1.65 GB/s per card |

A uniform 38.8 GB/s on every pair is what two bonded NVLink 2 links give (2 × 25.781 =
51.6 GB/s theoretical, 75% achieved); it is not reachable over this PCIe fabric. The host
numbers divide by four exactly (6.29 / 4 = 1.57): the 4:1 oversubscription is a measured
property of this machine, not an inference from the board's datasheet. Raw:
`interconnect-experiments/p2p-full.txt`, `interconnect-experiments/host-bw.txt`.

### 8.2 What the shared uplink actually carries during a long prefill

The phase tables earlier in this report give per-card maxima over a window, which do not
say whether the peaks coincide. [uplink-saturation-185.py](uplink-saturation-185.py)
records the raw 1 s series and sums the four cards per second, one 125351-token request,
working configuration:

| Metric | Value |
|---|---|
| RX summed over four cards, median second | 5.25 GB/s |
| RX sum, p95 second | 7.67 GB/s |
| RX sum, worst second | 9.33 GB/s |
| seconds at or above 6 GB/s | 33 of 75 |
| per-card peaks | 6.70 / 3.40 / 6.18 / 5.83 GB/s |
| TX sum, median / p95 / worst | 0.64 / 0.89 / 6.88 GB/s |

A Gen3 x8 link is 7.88 GB/s per direction. During a long prefill this machine runs at
about two thirds of its host link in a median second and at or above the nominal ceiling
in the worst ones. The worst second exceeds the nominal ceiling, which means either part
of that traffic moved card↔card through the on-board PEX8748 switch (which does not
consume the host link) or the driver's counters are not exact; this report does not
decide between the two, and the conclusion is the same either way: **a long prefill here
is close to the host-bandwidth limit, and the limit is in the host direction.** Raw
series: `interconnect-experiments/uplink-saturation.json`.

### 8.3 The stage-helper A/B

`STRATA_PREFILL_HELP=1` (the engine's own share rule) and
`STRATA_PREFILL_HELP=1` with `STRATA_PREFILL_HELP_FRAC=0.45` (the rule defeated, so the
helper works at every length), against the working configuration and a same-configuration
control. Reading speed, median of three runs, prompt tok/s:

| Prompt tokens | working | `HELP=1` | `HELP=1` repeat | `HELP` + `FRAC=0.45` | same, repeat | control |
|---|---|---|---|---|---|---|---|
| 120 | 203.9 | 206.5 | 207.0 | 206.7 | 207.2 | 205.8 |
| 2048 | 709.3 | 701.0 | 705.0 | 707.3 | 705.4 | 703.8 |
| 3072 | 817.8 | 815.7 | 814.8 | 820.4 | 823.4 | 816.9 |
| 4096 | 881.1 | 879.4 | 876.2 | 867.0 | 877.9 | 878.9 |
| 32768 | 1982.0 | 1982.5 | 1976.9 | 1975.8 | 1976.8 | 1976.7 |
| 131072 | 2500.8 | 2494.6 | 2501.6 | 2496.5 | 2496.6 | 2499.1 |

Every difference is inside the control band (±0.9%), and the largest single value
(-1.6% at 4096) is smaller than the spread between the two control sessions. **Neither
knob changes reading speed on this machine.**

What the probe does see is traffic. Card→host bytes during one 125351-token request,
summed over the four cards; each knob measured in two independent sessions:

| Configuration | RX total | TX total | TX per card |
|---|---|---|---|
| working | 270.5 GB | 35.2 GB | 7.2 / 6.8 / 13.1 / 8.1 |
| control | 265.0 GB | 29.1 GB | 7.2 / 7.1 / 6.4 / 8.4 |
| `HELP=1`, session 1 | 265.2 GB | **48.2 GB** | 16.7 / 12.7 / 10.5 / 8.2 |
| `HELP=1`, session 2 | 275.0 GB | 29.6 GB | 7.3 / 7.6 / 6.2 / 8.5 |
| `HELP` + `FRAC=0.45`, session 1 | 262.1 GB | **52.9 GB** | 18.8 / 13.5 / 13.1 / 7.5 |
| `HELP` + `FRAC=0.45`, session 2 | 269.0 GB | 35.8 GB | 14.0 / 6.8 / 6.9 / 8.0 |

The first `HELP=1` session also produced two stalls at 131072 tokens - wall 65.4 s and
72.5 s against 51.0-51.6 s everywhere else, with the reading rate unchanged (2493-2498
tok/s in every run). The repeated session reproduced neither the extra traffic nor the
stalls (50.9-51.0 s in all three runs).

**Neither knob reproduced its effect.** Both raised card→host traffic in their first
session and came back to the working level in the second one, and the sessions with extra
traffic are the sessions where something else went wrong as well. What is reproducible
across all six sessions is the absence of any speed change. The honest reading: on a
four-stage split the stage helper is a no-op for performance on this machine, and whether
it engages at all looks like a property of the session rather than of the configuration.

At 32768 tokens none of the configurations changed the traffic at all (TX totals 8.1-9.9
GB), which is consistent with the helper not arming there - but 32768 is far above the
~3277-token cutoff that `split_help_frac` implies from the source, so the rule is not
keyed on prompt length in the way the comment describes. That is a question for the
maintainer, not a finding.

### 8.4 What this adds up to

1. The NVLink path on this board is real, uniform and 25x the host path: 38.8 GB/s
   peer-to-peer against 1.56 GB/s per card for the host path when all four cards are
   streaming. Nothing in the driver or the topology gates it, and enabling peer access
   costs 0 MiB of VRAM.
2. The engine cannot reach it on a layer-split machine (§7), and the dominant traffic of
   a long prompt is host→card (§3, §8.2) - which NVLink cannot carry at all.
3. The one hand-over the engine can perform between cards (the stage helper's
   activations out and result rows back, ~122 MB per layer per 8192-token chunk at the
   engine's own constants) was forced on at long prompts and **cost no measurable time in
   either of its two sessions**, and it engaged at all in only one of them. Where it did
   engage it added 15-20 GB per card of PCIe traffic to a link that §8.2 shows is already
   used at ~67% of its ceiling in a median second - and that still changed nothing
   measurable. Either the transfers are hidden behind compute, or the helper's benefit
   cancels its cost here; these runs do not separate the two, and they do not show the
   helper paying for itself on this machine.
4. Therefore: on this machine, making the stage helper's route use NVLink would not buy
   measurable speed. It would matter for a configuration where the hand-over is what
   saturates the link - and the numbers needed to decide that are the ones in §8.1 and
   §8.2.

## 9. What is still open

1. Which route do the 61.9-78.4 GB per card take - the prompt-path expert ring, or
   activation staging? The volume matches both candidates and the counters cannot tell
   them apart; the code can, in one line.
2. What `split_help_frac` is actually keyed on, and what makes the helper engage in one
   session and not the next. Its source comment implies prompt length with a cutoff at
   ~3277 tokens; on this machine the helper did not change traffic at 32768 tokens in any
   configuration, and at 131072 it changed it in one session and not in the next.
3. Whether the worst-second sum above the nominal x8 ceiling is card↔card traffic through
   the on-board switch or imprecision in the driver's counters. A DCGM profiling build on
   a driver that exposes NVML Sampling & Reporting would settle it; neither exists here.
