# Community benchmark: Minisforum MS-01 + 4x NVIDIA Tesla V100-SXM2-16GB (NVLink carrier board)

Measured on 2026-10-14/15 by [Dmitry-B](https://github.com/Dmitry-B). This tests Strata 0.1.44.1 with qwen3.8-flash-next-iq3_s and a 262144-token context on a machine that is deliberately not a server: a **Minisforum MS-01 mini PC** whose single full-length slot (wired electrically as x8) drives a **four-slot SXM2 carrier board with NVLink** holding 4x Tesla V100-SXM2-16GB. Prompts are code-explanation text; greedy decoding, a 256-token output cap, three runs per configuration; TTFT measured over streaming. These are synthetic workloads; they do not establish general answer quality.

This report uses the same method, the same prompt generator and the same case set as the RTX 4090 reports of the same day ([2026-10-14-community-rtx4090-iq3xxs-200k-code](../2026-10-14-community-rtx4090-iq3xxs-200k-code/README.md)), so the two can be read side by side. It is also the first report from this machine in that format; the machine's own long-context protocol (needle/depth sweeps across 0.1.42, 0.1.43 and 0.1.44.1) lives outside this repository.

Short answer, with 0.1.43 re-measured in the same session and a same-binary control:

- **Reading long prompts: a small regression, and it is outside the noise band.** At 125011 / 185531 / 200015 / 260331 prompt tokens 0.1.44.1 reads 0.5% / 0.4% / 0.3% / 0.9% slower than 0.1.43, while the same-binary control moves by ±0.3%. At 4K-32K the deltas are +0.4…+0.6% against the same ±0.3% band, i.e. marginal.
- **Reading a short prompt: no change** (163-token prompt, 200.8 -> 203.7 tok/s, TTFT 0.83 -> 0.82 s), against a control band of +0.7%. The +31% that the same transition produced on a single-GPU RTX 4090 does not happen here, and the reason is visible in the engine's own start log - see *Why the two machines differ*.
- **Decode: nothing can be concluded on this machine.** The model stops by itself after ~95-110 tokens in almost every run of every configuration, including the generation-only case, so the lengths differ between runs and the column is not comparable. The reported medians move by +2.7…+10.5% with a control band of -4.4…+1.8%. The mechanism that gave +13.8% decode on the RTX 4090 (`STRATA_ROUTE_TAIL_SKIP`) is **auto-disabled on this machine** by the engine itself.
- **260331 prompt tokens (99.3% of the 262144 window) is served without errors**, TTFT 102.78 s, and reading speed at that length is ~1% below the machine's own peak at 185K - on both versions and in the control, so it is a length effect, not a version change.
- Recall 6/6 at 32K and 128K in both versions; draft acceptance did not drop.

## Hardware and software

**Host: a Minisforum MS-01 mini PC** (DMI: `sys_vendor` "Micro Computer (HK) Tech Limited", `product_name` "Venus Series", board "AHWSA", BIOS AHWSA.1.22 dated 2024-03-12), running Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic.

- CPU: 13th Gen Intel Core i9-13900H (Raptor Lake-P), 14 cores / 20 threads, 5400 MHz max, 400 MHz min, one NUMA node. Integrated graphics: Raptor Lake-P [Iris Xe Graphics].
- RAM: 96 GB installed (`MemTotal` 96228088 kB). **During these runs the host used about 16.7 GiB with the model loaded and peaked at 18368096 KiB = 17.5 GiB (18.8 GB) - under 19% of the installed memory.** RAM was never a constraint on this machine.
- Storage: model files, packs, the MTP pack, the control vector and the expert profiles are on a 2 TB NVMe drive (SOLIDIGM P44 Pro, `SSDPFKKW020X7`, 2048408248320 bytes, ext4, mounted at `/`). No network storage is in the measured path.
- Network: 2x Intel X710 10GbE SFP+, 2x Intel I226 (2.5GbE), MediaTek MT7922 Wi-Fi 6. Not used by the measured path.

**GPUs: 4x NVIDIA Tesla V100-SXM2-16GB on a carrier board, not in a server.** Board: **Spring V100 SXM2 4-GPU, model V100-SXM2-NVLINK_D V2.10** - an SXM2 mezzanine carrier board with NVLink 2.0 in NVIDIA's Hybrid Cube Mesh topology and a Broadcom **PEX8748** PCIe Gen3 switch on the host side, exposing two external SFF-8654 (Mini-SAS HD) host channels (CH1/CH2), each x8, configured as a single x16 or as 8+8 bifurcation. The four SXM2 modules are liquid-cooled through a 360 mm AIO loop, as the board's manual requires (with a >1500 W PSU). Measured at idle during the session: 37-39 °C, 53-67 W per card, 300.00 W power limit per card, 1530 MHz max SM clock. Driver 580.178.04. GPU clocks were not fixed.

**How the board is wired to the mini PC, and what it costs.** The board ships with a PCIe x16 -> 2 x SFF-8654 transition board (silkscreen `PCIE X16-2*X8 SFF 8654`), i.e. two x8 channels. **Only one cable is connected here, and the link runs as x8, because the MS-01's full-length slot is wired electrically as x8** - confirmed on the machine, not assumed: `/sys/bus/pci/devices/0000:00:01.0` (the CPU's PEG port) reports `max_link_width = 8`, and the negotiated link is 8.0 GT/s x8. The board's upstream port (PEX8748, `0000:01:00.0`) is x16-capable and negotiates **x8 Gen3**. Each GPU behind the switch (`03:00.0`-`06:00.0`) is x16-capable and negotiates **x8 Gen3**. So four cards, each with a Gen3 x8 downlink, share one Gen3 x8 uplink to the host: the host path is oversubscribed 4:1 (~7.9 GB/s shared). This is the physical reason the machine behaves so differently from a single-card box: the model is spread across four cards (`--layer-split auto`, e.g. `CUDA3 runs layers 37-47`), 98% of the experts are resident, and the host is not in the hot path of *generation*. The prompt path is a different story - see the probe below.

**NVLink is up and fully meshed, and the prompt traffic does not use it.** `nvidia-smi nvlink -s` reports **6 links per card, all at 25.781 GB/s** (12 links total), and `nvidia-smi topo -m` reports **NV2 between every pair of cards** - two bonded NVLink links per pair, which is the Hybrid Cube Mesh of the manual with all six channels of every V100 used: ~155 GB/s per direction per card, the ~300 GB/s bidirectional the board's documentation quotes.

The direct way to answer "which path does the engine use" is DCGM's NVLink byte counters, and **they are not obtainable on this host**: DCGM 4.7.0 is installed, but it has no profiling module here (`libdcgmmoduleprofiling` is absent from the system, and `dcgmi dmon -e 1012,1013` answers "This request is serviced by a module of DCGM that is not currently loaded") - both for an unprivileged host engine and for one started as root. So the probe measures the other side of the same question: **every byte that does not go over NVLink has to cross the card's PCIe link**, and those counters (`nvidia-smi dmon -s t`, 1 s resolution) work without privileges. Phases and raw data: [nvlink-probe-phases.json](nvlink-probe-phases.json), driver: [nvlink-probe.py](nvlink-probe.py).

| Probe phase | Per card, received | Per card, sent | Peak per card | Request |
| --- | --- | --- | --- | --- |
| idle, 12 s | 0 MB on all four cards | 0 MB | 0 | - |
| generation (85 -> 60 tokens) | 0-706 MB | 0-67 MB | 705 MB/s | 1.2 s |
| idle, 12 s | ~0 | 0 | 0 | - |
| prefill 31 375 tokens | 13 736-23 478 MB | 1 478-2 439 MB | 3 340-5 786 MB/s | 16.0 s |
| idle, 12 s | ~0 | 0 | 0 | - |
| **prefill 125 351 tokens** | **61 867-78 376 MB** | **6 891-11 931 MB** | **5 070-6 196 MB/s** | 50.2 s |
| idle, 12 s | 0 | 0 | 0 | - |

- The idle baseline is exactly zero on all four cards, so the phase attribution is reliable and the numbers are not background noise.
- **Generation barely touches the links** (<= 0.7 GB per card for a whole request), which is what `docs/MULTI_GPU.md` describes: "a token crosses from one card to the next once per verify window (a few hundred KB through pinned RAM), not twice per layer".
- **The prompt path is the PCIe-heavy one**: 62-78 GB into and 7-12 GB out of every card during one 125K-token request, sustained 1.2-1.6 GB/s per card with peaks of 5-6 GB/s against the ~7.9 GB/s ceiling of the Gen3 x8 link this machine has. At those peaks the link is close to saturated.
- The engine's peer-to-peer prompt routes are **not enabled in this configuration**: the stage helper is behind `STRATA_PREFILL_HELP` (not set in the working config) and `--peer-device` is not in the launch command; and the helper's own code takes the mapped-host route regardless of whether P2P exists - `src/prefill/prefill.cpp:1591`: `pp->p2p = false; // the mapped host route, whether or not the cards could reach each other (measured there)`. So NVLink is not the path this workload uses, and the code says the host route was chosen because it measured better.
- What these numbers do **not** license is the claim that NVLink carries exactly zero bytes - the direct counters are unavailable on this host. Every byte the engine is known to move during a prompt is accounted for on PCIe.
- **P2P works on this board and is not used.** Measured with the model loaded ([p2p-probe.cu](p2p-probe.cu), [results](p2p-probe-results.json)): `cudaDeviceCanAccessPeer` is 1/1 on all six pairs with `AccessSupported=1`, `NativeAtomicSupported=1`, `PerformanceRank=0`, and NVML reports `NVLINK=OK` for every pair. Peer copies between GPU1 and GPU2 run at **38.4 GB/s idle and 29.4-29.9 GB/s while the server is doing a 31375-token prefill** - a bandwidth this PCIe fabric cannot produce (7.9 GB/s per card, one shared x8 uplink), which is also indirect evidence that an enabled P2P route really would use NVLink here. Enabling peer access costs **0 MiB** of device memory, so it is not blocked by the 256-1350 MiB of VRAM that is free with the model loaded.
- One observation for the maintainer: the comment behind `pp->p2p = false` ("the mapped host route, whether or not the cards could reach each other") sits under measurements taken on **2x RTX 3090, PCIe 4.0 x16 each, no P2P** (`src/prefill/prefill.cpp:1560-1570`). On a board where the cards are bonded by NVLink at ~155 GB/s per direction, where the driver reports P2P available on every pair and where a peer copy measured 29-38 GB/s against ~2 GB/s of effective host bandwidth, that trade-off is a different one. The engine has not been tested with the P2P route on this machine - the route is hard-coded off at that line, so no environment variable reaches it. A platform dossier (capability matrix, measured peer bandwidth under load, the byte budget of the prompt path, the code map, and a validation protocol that works without DCGM counters) is in [nvlink-platform-dossier.md](nvlink-platform-dossier.md).
- Why it matters: with the host path oversubscribed 4:1, it is the prompt path that pays for it. That is a plausible mechanism for the two long-prompt observations above (the -0.3…-0.9% regression and the ~1% drop from 185K to 260K), but these runs do not separate it - it is a hypothesis here, not a result.

### Follow-up run: ceilings, the shared uplink, and the two prompt-path knobs

A second detached run (2026-10-15) measured the limits of both fabrics and then tested the
two knobs that control the prompt path. Full tables, raw series and scripts:
[nvlink-platform-dossier.md](nvlink-platform-dossier.md) §8, data in
[interconnect-experiments/](interconnect-experiments/).

- **Ceilings, model unloaded** ([p2p-full-probe.cu](p2p-full-probe.cu)): card↔card **38.8 GB/s on all six pairs, either direction**; host→card 6.29 GB/s one card at a time, **1.56 GB/s per card when all four stream at once** (6.29 / 4 = 1.57 - the 4:1 oversubscription measured, not inferred). NVLink is 25x the host path here.
- **What the shared x8 carries during a 125351-token prefill** ([uplink-saturation-185.py](uplink-saturation-185.py)): summing the four cards per second, median 5.25 GB/s, p95 7.67 GB/s, worst second 9.33 GB/s, 33 of 75 seconds at or above 6 GB/s - against a nominal Gen3 x8 ceiling of 7.88 GB/s per direction. A long prefill on this machine runs at roughly two thirds of its host link in a median second.
- **`--peer-device` with `--layer-split` is refused by the engine itself**: `strata generate: --peer-device cannot be combined with --layer-split (use one or the other)`. Together with the two constructors of `PeerPrefill` (`set_peer()` for `--peer-device`, `set_stage_helper()` with `p2p = false` hard-coded), that closes the question: in 0.1.44.1 no flag or environment variable makes a layer-split machine issue a GPU→GPU copy.
- **`STRATA_PREFILL_HELP` and `STRATA_PREFILL_HELP_FRAC` do not change reading speed** - measured in two sessions each, with a control, at 120 / 2048 / 3072 / 4096 / 32768 / 131072 tokens: every difference inside ±1.6%, all of them inside the control band. Where the helper engaged it raised card→host traffic by 15-20 GB per request and cost no measurable time; in the repeat session it did not engage at all. The one anomaly seen alongside it (two 14-21 s stalls at 131072 tokens with the reading rate unchanged) did not reproduce.

- No system CUDA is installed: the build uses a **private CUDA 12.8 toolkit** (`nvcc release 12.8, V12.8.93` at `/home/dgbox/strata185/cuda-12.8`) and a **private gcc-14** (Ubuntu 14.3.0-14ubuntu1, unpacked from .deb). `engine-cuda12/BUILD.json` records `"source": "local"`, `archs: [70]`, `STRATA_EXPERIMENTAL_SM60=ON`. The release publishes Windows archives only, so a source build is the only path here; both versions in the comparison were built on this machine with the same private toolchain.
- The source trees carry no git metadata (release archives), so the commit ids in `env.json` are recorded from the tags (`v0.1.44.1` = `296b59d2`, `v0.1.43` = `7695859a`) rather than read from a working tree.
- Background workloads: stock Ubuntu services. **Observer caveat, unlike the RTX 4090 reports:** this machine also serves the agent that drove the measurements, so the agent's own inference traffic goes through the same server. That traffic is present in the measured run, the control run and the previous-version run alike; it is the main reason the noise band here is wider than the pure-numeric spread inside a case.
- Engine binaries used: 0.1.44.1 md5 `c9a9737d49cf210f33ce7f6c98d4dd5c`, 0.1.43 md5 `d79b0506a08d045e6562f9e15755071b`.

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s (IQ3_S quantization of Qwen3.8-Flash-Next-GSQ-RCO); GGUF filenames, sizes and modification times are in env.json (no hashes).
- Context 262144; INT8 KV; `--mmap-experts`; GPU vision; `--vram-reserve-mib 512`; `--pool-workers 12`; `--spec 4` with an MTP draft pack; `draft_vocab=cyrillic`.
- Expert cache sizing from the start log: `layer split: CUDA3 runs layers 37-47, expert cache 5442 slots (11.37 GiB), 5442 of its 5632 profiled pairs`, `token graph hit path: 24088 resident experts, decided on the device`, `layer split: 98% of the experts resident, the prompt path's streamed ring 96 slots`, and the prompt path borrows cache slots from the other three cards (`the prompt path borrows 1935 CUDA0 cache slots (3.40 GiB)`, and 1841/1752/1634 slots from CUDA1/2/3).
- **Decode expert cache hit rate in these runs: 99.9-100.0%** (e.g. 272357 hits / 272550 lookups). This is the single most important number for reading the rest of the report: on this machine almost nothing is streamed, which is why two of the release's new defaults do nothing here.
- Experimental speed projection is enabled (`--control-vector-scaled …/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer`), and an online-learned expert profile is reused between restarts (`--expert-profile …/data/expert-profile.bin --expert-profile-save …/data/expert-profile-learned-185-iq3s.bin --expert-profile-save-every 10`). Both were also in use for the previous version.
- The server config sets `env: {"STRATA_STAGE_TRIM": "1"}`; that key was therefore present in every run below, including 0.1.43 and the opt-in run.
- **The engine's auto prompt chunk is `prompt chunk auto: 8192 tokens, a 96-slot ring` in both 0.1.44.1 and 0.1.43** - the same chunk size as on the RTX 4090 machine, which is what makes the two reports method-comparable.
- The two trees' server configs were compared before the swap and are **byte-identical**; the model files, packs, control vector and learned profile live outside the trees and are the same files for both versions.

```text
/home/dgbox/strata185/Strata/engine-cuda12/strata --serve --pack /home/dgbox/strata185/Strata-data/packs/iq3_s --native /home/dgbox/strata185/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf /home/dgbox/strata185/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile /home/dgbox/strata185/data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp /home/dgbox/strata185/Strata-data/mtp/rt --max-context 262144 --kv int8 --mmap-experts --vision --vram-reserve-mib 512 --control-vector-scaled /home/dgbox/strata185/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --spec-min-p 0.50 --pool-workers 12 --layer-split auto
```

Full server config: [config.json](config.json), environment details: [env.json](env.json).

## Method

Identical to the RTX 4090 reports of the same day, with one extension (lengths out to the 262144 window):

- Warm-up: a **full sweep** (4K … 260K) run first and discarded, then the benchmark's own 4K warm-up. Required: the first pass after a start is dominated by the cold expert/page cache.
- Every measured prompt carries a random marker, so the prompt-prefix cache is not reused; the table's reused column reports the actual reused counts from the engine (0 in every run here, in both versions).
- Throughput comes from the engine's timing fields (prompt_per_second / predicted_per_second). TTFT is the time to the first non-empty streaming delta, ignoring keep-alives. Wall is the whole request time measured at the client.
- temperature=0, reasoning_effort=none, 256-token output cap (1024 for the generation-only case).
- Memory: peak VRAM sampled every 2 s **per card** (this machine has four), plus a start snapshot.
- Three runs per case, medians reported. The whole session (warm-up, measurement, control, previous version, opt-in) ran as one detached process from start to finish, so nothing was interleaved by the driver.
- The measurement script is a local script (not part of this repository); it is the same script that produced the RTX 4090 reports, adapted only in provenance collection (four GPUs, and commit/branch taken from environment variables because the trees have no git metadata). Available on request.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT s median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 4096-prompt | 3971 | 0 | 181/182/105 | 3 | 881.6 (range 880.8-883.0, n=3) | 101.0 (range 99.2-113.0, n=3) | 4.53 s (range 4.52-4.53, n=3) |
| 16384-prompt | 15667 | 0 | 110/105/102 | 3 | 1548.5 (range 1545.5-1551.1, n=3) | 117.5 (range 110.8-118.9, n=3) | 10.15 s (range 10.14-10.17, n=3) |
| 32768-prompt | 31307 | 0 | 103/93/103 | 3 | 1985.3 (range 1979.3-1995.6, n=3) | 111.8 (range 110.4-111.8, n=3) | 15.81 s (range 15.73-15.86, n=3) |
| 66560-prompt | 63539 | 0 | 105/93/103 | 3 | 2345.3 (range 2343.6-2350.5, n=3) | 109.2 (range 105.3-109.7, n=3) | 27.16 s (range 27.1-27.18, n=3) |
| 131072-prompt | 125011 | 0 | 105/104/105 | 3 | 2490.7 (range 2488.7-2495.8, n=3) | 102.5 (range 100.8-102.5, n=3) | 50.29 s (range 50.19-50.32, n=3) |
| 194560-prompt | 185531 | 0 | 105/105/102 | 3 | 2560.5 (range 2553.3-2564.2, n=3) | 99.8 (range 97.5-103.1, n=3) | 72.59 s (range 72.48-72.79, n=3) |
| 209715-prompt | 200015 | 0 | 104/104/94 | 3 | 2551.7 (range 2549.2-2552.5, n=3) | 101.8 (range 101.6-109.4, n=3) | 78.52 s (range 78.49-78.59, n=3) |
| 273000-prompt | 260331 | 0 | 98/164/93 | 3 | 2536.9 (range 2534.7-2537.1, n=3) | 94.6 (range 85.2-100.7, n=3) | 102.78 s (range 102.77-102.87, n=3) |
| gen-only | 163 | 0 | 1024/104/104 | 3 | 203.7 (range 202.4-204.3, n=3) | 106.0 / 123.2 / 119.5 (lengths differ) | 0.82 s (range 0.82-0.83, n=3) |

- Total latency (client, wall) medians: 4096-prompt 6.32 s; 16384-prompt 11.07 s; 32768-prompt 16.69 s; 66560-prompt 28.09 s; 131072-prompt 51.30 s; 194560-prompt 73.56 s; 209715-prompt 79.49 s; 273000-prompt 103.83 s; gen-only 1.68 s (the one run that reached 1024 tokens took 10.47 s).
- Draft acceptance (accepted/total, all runs of a case): 4096-prompt 314/413 (76.0%); 16384-prompt 232/262 (88.5%); 32768-prompt 221/258 (85.7%); 66560-prompt 219/259 (84.6%); 131072-prompt 226/282 (80.1%); 194560-prompt 225/272 (82.7%); 209715-prompt 222/264 (84.1%); 273000-prompt 242/313 (77.3%); gen-only 824/1064 (77.4%).
- Memory: peak VRAM per card 15980 / 15386 / 15034 / 16128 MiB of 16384 - i.e. **256 MiB free on CUDA3**, unchanged from the start snapshot. RAM: ~16.7 GiB in use with the model loaded, peak 18368096 KiB (17.5 GiB) of the 96 GB installed - under 19%, with the host's memory nowhere near its limit. The engine's own sizing line says `CUDA3: 11.96 GiB free of 15.77, room for experts 11.37 GiB`. Nothing spilled, but the last card is the tight one.
- Every run with its draft statistics: [runs.json](runs.json).
- Recall check (needle): [needles.json](needles.json) - 6/6 found at 32K and 128K, depths 10/50/90.
- The model stops by itself after ~95-110 tokens on this prompt text in nearly every run, so the 256-token cap is rarely reached. This is a property of the IQ3_S model on repetitive text, not of the request path; it is also why the decode column is not usable here (see *Comparison*).

## Comparison with engine 0.1.43 on the same machine

Same machine, same four cards, same model, byte-identical server config, same launch command, same learned expert profile, same prompt chunk (8192 tokens / 96-slot ring). Only the version changed: 0.1.43 (tag `v0.1.43`, commit `7695859a`) -> 0.1.44.1 (tag `v0.1.44.1`, commit `296b59d2`), both built on this machine with the same private CUDA 12.8 / gcc-14 toolchain.

Two reference runs are included so the comparison is self-contained:

- [runs-0.1.43-previous.json](runs-0.1.43-previous.json) - the previous version, **measured in this same session**: the whole tree was swapped (`Strata` <-> `Strata-0.1.43.bak`), the server restarted (health in 60 s), the same warm-up sweep run and discarded, then the same sweep measured, then the tree swapped back;
- [runs-0.1.44.1-control.json](runs-0.1.44.1-control.json) - a full repeat of the sweep on **the identical 0.1.44.1 binary**. This is the noise band of the method on this machine.
- Recall of the previous version: [needles-0.1.43.json](needles-0.1.43.json) - 6/6.

| Configuration (actual prompt tokens) | Prompt tok/s 0.1.43 -> 0.1.44.1 | TTFT s 0.1.43 -> 0.1.44.1 | Decode tok/s 0.1.43 -> 0.1.44.1 (lengths differ) | Noise band (same binary): prompt / decode |
| --- | --- | --- | --- | --- |
| 3971 | 876.2 -> 881.6 (+0.6%) | 4.56 -> 4.53 | 100.6 -> 101.0 | -0.2% / -1.2% |
| 15667 | 1542.6 -> 1548.5 (+0.4%) | 10.20 -> 10.15 | 106.3 -> 117.5 | -0.1% / -3.2% |
| 31307 | 1973.2 -> 1985.3 (+0.6%) | 15.91 -> 15.81 | 103.8 -> 111.8 | -0.2% / -0.9% |
| 63539 | 2346.8 -> 2345.3 (-0.1%) | 27.14 -> 27.16 | 101.2 -> 109.2 | +0.1% / +0.8% |
| 125011 | 2502.1 -> 2490.7 (**-0.5%**) | 50.05 -> 50.29 | 97.8 -> 102.5 | +0.2% / -2.6% |
| 185531 | 2571.3 -> 2560.5 (**-0.4%**) | 72.27 -> 72.59 | 97.2 -> 99.8 | -0.1% / +1.8% |
| 200015 | 2560.3 -> 2551.7 (**-0.3%**) | 78.25 -> 78.52 | 93.0 -> 101.8 | -0.0% / -2.5% |
| 260331 | 2559.4 -> 2536.9 (**-0.9%**) | 101.89 -> 102.78 | 90.5 -> 94.6 | +0.3% / -3.1% |
| 163 (gen-only) | 200.8 -> 203.7 (+1.4%) | 0.83 -> 0.82 | 113.1 -> 106.0/123.2/119.5 | +0.7% / -4.4% |

### What is a change and what is not

- **Long-prompt reading: a small real regression.** Four lengths (125011, 185531, 200015, 260331) all move down, -0.3…-0.9%, and the same-binary control at those lengths stays within ±0.3%. The within-case ranges are tiny and do not overlap (260331 tokens: 2534.7-2537.1 on 0.1.44.1, 2557.0-2565.5 on 0.1.43), so the sign is not a sampling artifact. TTFT is 0.2-0.9 s higher at those lengths. This is the same direction as this machine's own long-context protocol reports for the same transition (-0.4…-1.9% prefill), at the low end of that range.
- **Short and medium prompts: no change worth reporting.** +0.4…+0.6% at 4K-32K against a ±0.3% band is marginal; the honest reading is "the same speed". The 163-token case moves +1.4% against a +0.7% band - also not a change.
- **Decode: not concluded.** The model ended at different lengths in nearly every run of every configuration in all three passes (for example at 3971 tokens: 181/182/105 on 0.1.44.1, 112/255/256 on 0.1.43, 105/186/162 in the control), so no decode delta at those rows means anything, and the control band for the column is -4.4…+1.8%. The generation-only case, which is the one designed to be comparable, reached the 1024-token cap in **one run out of nine** on this machine (106.0 tok/s on 0.1.44.1, 107.8 tok/s in the opt-in run below); the other runs stopped at ~95-104 tokens. With one sample per configuration there is no interval to compare, so the table lists the lengths instead of drawing a conclusion.
- **Reading speed is not flat to the top of the window, and that is not a version effect.** Both versions and the control peak at ~185K (2560-2571 tok/s) and are ~1% lower at 260331 (2536.9 / 2559.4 / 2543.5 respectively). The shape is the same in all three passes, so it belongs to the length, not to the release. On the single-card RTX 4090 machine the same sweep was flat up to 200K.
- Recall: 6/6 at 32K and 128K in both versions.
- Draft acceptance moves by 1-4 percentage points between the versions at most cases (4096-prompt 76.0% vs 78.1%; 32768-prompt 85.7% vs 87.2%; 260331 77.3% vs 81.2%), but the **same-binary control moves by up to 4.0 pp on the same cases** (4096-prompt 73.6% in the control vs 76.0% measured; 260331 81.3% vs 77.3%), and the generated lengths differ between passes, so no acceptance conclusion is drawn. All three passes stay in the 73-89% band depending on length.
- Answers are **not** bit-identical to 0.1.43 across the whole span in the sense that matters here: the 0.1.41/0.1.42 defaults that change answers are listed below, and both versions already had them.

## Why the two machines differ: the adaptive defaults

The RTX 4090 report of the same day attributes two large gains to two defaults that turned on in 0.1.41 and 0.1.42. Neither gain exists on this machine, and the engine's own start log says why. Both logs are quoted verbatim:

- **`STRATA_ROUTE_TAIL_SKIP` (the +13.8% decode gain on the 4090) is auto-disabled here.** RTX 4090 start log: `STRATA_ROUTE_TAIL_SKIP=7 is ON (default)`. This machine's start log: `STRATA_ROUTE_TAIL_SKIP is off by default: 98.0% of the experts are in the GPU cache, so few are missed and skipping them costs more draft acceptance than it saves; STRATA_ROUTE_TAIL_SKIP=7 turns it on`. The decode expert cache hit rate measured in these runs is 99.9-100.0%, consistent with that decision. So the decode improvement of 0.1.42 is, by design, a single-card / streaming-card improvement; a four-card machine where the experts are resident is not expected to see it, and this report confirms it does not.
- **The CPU prefill share (the +31% short-prompt gain on the 4090) is not armed here, and forcing it changes nothing.** The 4090 start log prints `prefill: the CPU share is ON by default for prompt chunks below 1024 tokens`; this machine's log never prints that line. `include/strata/prefill/share_rules.hpp` explains the rule: *"With every expert resident in the GPU's cache nothing is streamed, so there is nothing to take"*, and the code comment in `src/prefill/prefill.cpp` states the default is armed only for "CUDA builds, one GPU, no batch slots" - this is a four-GPU `--layer-split auto` build. To test it directly rather than by inference, the working config was copied with one extra key, `env: {"STRATA_PREFILL_CPU_SHARE": "1"}` (an explicit value is honoured whatever the rule says), the server restarted, and the short cases re-measured: raw runs in [runs-0.1.44.1-cpu-share-on.json](runs-0.1.44.1-cpu-share-on.json). Result: 163-token prompt 208.5 tok/s vs 203.7 in the default build (against a ±0.7% band - a small move, nothing like +31%), and 4K reading 879.1 vs 881.6 (no change). The engine printed no "CPU share" line in that run either. The conclusion is the one the source rule predicts: on a machine where 98% of the experts are resident, there is nothing to hand to the CPU pool.
- The corollary for the release notes: the "+31% short prompts / +10-20% decode" headline of 0.1.41-0.1.42 is real on machines that stream experts, and both features correctly stay out of the way on a machine that does not. This report is the negative half of that claim.

## Correctness and limitations

- The interconnect probe could not read NVLink byte counters: DCGM 4.7.0 on this host has no profiling module (`dcgmi dmon -e 1012,1013` -> "module … not currently loaded"), and that holds for a root host engine too. This is not a missing package: DCGM 4.x has no profiling module at all (the module packages in NVIDIA's own repository are config, diag, health, introspect, nvswitch, policy, sysmon), and in 4.x those counters come from the NVML Sampling & Reporting API, which **neither the installed driver 580.178.04 nor the newest driver in the archives, 595.91.07, exposes** - checked by reading the symbol tables of both. The probe therefore measures PCIe throughput per card, which bounds the question from the other side; the statement "NVLink carries nothing" is not made.
- Speed measurements do not establish general answer quality. The recall check passed 6/6 at 32K and 128K across depths 10/50/90 in both versions.
- The decode column of this method is unusable on this model/machine combination: the model stops early at unpredictable lengths, and the one fixed-length case reached its cap in one run out of nine. A future report on this machine should either force a fixed output length (e.g. `ignore_eos`) or use a prompt that does not trigger early stopping if decode is the subject.
- This machine serves the agent that drove the measurements, so agent traffic overlaps the runs. It affects the measured run, the control and the previous-version run alike, and the control band (±0.3% for prompt reading) shows it is small, but it is not zero.
- Prompts are synthetic (repeated text with a random marker); real workloads will show different prefix reuse and draft acceptance. Only the code-style prompt variant was run here; the Russian variant from the RTX 4090 reports was not repeated on this machine.
- The engine is built with a private CUDA 12.8 toolkit for `sm_70` with `STRATA_EXPERIMENTAL_SM60=ON`, while the release binaries are built with CUDA 13.0 for newer architectures. Comparisons inside this report are unaffected (both versions were built the same way on this machine), but they are not a comparison against published Linux binaries - there are none for this tag.
- VRAM headroom on this configuration is 256 MiB on the fourth card at 260K tokens. Nothing failed, but this is the constraint to watch if the context or the expert cache grows.
- Not tested here: `--conversation-cache-mib`, `--kv k8v4`, `STRATA_IO_PREFETCH`, `--batch` / `--batch-groups`, `STRATA_SPEC_GUMBEL`, and the pinned-expert runs from docs/RESEARCH_RUNS.md.
