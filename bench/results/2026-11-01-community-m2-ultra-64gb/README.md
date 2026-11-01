# Community benchmark on Apple M2 Ultra 64 GB (headless, Metal)

Measured on 2026-11-01 by `yoffy`. Strata **0.1.55** (commit `c62d5e3a`), engine
**0.1.55-metal**, Qwen3.8-Flash-Next **IQ3_S**, 262,144-token context, vision on, experts
streamed from SSD into the Metal expert arena.

The report is an A/B of one setting: `STRATA_METAL_LEAVE_SHARE` unset (the default
`kLeaveShare = 0.40`, `metal/src/engine/arena_budget.h`) versus `0.25`. On this machine the
default cap is set by the 40% share, not by the GPU working set or the file-cache floor;
0.25 moves the cap to the `filecache_min + 12 GB` term. The arena grows 29.5 -> 37.9 GB
and **every speed arm improves 15-24%**, with swapouts zero across a 118-minute resident
soak (mostly idle, containing both runs) and compressor occupancy at the level the M5
Max clean runs report. Main limitations: one machine, one suite run per
configuration, speed only (no correctness checks), and a configuration that is not
stock - headless operation, a raised `iogpu.wired_limit_mb`, and the model on an
external Thunderbolt SSD; the result may not carry to a desktop Mac at the default
wired limit or to internal storage.

## Hardware and software

- **GPU:** Apple M2 Ultra (arm, 24 cores / 24 threads), 64.0 GiB unified memory.
  `metal/probe/mtl_probe.cpp` on this machine: `maxBufferLength` 38.88 GiB,
  `recommendedMaxWorkingSetSize` 61.00 GiB, page size 16 KiB.
- **System settings that are not stock** (both apply to every run here):
  - `sysctl iogpu.wired_limit_mb = 62464` (61 GiB, raised from the default ~3/4 of RAM).
    The probe's `recommendedMaxWorkingSetSize` equals this value exactly, so the Metal
    device property reflects the IOKit wired limit on this machine.
  - **Headless:** the machine is used over SSH only; no GUI login session, so the macOS
    and desktop daemons the `arena_keep` floor assumes are not running (observed 2-3 GiB).
  - `sysctl vm.vm_page_filecache_min = 434049` pages = 6.6 GiB (10.3% of RAM; the M5 Max
    report in `arena_budget.h` records 27% there).
- **Storage:** internal `APPLE SSD AP1024Z` (1 TB), and a 4 TB external Thunderbolt SSD
  (reported as `MS950G70PCIe4 4096G`, a PCIe 4 NVMe in an external enclosure) - **the
  model files and the expert arena's reads live on the external drive**, so the SSD
  streaming in both A and B goes over Thunderbolt. The A/B delta is tied to that
  storage; the Thunderbolt link generation and bandwidth were not measured.
- **OS:** Darwin 27.2.0. **Strata** project 0.1.55, commit `c62d5e3a`; **engine**
  0.1.55-metal, source setup (`uv run ./setup.sh --model IQ3_S --vision yes --context
  262144`).
- **Background:** the machine serves an interactive agent session at other times; during
  each suite run the benchmark's own server was the only GPU user. Nothing else was
  isolated or pinned.

## Model and configuration

- Model id `qwen3.8-flash-next-iq3_s`; shards `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`
  and `...-00002-of-00002.gguf` (the engine log says "about 55 GB" of experts). Repository
  and revision not recorded.
- Engine arguments as configured (identical in A and B):

```text
strata --pack iq3_s --native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
  --ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
  --expert-profile expert-profile.bin --expert-cache 64 --prefill auto --spec 2
  --suffix-draft 0 --max-context 262144 --mmap-experts --vision
```

- Config environment: `STRATA_METAL_STREAM=1` (streaming forced; auto would also stream:
  55 GB of weights plus the 12 GiB overhead exceed the 44.8 GiB `resident_cap` computed
  for this machine).
- **A:** no `STRATA_METAL_LEAVE_SHARE` (default 0.40).
- **B:** `STRATA_METAL_LEAVE_SHARE=0.25` set in the shell before `tools/strata_bench.py`;
  the benchmark starts the server (`start_server` passes no `env=`, so the child inherits
  it) and the engine inherits it in turn.

The engine's own plan lines (from the interactive server log of the same configuration;
the `engine-log-*.txt` excerpts of the bench runs carry only the first lines):

```text
A: planned resident footprint 41.2 GB = arena 29.5 + 11.2 held before it + 0.5 reserved
   for the n-gram table; cap 41.2 GB = the lesser of the GPU working set 65.5 GB and the
   RAM 68.7 GB less 27.5 GB left to macOS and its file cache
A: expert arena 29.54 GB, 13453 slots in 3 size classes, 13453 experts read from the
   profile's top (24576 ranked) ...; 4 buffers
A: 29.5 GB of the arena wired (mlock) in 1227 ms

B: planned resident footprint 49.6 GB = arena 37.9 + 11.2 held before it + 0.5 reserved
   for the n-gram table; cap 49.6 GB = the lesser of the GPU working set 65.5 GB and the
   RAM 68.7 GB less 19.1 GB left to macOS and its file cache
B: expert arena 37.94 GB, 17268 slots in 3 size classes, 17268 experts read from the
   profile's top (24576 ranked) ...; 4 buffers
B: 37.9 GB of the arena wired (mlock) in 10796 ms
```

Full blocks, with the fill and after-fill lines, are in
[engine-stream-lines.txt](engine-stream-lines.txt). The arena holds 13,453 of the
profile's 24,576 ranked experts in A and 17,268 in B (55% -> 70% of the top-ranked set),
which is the mechanism behind the speed change.

One later restart with the same 0.25 planned a cap of 51.5 GB (arena 39.8) because
`vm_page_filecache_min` had dropped to about 5 GB by then: the floor term of the cap is
dynamic, so the same setting can plan different sizes at different times.

## Method

- `python tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges; every
  prompt carries a unique tag; one warm-up chat and a 1K prompt discarded; model loading
  not timed). The benchmark started a **fresh server per configuration** from the install
  config (cold engine, cold caches), same machine, same build, same day; the server was
  stopped between runs.
- Comparison: `python tools/strata_bench.py --compare <A> --compare-only <B>`.
- Memory was watched with `vm_stat` and `sysctl vm.swapusage` (page = 16 KiB): during
  interactive use after B's fill, and as a 118-minute soak at 30 s intervals that covered
  both bench runs ([soak-vmstat.log](soak-vmstat.log)).

## Results

| Metric | A (default) | B (0.25) | change |
| --- | ---: | ---: | ---: |
| chat decode tok/s | 26.30 | 31.70 | +20.5% |
| chat first token (s) | 1.31 | 1.07 | -18.3% |
| 4K prompt tok/s | 370.30 | 426.90 | +15.3% |
| 4K first token (s) | 10.89 | 9.45 | -13.2% |
| 16K prompt tok/s | 351.70 | 435.70 | +23.9% |
| 16K first token (s) | 45.70 | 36.89 | -19.3% |
| agent cache hit % | 69.44 | 69.44 | -0.0% |
| agent re-read tokens/session | 5.00 | 5.00 | +0.0% |
| agent turn first token (s) | 11.63 | 8.50 | -26.9% |
| agent decode tok/s | 28.00 | 32.20 | +15.0% |

The agent cache hit and re-read counts are identical in both runs, so the conversation
reuse behaved the same way in A and B; the speed difference is not cache reuse. The whole
suite took 452 s in A and 372 s in B. Per-run medians and ranges for each configuration
are in [run-A.md](run-A.md) and [run-B.md](run-B.md); per-request data in
[results-A.json](results-A.json) and [results-B.json](results-B.json).

### Memory observations (config B unless noted)

- After the fill: the process holds 49.2 GB (40.5 GB resident), planned 49.6 GB, cap
  49.6 GB. Under A the same machine held about 43 GiB for the engine (observed during
  interactive use, not a suite measurement).
- During the arena fill: compressor occupancy rose 219,125 -> 321,223 pages (3.3 -> 4.9
  GiB), Compressions +10.2M, **Swapouts +0**.
- Steady state afterwards: occupancy +4,005 pages per sample interval, Compressions
  +0.67M, **Swapouts +0**; `vm.swapusage` used 3 MB. The occupancy level (0.32M pages) is
  the same band the M5 Max clean runs report in `arena_budget.h` (~0.3M pages, swapouts
  0); the panic signatures there are an order of magnitude higher (3.9M pages, +29,000
  swapouts).
- **Soak** ([soak-vmstat.log](soak-vmstat.log), `vm_stat` + `vm.swapusage` every 30 s,
  236 samples, 19:48:33 - 21:46:12, 118 min): the engine was resident the whole window
  and mostly idle inside it - the two bench runs (~14 min of real work) and intermittent
  interactive requests, no sustained load. For the failure mode the soak targets (macOS
  reclaiming the unwired part while the arena holds its mlock: the GPU's own wiring ends
  seconds after the last command buffer, `arena_budget.h`), idle residency is the main
  condition. Result: **Swapouts +0**, swap usage flat at 215 MB, compressor occupancy
  stable at 322k - 328k pages with one transient sample of 564k at 21:41 (a later engine
  restart, back at the next sample), Compressions +5.75M / Decompressions +5.73M from
  the bench and interactive churn.
- Caveat on the same counters: cumulative Swapouts were 22,492 at the earlier interactive
  sample and 257,340 at the soak start - about 235k pages (3.7 GB) were swapped out at
  some point before the soak window, and the boot-cumulative counters cannot say which
  session did it (the day included several restarts, one planning a 51.5 GB cap). The
  soak window itself, including both measured bench runs, added none.

## Correctness and limitations

- **Speed only.** No needle, recall or task check was run: correctness is not measured.
- One suite run per configuration (3 requests per test inside the suite); the whole suite
  was not repeated. The agent-session token counts differ a little between runs because
  the conversation feeds on the model's own replies.
- The steady-state counters before the soak come from interactive use after the fill.
  The soak (118 min, mostly idle, containing both bench runs) added no swapouts; the
  ~235k swapouts recorded earlier the same day are not attributed to a session (see
  Memory observations).
- Sustained load over tens of minutes was **not tested**: the heaviest stretches are the
  two ~6-7 minute bench runs. Concurrent disk or CPU work was also not tested.
- The configuration is a headless Mac with `iogpu.wired_limit_mb` raised to 61 GiB. B
  leaves 49.2 GB of 64 GiB (72%) as the process footprint; nothing ran concurrently
  during the measurements, and the `arena_budget.h` cliff data (clean at 60% footprint,
  panics at 70% under a concurrent 40 GB write, on a desktop M5 Max) says concurrent load
  is the risk this machine did not test.
- The plan and `mlock` lines quoted above come from the interactive server log of the
  same configuration; the bench runs' own excerpts carry only the first log lines. The
  bench runs' after-fill lines were not kept.
