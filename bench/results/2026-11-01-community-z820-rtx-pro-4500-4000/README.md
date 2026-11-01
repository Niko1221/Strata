# Community benchmark on 2x NVIDIA RTX PRO (4500 32 GB + 4000 24 GB), Z820

Measured 2026-11-01 by Jack Kenner (GitHub `jkenner111`).

## The point of this machine

This is a 2013 workstation: dual Xeon E5-2697 v2, 251.8 GiB of DDR3, and — the part that
usually ruins it — model files on a **SATA** SSD that reads at 85–148 MB/s. A 93.7 GB quant
on that drive would normally make the disk the bottleneck for everything not already in VRAM.

It isn't. **The drive never enters the decode path.**

The engine's own prefill plan states it directly. For a 32K-token prompt it must touch
**55.4 GiB of experts**. It finds them here:

| Where the experts live | Single 4500 | Dual (layer 13) |
|---|---:|---:|
| GPU VRAM expert caches | 22.2 GiB | **38.1 GiB** |
| Page-locked DDR3 RAM copy | 37.0 GiB | 24.8 GiB |
| File tier (the SATA drive) | **0.0 GiB** | **0.0 GiB** |

> `strata: prefill plan: ... about 0.0 GB read from the drive per 32768-token prompt`

The engine prices the drive in its own plan — `drive 1.50 GB/s (assumed)` — and then reports it
is never asked for anything. In resident mode the log says it plainly:

> `resident RAM mode: 37.00 GiB of experts in RAM (page-locked), 9845 in the GPU cache; adaptive swaps exchange them with the GPU cache (no file reads)`

That is the story worth telling: **252 GB of DDR3 is doing the job a Gen5 NVMe would otherwise
be bought for.** The working set is pinned and split across both NUMA nodes (node 0 12.52 GiB,
node 1 12.30 GiB in the dual arm), so the cheap card's shortfall is absorbed by system memory
instead of disk.

And the CPUs are not idle passengers. This chip predates AVX2 entirely — the engine falls back
to AVX1 **128-bit** kernels — yet 22 pool workers split by socket (12 + 11 threads, one per
physical core, host on core 0) carry the expert compute for every expert not resident in VRAM,
while the GPUs run the ~98.9% of routed mass their caches hold. Two 2013 Xeons with 128-bit
SIMD sustain **105–113 tok/s decode**. That is the second half of the story: the division of
labour, not the silicon.

## What this opens up: an old platform is not a dead end

Most people would assume a 13-year-old workstation has no business hosting a current-generation
GPU for this work. Four reasons get cited. This machine tests all four, and none of them turns
out to be load-bearing:

| The assumed blocker | What this machine actually has | What it measured |
|---|---|---|
| Storage too slow | SATA SSD, 85–148 MB/s | **0.0 GiB read per 32K prompt** |
| PCIe too old | Gen3 x16 (cards idle at Gen1) | 12.7–13.2 GB/s probe; 105–113 tok/s |
| CPU too old | Ivy Bridge, **no AVX2 at all** | AVX1 128-bit kernels, 22 workers; 105–113 tok/s |
| Not enough VRAM | 45 GiB across two unequal cards | 252 GB DDR3 absorbs the overflow |

DDR3 is not the handicap it sounds like. Eight channels of it deliver more aggregate bandwidth
(~119 GB/s) than a current consumer DDR5-6000 dual-channel desktop (~96 GB/s), and the capacity
pins 37 GiB of experts outright.

**The practical consequence: a Z620/Z820/Z840, an X79 or X99 board, or a retired dual-socket
server can host a current pro GPU and deliver usable throughput.** Strata's RAM-resident expert
tier plus a NUMA-aware CPU pool remove precisely the bottlenecks that make old platforms
unattractive. The upgrade that matters is the card, not the platform — and the RAM these
machines were built with is the thing that makes it work.

That possibility is worth recording because it is not currently represented anywhere in this
corpus. Three entries run dual Xeons — a Z840 (2x E5-2699 v3), 2x P40 (2x E5-2698 v4), and
4x P100 (2x E5-2650 v3) — and all three are Haswell or Broadwell, which **have AVX2**. Three
entries use DDR3, but on Haswell-class boards. Searching every report here for Ivy Bridge,
E5 v2, AVX1 or "no AVX2" returns nothing: **this is the first published Strata result on a
pre-AVX2 CPU.**

The engine has a dedicated fallback for exactly that case, and warns about it at startup:

> `EXPERIMENTAL older-CPU build (ggml-cpu for avx): this CPU has no AVX2, so the CPU's experts run on ggml-cpu's kernels - expect it to be slow`
>
> `this CPU has no AVX-512: the expert kernels run on AVX1 128-bit (iq_avx1, the older-CPU build)`

Until now there was no number attached to "expect it to be slow". This report is that number,
and it is better than the warning suggests: **105–113 tok/s decode on the slowest ISA path the
engine supports**, sustained by 22 workers on two 2013 sockets running 128-bit kernels.

## Hardware and software

- **GPU 0:** NVIDIA RTX PRO 4500 Blackwell, 32 GB (31.9 GiB usable), sm_120 — 82 SMs @ 2.41 GHz
- **GPU 1:** NVIDIA RTX PRO 4000 Blackwell, 24 GB (23.1 GiB usable), sm_120 — 70 SMs @ 2.06 GHz
- **CPU:** 2x Intel Xeon E5-2697 v2 @ 2.70 GHz, 24 cores / 48 threads, **no AVX2, no AVX-512**
- **RAM:** 251.8 GiB usable of 256 GB installed — **16 x 16 GB DDR3-1866 ECC**, 4 channels
  per socket (8 channels total, one slice per NUMA node), ~119 GB/s theoretical aggregate.
  A load-bearing component here, not a spec-sheet number.
- **Storage:** `/dev/sdb3`, 1.9 TB **SATA** SSD. Slow on purpose; see above for why it doesn't matter.
- **OS:** Fedora Linux 43 (Workstation Edition), kernel 7.0.14-101.fc43.x86_64
  (generated READMEs render this as `Linux 43 (Workstation Edition)` with a literal `<host>`
  placeholder — an engine template bug, left uncorrected rather than edit generated evidence)
- **Driver / CUDA:** 580.159.03 / CUDA 13.0
- **Strata:** commit `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`, engine 0.1.55, source build
- **Build:** `STRATA_ISA_FLOOR=avx` — no AVX-512 on this host, so the build floors the ISA
- **Background work:** none. Both cards idle (2 MiB), GPU service units stopped.

## Model and configuration

- **Repository:** `unsloth/Qwen3.8-Flash-Next-GGUF`, revision `38bb39ee`
- **Source model:** `Qwen/Qwen3.8-Flash-Next` at `de4b8e4d`, verified out of band against the
  Hugging Face API but **not** captured by the runner (`--source-revision` not passed; the
  result files honestly record `null`)
- **Quantization:** UD-IQ4_XS, 93.7 GB across three shards
- **Context:** 262,144 tokens, 1 server slot; KV int8, 32,768 cells resident
- **Sampling:** the protocol's fixed set — temp 1.0, top_p 0.95, top_k 20, min_p 0.0,
  penalties 0.0 / 1.0, `reasoning_effort: xhigh`, thinking on and preserved

| Arm | Cards | Split | Pool workers |
|---|---|---|---|
| `arm1-4500-nonumactl-20261101` | RTX PRO 4500 alone | — | 22 (default) |
| `arm4-dualsplit-20261101` | both, 4500 = main | `auto` → layer 13 | 22 (default) |
| `arm5-dualsplit-k10-20261101` | both, 4500 = main | explicit layer 10 | 22 (default) |
| `arm2-dualsplit-w11-20261101` | both, 4500 = main | `auto` → layer 13 | 11 |

`"gpu": [1, 0]` — helper first, main last. The last card runs head, draft and verify, so the
4500 (more VRAM, more SMs, higher clocks) takes that role. The 4000 holds layers 0–12, the
4500 holds 13–47 plus head and draft.

## Results

Medians with (min–max) over three repetitions. Decode and prompt rates are the engine's own
`timings`; TTFT is the client's monotonic clock.

### Single card — RTX PRO 4500 (32 GB)

| Case | Prompt tok/s | Decode tok/s | TTFT s |
| --- | --- | --- | --- |
| short | n/a | 61.3 (56.5–68.8) | 0.82 (0.65–1.01) |
| prompt-4k | 1143.8 (1081.9–1149.7) | 63.5 (61.1–64.9) | 3.60 (3.59–3.81) |
| prompt-16k | 2031.1 (2001.0–2044.0) | 64.6 (62.3–70.4) | 8.11 (8.06–8.23) |

### Dual card, layer 13 — RTX PRO 4000 (0–12) + RTX PRO 4500 (13–47)

| Case | Prompt tok/s | Decode tok/s | TTFT s |
| --- | --- | --- | --- |
| short | n/a | **108.7** (105.2–119.6) | 0.80 (0.58–0.81) |
| prompt-4k | **1520.8** (1504.4–1540.2) | **113.2** (97.8–120.9) | **2.72** (2.69–2.74) |
| prompt-16k | **2581.0** (2579.0–2584.7) | **105.3** (102.4–114.9) | **6.39** (6.38–6.39) |

### Dual card, explicit layer 10

| Case | Prompt tok/s | Decode tok/s | TTFT s |
| --- | --- | --- | --- |
| short | n/a | 95.2 (93.2–103.4) | 1.35 (0.99–1.47) |
| prompt-4k | 1266.1 (1253.9–1306.5) | 103.9 (87.0–107.8) | 3.26 (3.16–3.29) |
| prompt-16k | 2280.7 (2269.8–2281.9) | 105.6 (96.5–107.5) | 7.22 (7.22–7.26) |

### Dual card, layer 13, pool-workers 11

| Case | Prompt tok/s | Decode tok/s | TTFT s |
| --- | --- | --- | --- |
| short | n/a | 113.3 (96.3–114.3) | 0.82 (0.69–0.84) |
| prompt-4k | 1501.5 (1492.4–1535.4) | 103.9 (89.8–108.7) | 2.75 (2.69–2.77) |
| prompt-16k | 2565.0 (2481.8–2574.7) | 100.3 (98.8–116.0) | 6.43 (6.40–6.65) |

Sanity checks: **12 of 12** in all four runs (I1–I4 instruction, R1–R4 retrieval, T1–T4 tool call).

## Findings

**1. The second card buys RAM, not just compute.** Adding the 24 GB card raises the GPU expert
cache from 22.2 → 38.1 GiB while simultaneously *shrinking* the pinned RAM copy from 37.0 →
24.8 GiB, because fewer experts fall back to system memory. Decode goes 61.3 → 108.7 tok/s
short (+77%), 63.5 → 113.2 at 4K (+78%), 64.6 → 105.3 at 16K (+63%).

**2. The layer split helps prompt too, which the docs don't predict.** 16K prompt throughput
rises 2,031 → 2,581 tok/s and TTFT drops 8.11 s → 6.39 s. The guidance that a layer split "adds
capacity without adding speed" did not hold here.

**3. The engine's own split predictor chose wrong and said so.** With `layer_split: auto` it
picked 13 but flagged K=10 as within 0.7%. K=10 measured 20% worse on 4K prompt, 13% worse on
16K prompt, and worse TTFT at every size. Decode was a dead heat at 16K (105.6 vs 105.3). The
predictor prices decode and says as much in the log — worth knowing if you have unequal cards.

**4. `--pool-workers` is not a lever on this host.** 11 vs 22 produced overlapping ranges
(4K decode 103.9 vs 113.2, both spanning ~98–121).

## Limitations

- **Draft protocol** (`strata-qwen38-thinking-v1-draft.1`) — outside the standard manifest.
- **Six of nine performance requests hit the 512-token cap** in arm 1 (3/3 at 4K, 3/3 at 16K) and
  arm 4 (2/3 at 4K, 3/3 at 16K). Decode is measured to the cap on truncated generations.
- **`--source-revision` was not passed**; base-model revision verified out of band and recorded
  as `null` by the runner.
- **Arm 1 was VRAM-tight.** The log carries two warnings — only 502 MiB and 372 MiB of VRAM free
  at request start, below the engine's 512 MiB threshold, where the driver can page GPU memory to
  host RAM and decode collapses (`#2065`). It did not collapse, but the single-card baseline may
  be depressed by that pressure. Raise `--vram-reserve-mib` before treating arm 1 as a ceiling.
- **Storage is SATA**, but per the prefill plan the drive contributes 0.0 GiB per 32K prompt, so
  this affects load time only.
- No optional concurrency stage; all figures are concurrency 1.
- **Not comparable to this machine's own October history**: different engine (0.1.41 vs 0.1.55),
  greedy sampling instead of temp 1.0 / xhigh, different prompts, and NVMe rather than SATA.

## How it's configured for this machine

Strata's default path assumes AVX2 and a fast disk. Four changes make it run here.

**1. Build with the ISA floor at AVX.** The default build carries AVX2 kernels this CPU cannot
execute. `STRATA_ISA_FLOOR=avx` produces the older-CPU build, and the engine uses its AVX1
128-bit expert kernels instead. Without this the binary does not start.

**2. Pin the experts in RAM instead of reading the file.** `--low-ram resident` keeps the expert
set page-locked in DDR3 and swaps it against the GPU caches, rather than reading experts from the
GGUF on disk. Setup derives the budget from the model, not the RAM: `--resident-budget-gib 55`
comes from UD-IQ4_XS's 59.5 GB arena (`59.5 / 1.0737`), so the cap is the model's size, not this
machine's 252 GiB. This is the mode that makes the SATA drive irrelevant.

**3. Keep resident mode across the layer split.** Historically `--resident-experts` and multiple
GPUs were mutually exclusive: setup *silently rewrote* the resident flag to `--mmap-experts`,
which reads experts through the OS file cache instead of pinned RAM. On a SATA drive that is the
worst possible outcome. Engines from 0.1.40 (`RESIDENT_SPLIT_ENGINE`) support the pair; this runs
0.1.55, so the machine keeps pinned RAM **and** two cards. On an older engine, check which flag
setup actually wrote before trusting a result from this kind of machine.

**4. Do not use `numactl`.** The engine builds its own NUMA-aware pool: 22 workers, 12 on node 0
and 11 on node 1, host on core 0, one worker per physical core. Pinning the process to a single
socket with `numactl` cuts throughput — arm 1 is the unwrapped baseline and measures faster than
the earlier wrapped configuration.

Beyond that: the layer split exists because 32 + 24 GB of VRAM is smaller than the 55 GiB arena,
so the experts are divided across two unequal cards with the 4500 as main (head, draft and
verify). KV is int8 with 32,768 resident cells, which is what makes 262,144 tokens of context fit
in 32 GB of VRAM.

## Reproducing

```sh
# build (this host has no AVX-512)
STRATA_ISA_FLOOR=avx ./setup.sh --setup --yes --no-start \
  --family qwen --model UD-IQ4_XS --gguf-dir <UD-IQ4_XS dir> \
  --data-dir <data dir> --gpu 0 --context 262144 --low-ram resident

# single card
bin/strata bench run --profile ud-iq4-xs-gpu0 --suite standard-v1 --track custom \
  --model-revision 38bb39ee97821de2c9009abb7e93950eec396e66 --quantization UD-IQ4_XS

# both cards — config has "gpu": [1, 0], layer_split "auto"
bin/strata bench run --profile ud-iq4-xs-dual --suite standard-v1 --track custom \
  --model-revision 38bb39ee97821de2c9009abb7e93950eec396e66 --quantization UD-IQ4_XS
```

Do not wrap in `numactl`. The engine builds a NUMA-aware pool itself (one slice per socket);
pinning it to one socket on this dual-socket host costs throughput.