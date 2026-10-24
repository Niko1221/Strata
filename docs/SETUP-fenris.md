# fenris setup — this fork's own rig and configuration

This is the documentation that makes this fork more than a copy of upstream: it
records the exact machine, build, configuration and workflow that produced the
contributions in this repository (benchmark reports, issues #3690/#3866, PR #3863).

## Hardware and OS

- **Machine:** Dell Precision T5610 (dual-socket Ivy Bridge workstation; mainboard 0WN7Y6, BIOS A19).
- **GPU:** 2x NVIDIA RTX 3060 12 GB (sm_86), layer split across both (`--layer-split auto`, gpu 0+1).
  Dual-GPU on one NUMA node; PCIe link-state power management off; power/clock capped to 100 W / 300..1650 MHz per card (two different VBIOS otherwise boost asymmetrically and trip the PSU).
- **CPU:** 2x Xeon E5-2697 v2 (Ivy Bridge, 12c/24t per socket, **no AVX2**). The no-AVX2 path is why every engine here is a source build with `STRATA_ISA_FLOOR=avx` (AVX2 prebuilts SIGILL).
- **RAM:** 128 GB DDR3-1866, **quad-channel per socket** (8x 16 GB, DIMM1-4 on each CPU) - relevant whenever expert-cache bandwidth or CPU-decode worker counts are tuned.
- **OS:** Windows 11. TdrDelay=30, power plan "Ultimative Leistung".

## Build (engine + vision)

Both from a source worktree at a tag (example v0.2.3.1 / engine 0.1.48.1):

```
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_Q6K_EXPERTS=ON \
      -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_ISA_FLOOR=avx \
      -DCMAKE_CUDA_COMPILER=.../nvcc -DCUDAToolkit_ROOT=.../CUDA/v13.3   # then --build strata
```

Vision reader separately (`tools/vision` with `-DLLAMA_DIR=<build>/_deps/strata_llamacpp-src`,
`-DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`) -> `build-vision/bin/strata-vision.exe`.

## Production model stack

- **Active:** Swift-IQ3_XXS-**abliterated** (SC117 / Swift-1.5-...-GSQ-RCO-abliterated). Uncensored variant (only the abliterated files are).
- **Parked:** Swift-IQ2_XS (abliterated), Swift-IQ3_XXS standard, Qwen3.6-35B-A3B (test only; 0 % conversation cache on this rig -> not for agent workloads).
- Models live on **E:** (a striped SATA-SSD, RAID-0, for the 28 GB PLE-table reads); source/staging on D:.

Production config (`strata-swift-iq3_xxs-abl.json`) essentials:
`--pipeline-windows 2`, `--kv q4_0 --kv-resident 32768`, `--prefill auto:32768`, `--spec 4`,
`--conversation-cache-mib 8192`, `--vram-reserve-mib 1344` (1280 pages on this quant, see the engine's #2065 warning),
env `STRATA_ARENA_PIN_GIB=26`, `STRATA_STAGE_TRIM=1`, `STRATA_MTP_STAGE0=1`, `--vision` (reader: our old v0.1.45 build - it is fast and correct; the 0.1.48 fixed reader is validated and ready).

Single start path: `start-strata.bat` (applies power caps, launches serve/server.py with the config). No auto-restart loop.

## Workflow / contribution practice

- **Benchmarks are measured on a localhost-only server** (LAN clients on the parallel:1 slot destroy TTFT/latency); same-day, three runs per size, `tools/strata_bench.py --url` then posted per `docs/COMMUNITY_BENCHMARKS.md`.
- Quality probes: a 20-question tricky set (counting traps, negation, format, logic) + a censoring probe; results under `bench/`.
- GitHub posts follow a draft -> owner approval -> post flow (permanent auth via `gh`, token in the Windows credential manager).
- On this box, `--pool-workers 11` (2-socket hint) and `--prefill auto:32768` are open A/B candidates.