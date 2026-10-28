# RX 6900 XT (single card), Qwen3.8-Flash-Next: STRATA_HC_PERSIST=1, VRAM reserve, adapt swaps, CPU pool workers

Measured on 2026-10-28 with `tools/strata_bench.py` (community suite, not the standard benchmark of
[BENCHMARK_PROTOCOL.md](../../../docs/BENCHMARK_PROTOCOL.md)). One RX 6900 XT on Linux, three official packs:
Coder IQ1_M, original IQ2_XS and Swift 1.5 IQ3_S.

[AMD_HIP.md](../../../docs/AMD_HIP.md#rdna2-gfx1030) lists `STRATA_HC_PERSIST=1` as opt-in on gfx103x, measured on
2x RX 6900 XT ([2026-10-12-hc-persist-gfx1030](../2026-10-12-hc-persist-gfx1030/README.md)). This report adds a
single-card measurement on ROCm 7.2.4 from Arch packages.

## Result

- **`STRATA_HC_PERSIST=1`: decode +4.0 % to +7.8 % (chat) and +3.2 % to +5.8 % (agent) on all three packs**, prompt
  speed unchanged. Every pass of every pack is at or above its base pass.
- `--vram-reserve-mib` 3072 -> 2048 / 1536 on IQ2_XS (5660 -> 6410 / 6784 expert slots): -1.3 % to +3.3 %, inside
  the pass-to-pass spread.
- `--adapt-swaps` 80 -> 160 (Coder, IQ2_XS): no change (-0.5 % to +2.2 %).
- `--pool-workers` 15 (default) / 12 / 8 on IQ2_XS: the default is fastest; 8 workers cost 1-3 %. The CPU side of
  this machine is bound by memory bandwidth (dual-channel DDR4-3600), not by cores.

Medians over all requests of three passes (9 chat requests, 15 agent turns, 3 prompts of each size per pass).
Arms ran interleaved (A, B, C, A, B, C, ...) with a fresh server for every arm. 39 runs, no failed request.

### STRATA_HC_PERSIST=1

| Pack | Chat decode tok/s | Agent decode tok/s | 4K prompt tok/s | 16K prompt tok/s |
| --- | ---: | ---: | ---: | ---: |
| Coder IQ1_M, base | 46.0 | 54.0 | 1,159.9 | 1,253.5 |
| Coder IQ1_M, HC_PERSIST | **49.6 (+7.8 %)** | **57.1 (+5.8 %)** | 1,159.6 (-0.0 %) | 1,252.4 (-0.1 %) |
| IQ2_XS, base | 61.9 | 66.8 | 1,106.6 | 1,210.7 |
| IQ2_XS, HC_PERSIST | **64.9 (+4.8 %)** | **70.7 (+5.8 %)** | 1,106.8 (+0.0 %) | 1,211.4 (+0.1 %) |
| Swift IQ3_S, base | 45.0 | 51.0 | 1,064.7 | 1,204.3 |
| Swift IQ3_S, HC_PERSIST | **46.8 (+4.0 %)** | **52.6 (+3.2 %)** | 1,050.0 (-1.4 %) | 1,197.0 (-0.6 %) |

Per-pass medians (chat / agent tok/s): Coder base 45/47/47 and 54/54/54, HC 50/49/50 and 55/55/59; IQ2_XS base
62/61/62 and 66/66/67, HC 63/65/66 and 72/70/72; Swift base 44/45/45 and 51/51/50, HC 46/46/48 and 51/55/52.
Swift's lower 4K prompt median comes from one HC pass (1,010 tok/s; the other two 1,069 / 1,048); the base arm had
one such pass too (1,048).

The engine log confirms the switch on every HC arm and its absence on every base arm:

```
strata hc: CUDA0: STRATA_HC_PERSIST=1: the verifier's hyper-connection read runs as one persistent kernel (80 blocks on
40 multiprocessors, up to 8 tokens; 0 token counts timed out in the check; ...
```

### VRAM reserve and adapt swaps (no HC_PERSIST)

| Arm | Expert slots | Chat decode tok/s | Agent decode tok/s | 4K prompt tok/s | 16K prompt tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| IQ2_XS, reserve 3072 (base) | 5,660 | 61.9 | 66.8 | 1,106.6 | 1,210.7 |
| IQ2_XS, reserve 2048 | 6,410 | 62.4 (+0.8 %) | 66.0 (-1.3 %) | 1,110.0 (+0.3 %) | 1,212.8 (+0.2 %) |
| IQ2_XS, reserve 1536 | 6,784 | 62.9 (+1.6 %) | 69.0 (+3.3 %) | 1,113.0 (+0.6 %) | 1,213.1 (+0.2 %) |
| IQ2_XS, adapt swaps 160 | 5,660 | 61.7 (-0.3 %) | 66.5 (-0.5 %) | 1,107.3 (+0.1 %) | 1,211.6 (+0.1 %) |
| Coder IQ1_M, adapt swaps 160 | 3,535 | 47.0 (+2.2 %) | 54.0 (+0.1 %) | 1,155.9 (-0.3 %) | 1,251.1 (-0.2 %) |

Per-pass agent medians for reserve 1536: 67/70/69 against the base's 66/66/67. No out-of-memory or paging at 1536
with the desktop session running.

### CPU pool workers (IQ2_XS, with HC_PERSIST)

Second config: reserve 2700 instead of 3072 and `STRATA_HC_PERSIST=1` (this machine's daily settings), so its
absolute numbers differ from the tables above.

| `--pool-workers` | Chat decode tok/s | Agent decode tok/s | 4K prompt tok/s | 16K prompt tok/s |
| --- | ---: | ---: | ---: | ---: |
| **15 (default: physical cores minus the host's)** | **66.8** | **72.0** | 1,109.3 | 1,211.3 |
| 12 | 65.0 (-2.7 %) | 72.2 (+0.3 %) | 1,109.5 (+0.0 %) | 1,212.2 (+0.1 %) |
| 8 | 64.6 (-3.3 %) | 71.2 (-1.1 %) | 1,109.1 (-0.0 %) | 1,211.6 (+0.0 %) |

## Hardware and software

| | |
| --- | --- |
| GPU | AMD Radeon RX 6900 XT 16 GB (gfx1030), PCIe 4.0 x16 (16.0 GT/s, width 16), Resizable BAR on, power cap 272 W (driver default) |
| CPU | AMD Ryzen 9 5950X, 16 cores / 32 threads |
| RAM | 64 GB (62.7 GiB visible) DDR4-3600 CL16, dual channel, XMP |
| Storage | model files on a Samsung 990 PRO 2 TB (NVMe) |
| OS | Arch-based (Omarchy), kernel 7.2.8, Hyprland desktop session running during the measurement |
| ROCm | Arch packages: rocm-core 7.2.4-1, hip-runtime-amd 7.2.4-1, rocblas 7.2.4-3 |
| Strata | engine 0.1.52, source build for gfx1030 (`-DCMAKE_HIP_ARCHITECTURES=gfx1030`) |

**Source.** A fork branch at `5a43a3d9`, which is upstream `main` at `5e8a4cca` (2026-10-27) plus 16 local commits.
For the Qwen3.8 path these add one data file: `tools/hip/gfx1030-rocblas-5.2.0.dabb6df2b98-dirty.txt`, the rocBLAS
solution table for this ROCm build (used through `STRATA_ROCBLAS_TUNING`; prompt GEMMs only; proposed in #3924).
The other commits touch the Qwen3.6 engine path (`src/qwen35`), the local switch script, the web page and the image
helper, none of which the Qwen3.8 decode runs.

**Models.** Downloaded by `setup.py` at its pinned revisions with SHA-256 checks:

| Pack | Repository @ revision | Context | Expert profile |
| --- | --- | ---: | --- |
| Coder IQ1_M | ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF @ `5348543e` | 262,144 | `data/expert-profile-coder.bin` |
| IQ2_XS | ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF @ `ed59f920` | 262,144 | `data/expert-profile.bin` |
| Swift 1.5 IQ3_S | ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF @ `99bb8f7f` | 131,072 | `data/expert-profile.bin` |

## Settings

Base arms are the configs `setup.py` wrote for this machine (with `--vision yes`, `--vram-reserve-mib 3072` and its
calibration), copied to `configs/<arm>.json` with paths replaced by placeholders. Each other arm changes exactly one
setting of its base (see `run.sh`).

- Coder and IQ2_XS: `--expert-cache auto --prefill auto --spec 4 --mtp <MTP runtime> --kv int8 --kv-resident 32768
  --vision --pcie-frac 0.20 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97 --vram-reserve-mib 3072`;
  `--spec-min-p` 0.50 (Coder) / 0.70 (IQ2_XS).
- Swift: as above with `--max-context 131072`, `--adapt-swaps 160`, `--spec-min-p 0.70`, no `--kv-resident`.
- Environment for all: `STRATA_ROCBLAS_TUNING=<STRATA>/tools/hip`, `STRATA_HIP_PROMPT_F16=1`, `STRATA_SELECT_SIMT=1`.
- 15 CPU pool workers (default), vision reader on the CPU (300 image tokens; no images in the suite).
- Suite defaults: thinking off, the server's sampling, three runs per test.

## Files

| Path | Content |
| --- | --- |
| `raw/<arm>-p<pass>.json` | the `results.json` of every run (39 files): every request with its token counts, timings and drafts, plus the machine facts |
| `configs/<arm>.json` | the sanitized config of every arm |
| `engine-log-excerpts/` | the first 80 lines of the engine log of each base and HC arm (pass 1) |
| `run.sh`, `run-pool-workers.sh` | the drivers (`STRATA_DIR=... ./run.sh`; the pool-worker run also needs `CONFIGS_DIR`) |
| `analyze.py` | pools every request of all passes per arm: `python3 analyze.py .` prints the tables above |

## Limits

- One machine, one session per arm and three passes. Run-to-run spread of the chat median is about +-2 tok/s, so the
  reserve and swaps results do not separate from noise; the HC_PERSIST result holds in every pass.
- HC_PERSIST is not bitwise the default read ([AMD_HIP.md](../../../docs/AMD_HIP.md#rdna2-gfx1030)); this report
  measures speed only, no quality comparison beyond the engine's own start-up check.
- Not the standard benchmark: thinking off, the suite's word-list prompts, no recall checks.
- Desktop session open; no other GPU or heavy CPU load during the runs.
