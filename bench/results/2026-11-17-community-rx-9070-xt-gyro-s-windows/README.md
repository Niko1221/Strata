# Community benchmark on AMD Radeon RX 9070 XT (Windows, Gyro-S, 31 GiB RAM)

Measured on 2026-11-17 by @ric20527. The official Windows HIP `rx9070` engine with Qwen3.8-Flash-Next Gyro-S at a
131,072-token context, on a PC whose RAM (31.4 GiB) cannot hold the experts the GPU does not cache. Three runs on the
same day, same build, same model: the configuration the desktop app wrote, a hand-tuned one, and the standard benchmark
on the tuned one. Main limitation: one PC, one model, Claude desktop app and Chrome open during the runs.

## Hardware and software

- GPU: AMD Radeon RX 9070 XT 16 GiB (gfx1201); PCIe host->device 55.0 GB/s (the engine's own probe)
- CPU: AMD Ryzen 5 7600X3D (6 cores, 12 threads, AVX-512); RAM: 31.4 GiB installed; page file 32 GiB on the system SSD
- Storage of the model files: SK Hynix HFS512GDE9X084N NVMe (512 GB)
- OS: Windows 11 Pro 10.0.26300; GPU driver 32.0.32015.16001
- Strata: desktop app 0.2.14, control 0.1.60.1 (commit `3af562d5`), official engine package
  `0.1.60-windows-x86_64-hip-rx9070` (ROCm 10.2, archs gfx1200/gfx1201), not a source build
- Background: Claude desktop app and Google Chrome open (about 3 GB of RAM and 1.4 GB of VRAM used by the desktop and
  these programs before Strata loads); no other GPU work

## Model and configuration

- Model: `Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf` (54 GB, sha256 `c6ea1b08ca312f99...`), native pack `gyro-gyro-s`
  (24.0 GiB of experts), MTP draft layer, the control's `expert-profile.bin` (`8f59b4aa8873209d...`)
- Context 131,072 tokens, `--kv int8`, `--kv-resident 32768`, `--resident-experts`, `--prefill auto`, `--spec 4`,
  `--spec-min-p 0.5`, vision on (`strata-vision` on the CPU)
- No calibration, no experimental speed projection

The two configurations differ only in these settings:

| Setting | App defaults | Tuned | Why |
| --- | --- | --- | --- |
| `STRATA_HIP_WMMA` | unset (off on Windows) | `1` | prompt attention on the matrix cores |
| `vram_elastic` (config) | `true` | `false` | with `true` the engine kept the picture reader's 2048 MiB in the VRAM reserve although vision runs on the CPU and `--vram-elastic` is NVIDIA-only (log: "the resident RAM mode cannot resize the expert cache, so the picture reader's 2048 MiB stay in the reserve") |
| `--vram-reserve-mib` | 1536 | 1024 | user choice; the engine warns below about 1 GiB free |
| `--pool-workers` | 5 | default (5 + host thread) | no difference measured |
| `--resident-budget-gib` | unset | 17 | an explicit RAM budget for the experts the GPU does not hold |
| `STRATA_RESIDENT_PIN` | `0` | `1` | |
| `STRATA_PARTIAL_PIN`, `STRATA_PARTIAL_PIN_GIB` | unset | `1`, `8` | the hottest 8 GiB of the RAM copy registered for the GPU, so it reads those experts over PCIe instead of the CPU computing them |
| `STRATA_UNBUFFERED_LOAD` | unset | `0` | with an explicit budget the load read the pack unbuffered and a fresh 16K prompt read at 1,137 instead of 1,524 tok/s (one A/B pair) |

Resulting placement (engine log): app defaults 7,106 expert slots in VRAM (6.95 GiB) and 15.83 GiB of experts in RAM;
tuned 8,238 slots (8.05 GiB) and 17.00 GiB in RAM, 8.00 GiB of it registered for the GPU.

## Method

- `quick-app-defaults/`, `quick-tuned/`: `python tools/strata_bench.py --config <file> --long`, three runs per test,
  thinking off, temperature 0, each from a freshly started engine. Run back to back (13:29 and 13:33 local time).
- `standard-tuned/`: `strata bench run --profile <tuned config> --track custom` (protocol
  `strata-qwen38-thinking-v1-draft.1`), status complete, from a freshly started engine right after.
- Model loading is not timed. Every file's own README describes its method in full.

## Results

Quick suite, median (range), same day:

| Test | App defaults | Tuned | Change |
| --- | ---: | ---: | ---: |
| short chat decode tok/s (n=9) | 50.2 (39.4-55.7) | 71.3 (60.8-89.9) | +42 % |
| short chat first token s | 0.60 (0.48-0.83) | 0.35 (0.29-0.57) | -42 % |
| 4K fresh prompt tok/s (n=3) | 829.0 (769.1-832.1) | 1236.2 (1222.4-1248.3) | +49 % |
| 16K fresh prompt tok/s (n=3) | 1346.6 (1317.2-1354.0) | 1571.3 (1560.6-1575.4) | +17 % |
| 32K fresh prompt tok/s (n=3) | 1322.8 (1320.4-1329.8) | 1568.9 (1567.2-1570.0) | +19 % |
| agent session cache hit % | 70.8 | 70.8 | 0 |
| agent turn first token s (n=15) | 2.23 (1.80-2.54) | 1.93 (1.42-2.16) | -14 % |
| agent decode tok/s (n=15) | 61.2 (19.9-76.1) | 60.4 (38.0-100.7) | within the spread |

Standard benchmark on the tuned configuration (thinking on, the model card's sampling): short decode 83.7 tok/s
(76.9-84.4), 4K prompt 1276.7 tok/s (1175.6-1278.5), 16K prompt 1608.3 tok/s (1602.5-1612.1), decode on the 16K case
81.2 tok/s (78.6-87.9); **sanity checks 12 of 12**. Four of nine performance requests reached the 512-token cap while
reasoning, as the protocol allows; no request failed.

From normal use of the tuned configuration over the following hours (the engine log, not a controlled run):
conversations of 60-90K tokens decoded at 55-92 tok/s, and fresh prompt reads of 5-75K tokens ran at about
1,300-1,460 tok/s. The decode expert cache hit rate stayed at 88-95 %, and KV streaming hit VRAM 98.9 % of the time.

## What did not help on this PC

Each of these was a single A/B run on 2026-11-16/17 with `perfil.py`-style requests (600-token code and prose answers and a 16K prompt). All of them were within about ±5 % of the tuned configuration or worse:

- `--host-core sibling` (80.5 vs 80.4 tok/s mean over 6 requests each); `--spec 5`; `--spec-min-p 0.35` and `0.65`;
  `--lookup-chain 2`
- 11 CPU pool workers instead of 5 (much slower); `STRATA_RESIDENT_HEADROOM_GIB=2`
- `STRATA_HIPBLASLT_TABLE=on` (the table loads, no change measured)
- `STRATA_RESIDENT_PIN=1` alone: the driver refused to register 17 GiB at once
- `STRATA_PARTIAL_PIN_GIB` above about 12: with 12.73 GiB the driver ran out of lockable memory and the engine did not
  start. 6, 8, 10 and 12 GiB decoded alike (within the noise); 8 was kept for its margin below that limit.

## Correctness and limitations

- The standard run's twelve sanity checks passed. The quick suite is speed only.
- Only one model (Gyro-S TQ1_0) and one PC. The per-setting reasons above come from single A/B runs. Only the combined
  effect was measured three times on the same day.
- Speed depends on free RAM at load: with about 2 GB less available (more programs open), the RAM budget is clipped
  (17 -> 14.5 GiB) and 16K prompts read at 400-600 instead of about 1,550 tok/s.
- Switching between two long conversations re-reads the whole prompt here (45-53 s at 64-75K tokens in normal use),
  because the conversation cache stays off for lack of free RAM. That is reported separately; it is not part of these
  numbers.
- Paths in the logs are replaced by `<user>` / `<strata>` placeholders. The Spanish folder names (`Modelos Ia`) are the
  owner's.
