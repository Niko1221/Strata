# Community benchmark: Swift IQ3_XXS @ 262K on RTX 4080 SUPER (engine 0.1.56)

Measured by: MASTA/Eich (XX) on a Ryzen 9 7950X3D build, 2026-11-03, engine 0.1.56 (v0.2.10).
Bench via `tools/strata_bench.py`, needles via `tools/needle_bench.py` from `D:\Strata`.

## Hardware / Software
- CPU: AMD Ryzen 9 7950X3D 16-core/32-thread (Zen 4, AVX-512)
- GPU: NVIDIA GeForce RTX 4080 SUPER 16 GB (sm_89), driver 617.42, CUDA 13.4
- RAM: 64 GB (63.7 GiB), NVMe SSD
- OS: Windows 11, Python 3.14
- Engine: Strata 0.1.56 (commit 06698c674673, CUDA 13.0, archs 75/86/89/120)

## Model / Config
Swift (huihui-ai Swift-1.5) IQ3_XXS, 262K context, KV int8, Vision on, PLE-fp8.

Launch (from `strata-swift-iq3_xxs.json`):
```
strata.exe --pack swift --native <Swift IQ3_XXS GGUF>
  --ple-gguf D:/Strata-data/ple/ple-fp8.gguf --expert-profile auto
  --expert-cache auto --prefill auto --spec 4 --max-context 262144 --kv int8
  --kv-resident 32768 --vision --vram-reserve-mib 1000 --pcie-frac 0.00
  --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

## Results (median of run2, closest to the 3-run median)
| metric | value |
|---|---|
| chat decode (tok/s) | 105.4 |
| chat TTFT (s) | 0.288 |
| 4K prompt (tok/s) | 2394.5 |
| 16K prompt (tok/s) | 3709.2 |
| agent decode (tok/s) | 115.3 |
| agent turn TTFT (s) | 1.564 |

3-run aggregate: chat 105.4 (101.8-108.6), agent 115.3 (113.7-125.9), 4K 2394.5, 16K 3709.2.

## Agent cache (3 sessions)
cache hit 69.93%, re-read 74 tok/session, new 12920 tok/session.

## Recall checks
6 of 6 needles found (32K and 256K, depths 10/50/90). Raw runs: needles.json
