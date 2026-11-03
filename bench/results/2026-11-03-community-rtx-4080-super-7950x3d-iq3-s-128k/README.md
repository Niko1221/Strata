# Community benchmark: qwen3.8-flash-next IQ3_S @ 128K on RTX 4080 SUPER (engine 0.1.56)

Measured by: MASTA/Eich (XX) on a Ryzen 9 7950X3D build, 2026-11-03, engine 0.1.56 (v0.2.10).
Bench via `tools/strata_bench.py`, needles via `tools/needle_bench.py` from `D:\Strata`.

## Hardware / Software
- CPU: AMD Ryzen 9 7950X3D 16-core/32-thread (Zen 4, AVX-512)
- GPU: NVIDIA GeForce RTX 4080 SUPER 16 GB (sm_89), driver 617.42, CUDA 13.4
- RAM: 64 GB (63.7 GiB), NVMe SSD
- OS: Windows 11, Python 3.14
- Engine: Strata 0.1.56 (commit 06698c674673, CUDA 13.0, archs 75/86/89/120)

## Model / Config
qwen3.8-flash-next IQ3_S (norm), 128K context, KV int8, Vision on, PLE-fp8.

Launch (from `strata-iq3_s.json`):
```
strata.exe --pack iq3_s --native <qwen3.8-flash-next IQ3_S GGUF>
  --ple-gguf D:/Strata-data/ple/ple-fp8.gguf --expert-profile auto
  --expert-cache auto --prefill auto --spec 4 --max-context 131072 --kv int8
  --kv-resident 32768 --vision --vram-reserve-mib 700 --pcie-frac 0.20
  --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97
```

## Results (median of run3, closest to the 3-run median)
| metric | value |
|---|---|
| chat decode (tok/s) | 91.9 |
| chat TTFT (s) | 0.286 |
| 4K prompt (tok/s) | 2078.9 |
| 16K prompt (tok/s) | 3694.2 |
| agent decode (tok/s) | 108.9 |
| agent turn TTFT (s) | 1.765 |

3-run aggregate: chat 91.9 (91.1-92.2), agent 108.9 (107.1-110.4), 4K 2070.4, 16K 3698.0.

## Agent cache (3 sessions)
cache hit 70.03%, re-read 3 tok/session, new 12919 tok/session.

## Recall checks
6 of 6 needles found (32K and 128K, depths 10/50/90). Raw runs: needles.json
