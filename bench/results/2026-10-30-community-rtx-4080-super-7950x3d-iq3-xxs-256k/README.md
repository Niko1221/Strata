# Community benchmark: qwen3.8-flash-next-iq3_xxs @ 256K on RTX 4080 SUPER

Measured by: MASTA/Eich (XX) on a Ryzen 9 7950X3D build, 2026-10-30, with Edge closed
and all background load off (clean Firefox off too). Bench and needles via
`tools/strata_bench.py` / `tools/needle_bench.py` from `D:\Strata`.

## Hardware / Software

- CPU: AMD Ryzen 9 7950X3D 16-core/32-thread (Zen 4, AVX-512)
- GPU: NVIDIA GeForce RTX 4080 SUPER 16 GB (sm_89), driver 617.42
- RAM: 64 GB (63.7 GiB), NVMe SSD
- OS: Windows 11 (10.0.26200)
- Python: 3.14.4
- Engine: Strata 0.1.54 (commit 8c1aea4bca1b, CUDA 13.0, archs 75/86/89/120)

## Model / Config

qwen3.8-flash-next IQ3_XXS (norm), 256K context, KV int8, Vision on, PLE-fp8.

Launch string (from the installed `strata-iq3_xxs.json`):
```
strata.exe --pack iq3_xxs --native <qwen3.8-flash-next IQ3_XXS GGUF>
  --ple-gguf D:/Strata-data/ple/ple-fp8.gguf --expert-profile auto
  --expert-cache auto --prefill auto --spec 4 --mtp auto
  --max-context 262144 --kv int8 --kv-resident 32768 --vision
  --vram-reserve-mib 700 --pcie-frac 0.35 --spec-min-p 0.70
  --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```
Calibrated on the 0.1.54 engine (pcie-frac 0.35 / spec-min-p 0.70 / adapt 1-160-0.97).

## Results (median of 3 runs)

| metric | value |
|---|---|
| chat decode (tok/s) | 93.4 |
| chat TTFT (s) | 0.337 |
| 4K prompt (tok/s) | 2320.6 |
| 4K TTFT (s) | 1.759 |
| 16K prompt (tok/s) | 3526.4 |
| 16K TTFT (s) | 4.59 |
| agent decode (tok/s) | 114.2 |
| agent turn TTFT (s) | 1.684 |
| agent first-turn TTFT (s) | 1.035 |

## Agent cache

3 sessions, median over the agent runs (70.18% cache hit):

| metric | median | min | max |
|---|---|---|---|
| cache hit % | 70.18 | 70.04 | 70.29 |
| re-read tokens / session | 74 | -- | 71 |
| new tokens / session | 12769 | -- | -- |

(Median reported over the agent runs; `n=3` sessions.)

## Recall checks

6 of 6 needles found. Raw runs: needles.json

| length | depth | result |
|---|---|---|
| 32k | 10% | found |
| 32k | 50% | found |
| 32k | 90% | found |
| 256k | 10% | found |
| 256k | 50% | found |
| 256k | 90% | found |
