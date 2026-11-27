# Hardware and runtime

- Minisforum AI X1-255; AMD Ryzen 7 255 (8 cores / 16 threads).
- 128 GB DDR5-5600 installed (operator-reported); Linux visible memory 121.34 GiB.
- Single NVIDIA GeForce RTX 3060 12 GiB, 170 W power limit; DEG1 OCuLink (operator-reported); all 995 sampled links Gen4 x4.
- Ubuntu kernel `7.0.0-44-generic`; NVIDIA driver `595.99.02`; CUDA reported by nvidia-smi `13.2`.
- Strata engine/control `0.1.69`; terminal `0.2.22`; bundled Python `3.12.15`.
- GPU Unlock ON (kernel log marker verified in the original archive).
- Model `qwen3.8-flash-next-q2_0`; local GGUF source files used, canonical prepared pack (operator-reported/prior pack inspection). No original GGUF files included.
- Original archive SHA256: `05a80c5fc81e48bb87fdaa059c2533ba0cc7023ea21702f35e9f664096b1a7a8`.
- Benchmark 2026-11-25 23:34 UTC to 2026-11-26 00:09 UTC.

Raw nvidia-smi -q and host logs intentionally excluded; no GPU UUID/PDI, local username or hostname included.
