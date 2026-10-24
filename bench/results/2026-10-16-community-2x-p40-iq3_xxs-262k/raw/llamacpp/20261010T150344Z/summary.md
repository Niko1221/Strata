# flash-next-llamacpp-proxy — 20261010T150344Z

model `flash-next-llamacpp` via `http://localhost:4000`; max_tokens 256; temperature 0; cache_prompt false; ignore_eos true. C10 first table: llama.cpp b11330, IQ3_XXS, --fit placement (VRAM 20.9/20.8 GB), q8_0 KV, FA on, lazy n-gram shard, 1 slot, no MTP. C10 depth A/B step 1: --fit-target 1024 (was 2048). Placement from -lv 4: GPU0 25 layers (1 overflowing), GPU1 24 (7 overflowing) + draft; model buffers 20,150/18,410 MiB; CUDA_Host 6,290 MiB (~8 layers' experts on CPU, was ~11 / 8,311 MiB). Base = revert run 20261010T145646Z (target 2048). Stack 49 recreated 15:03 UTC; load 31 s; idle RSS 7.8 GB; VRAM 21.64/21.72 GB.

| conc | prompt | rep | prompt_n | prefill tok/s | decode tok/s (per stream) | aggregate decode tok/s (wall) | wall s | ext load before→after | draft accept |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 256 | 0 | 356 | 144.1 | 21.85 | 17.807 | 14.376 | 0.0→0.0 | 0.494 |
| 1 | 256 | 1 | 360 | 147.6 | 23.24 | 18.688 | 13.698 | 0.0→0.0 | 0.522 |
| 1 | 4096 | 0 | 4120 | 231.4 | 21.57 | 8.569 | 29.876 | 0.0→0.0 | 0.515 |
| 1 | 4096 | 1 | 4123 | 234.1 | 22.63 | 8.764 | 29.21 | 0.0→0.0 | 0.525 |
| 1 | 16384 | 0 | 16515 | 217.2 | 20.77 | 2.887 | 88.66 | 0.0→0.0 | 0.531 |

TTFT (streaming, 64 tokens): 2.864 s, 2.647 s
