# flash-next-strata-proxy — 20261015T212756Z

model `flash-next-strata` via `http://localhost:4000`; max_tokens 256; temperature none; cache_prompt false; ignore_eos true. E2 BARE-METAL, standard pass: host-built Strata (nvcc 12.9.2, GCC 12.3) on the same ctx256k-fit config, sampled, via proxy, 2026-10-15

| conc | prompt | rep | prompt_n | prefill tok/s | decode tok/s (per stream) | aggregate decode tok/s (wall) | wall s | ext load before→after | draft accept |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 256 | 0 | 357 | 144.5 | 17.50 | 14.984 | 17.085 | 0.0→0.0 | 0.630 |
| 1 | 256 | 1 | 357 | 145.6 | 18.30 | 15.573 | 16.439 | 0.0→0.0 | 0.676 |
| 1 | 1024 | 0 | 1116 | 246.1 | 17.30 | 13.255 | 19.314 | 0.0→0.0 | 0.605 |
| 1 | 1024 | 1 | 1118 | 248.1 | 15.80 | 12.333 | 20.757 | 0.0→0.0 | 0.568 |
| 1 | 4096 | 0 | 4123 | 317.4 | 17.20 | 9.184 | 27.876 | 0.0→0.0 | 0.619 |
| 1 | 4096 | 1 | 4123 | 318.2 | 17.40 | 9.249 | 27.68 | 0.0→0.0 | 0.624 |
| 1 | 16384 | 0 | 16514 | 502.2 | 16.20 | 5.24 | 48.853 | 0.0→0.0 | 0.558 |
| 1 | 32768 | 0 | 33260 | 588.5 | 16.50 | 3.541 | 72.3 | 0.0→0.0 | 0.579 |

TTFT (streaming, 64 tokens): 4.997 s, 4.544 s
