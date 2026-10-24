# flash-next-strata-proxy — 20261016T142535Z

model `flash-next-strata` via `http://localhost:4000`; max_tokens 256; temperature none; cache_prompt false; ignore_eos true. v0.1.40.4 TRIAL standard pass (plan step 10): image 3753c531cb42 deployed 14:24 UTC by Portainer PUT; Pascal decode fix #1469

| conc | prompt | rep | prompt_n | prefill tok/s | decode tok/s (per stream) | aggregate decode tok/s (wall) | wall s | ext load before→after | draft accept |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 256 | 0 | 355 | 152.5 | 33.30 | 25.547 | 10.021 | 0.0→0.0 | 0.596 |
| 1 | 256 | 1 | 356 | 154.3 | 33.20 | 25.545 | 10.021 | 0.0→0.0 | 0.589 |
| 1 | 1024 | 0 | 1119 | 262.3 | 35.10 | 22.104 | 11.582 | 0.0→0.0 | 0.638 |
| 1 | 1024 | 1 | 1114 | 260.2 | 32.50 | 21.005 | 12.188 | 0.0→0.0 | 0.532 |
| 1 | 4096 | 0 | 4123 | 332.1 | 36.00 | 13.084 | 19.565 | 0.0→0.0 | 0.684 |
| 1 | 4096 | 1 | 4124 | 332.3 | 44.80 | 14.094 | 18.164 | 0.0→0.0 | 0.877 |
| 1 | 16384 | 0 | 16512 | 531.4 | 29.80 | 6.437 | 39.771 | 0.0→0.0 | 0.510 |
| 1 | 32768 | 0 | 33256 | 622.7 | 33.30 | 4.174 | 61.326 | 0.0→0.0 | 0.637 |

TTFT (streaming, 64 tokens): 4.779 s, 4.268 s
