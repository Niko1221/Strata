# flash-next-strata-proxy — 20261016T185153Z

model `flash-next-strata` via `http://localhost:4000`; max_tokens 256; temperature none; cache_prompt false; ignore_eos true. plan step C TRIAL standard pass: v0.1.44.1 held config + --spec-min-p 0.7 (CONFIG strata-iq3_xxs.v0.1.44.1-held.minp0.7.json), PUT 18:47 UTC; control = 20261016T153800Z (gate 0.5)

| conc | prompt | rep | prompt_n | prefill tok/s | decode tok/s (per stream) | aggregate decode tok/s (wall) | wall s | ext load before→after | draft accept |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 256 | 0 | 358 | 182.2 | 44.80 | 33.319 | 7.683 | 0.0→0.0 | 0.880 |
| 1 | 256 | 1 | 359 | 184.9 | 40.90 | 31.162 | 8.215 | 0.0→0.0 | 0.822 |
| 1 | 1024 | 0 | 1116 | 314.7 | 36.40 | 24.179 | 10.588 | 0.0→0.0 | 0.712 |
| 1 | 1024 | 1 | 1115 | 313.0 | 36.60 | 24.214 | 10.572 | 0.0→0.0 | 0.697 |
| 1 | 4096 | 0 | 4123 | 356.1 | 35.40 | 13.596 | 18.828 | 0.0→0.0 | 0.775 |
| 1 | 4096 | 1 | 4118 | 356.2 | 38.10 | 13.99 | 18.299 | 0.0→0.0 | 0.723 |
| 1 | 16384 | 0 | 16510 | 560.1 | 36.40 | 7.004 | 36.55 | 0.0→0.0 | 0.703 |
| 1 | 32768 | 0 | 33261 | 650.4 | 37.10 | 4.406 | 58.108 | 0.0→0.0 | 0.826 |

TTFT (streaming, 64 tokens): 4.035 s, 3.58 s
