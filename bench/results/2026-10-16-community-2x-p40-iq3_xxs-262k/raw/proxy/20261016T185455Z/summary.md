# flash-next-strata-proxy — 20261016T185455Z

model `flash-next-strata` via `http://localhost:4000`; max_tokens 1024; temperature none; cache_prompt false; ignore_eos true. plan step C TRIAL long-decode pass: --spec-min-p 0.7 vs control 20261016T154228Z (gate 0.5)

| conc | prompt | rep | prompt_n | prefill tok/s | decode tok/s (per stream) | aggregate decode tok/s (wall) | wall s | ext load before→after | draft accept |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 0 | 4120 | 358.9 | 36.50 | 25.916 | 39.512 | 0.0→0.0 | 0.754 |
| 1 | 4096 | 1 | 4120 | 360.4 | 41.30 | 28.265 | 36.229 | 0.0→0.0 | 0.850 |
| 1 | 4096 | 2 | 4118 | 360.1 | 35.30 | 25.292 | 40.488 | 0.0→0.0 | 0.738 |

TTFT (streaming, 64 tokens): 3.522 s, 3.447 s
