# flash-next-strata-proxy — 20261016T154228Z

model `flash-next-strata` via `http://localhost:4000`; max_tokens 1024; temperature none; cache_prompt false; ignore_eos true. v0.1.44.1 TRIAL long-decode pass (plan step 14) vs v0.1.40.4 20261016T142855Z

| conc | prompt | rep | prompt_n | prefill tok/s | decode tok/s (per stream) | aggregate decode tok/s (wall) | wall s | ext load before→after | draft accept |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 0 | 4124 | 360.5 | 36.00 | 25.656 | 39.913 | 0.0→0.0 | 0.582 |
| 1 | 4096 | 1 | 4118 | 361.4 | 35.70 | 25.558 | 40.065 | 0.0→1.0 | 0.573 |
| 1 | 4096 | 2 | 4122 | 359.0 | 40.30 | 25.129 | 40.75 | 1.0→1.0 | 0.707 |

TTFT (streaming, 64 tokens): 4.811 s, 3.441 s
