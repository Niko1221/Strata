# flash-next-strata-proxy — 20261016T142855Z

model `flash-next-strata` via `http://localhost:4000`; max_tokens 1024; temperature none; cache_prompt false; ignore_eos true. v0.1.40.4 TRIAL long-decode pass (plan step 10) vs controls 20261016T142027Z and 20261015T210945Z

| conc | prompt | rep | prompt_n | prefill tok/s | decode tok/s (per stream) | aggregate decode tok/s (wall) | wall s | ext load before→after | draft accept |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 0 | 4123 | 335.6 | 35.40 | 24.827 | 41.245 | 0.0→0.0 | 0.671 |
| 1 | 4096 | 1 | 4123 | 337.0 | 32.60 | 23.455 | 43.658 | 0.0→1.0 | 0.623 |
| 1 | 4096 | 2 | 4122 | 334.3 | 33.20 | 21.962 | 46.626 | 1.0→1.0 | 0.588 |

TTFT (streaming, 64 tokens): 8.25 s, 4.152 s
