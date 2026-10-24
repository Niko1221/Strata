# flash-next-strata-proxy — 20261016T153800Z

model `flash-next-strata` via `http://localhost:4000`; max_tokens 256; temperature none; cache_prompt false; ignore_eos true. v0.1.44.1 TRIAL standard pass (plan step 14): image 54dd82ad9286 + held config (ROUTE_TAIL_SKIP=0, PREFILL_CPU_SHARE=0), deployed 15:36 UTC by Portainer PUT; control = v0.1.40.4 runs 20261016T142535Z / T142855Z

| conc | prompt | rep | prompt_n | prefill tok/s | decode tok/s (per stream) | aggregate decode tok/s (wall) | wall s | ext load before→after | draft accept |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 256 | 0 | 356 | 172.9 | 36.90 | 21.185 | 12.084 | 1.0→1.0 | 0.567 |
| 1 | 256 | 1 | 354 | 167.4 | 38.40 | 14.519 | 17.632 | 1.0→1.0 | 0.619 |
| 1 | 1024 | 0 | 1116 | 286.4 | 37.40 | 14.958 | 17.114 | 1.0→0.0 | 0.583 |
| 1 | 1024 | 1 | 1119 | 316.1 | 37.20 | 24.553 | 10.426 | 0.0→0.0 | 0.557 |
| 1 | 4096 | 0 | 4125 | 356.5 | 37.40 | 13.89 | 18.431 | 0.0→0.0 | 0.602 |
| 1 | 4096 | 1 | 4122 | 356.4 | 36.90 | 13.824 | 18.518 | 0.0→0.0 | 0.580 |
| 1 | 16384 | 0 | 16508 | 560.6 | 36.50 | 7.016 | 36.486 | 0.0→0.0 | 0.604 |
| 1 | 32768 | 0 | 33260 | 652.8 | 35.60 | 4.399 | 58.191 | 0.0→1.0 | 0.587 |

TTFT (streaming, 64 tokens): 38.604 s, 4.616 s
