# flash-next-strata-proxy — 20261016T195116Z

model `flash-next-strata` via `http://localhost:4000`; max_tokens 1024; temperature none; cache_prompt false; ignore_eos true. §5.4 row 2 re-measure: SOLO long-decode pass on the batch-slot variant (cost of the slots to a request alone) vs held control 20261016T154228Z

| conc | prompt | rep | prompt_n | prefill tok/s | decode tok/s (per stream) | aggregate decode tok/s (wall) | wall s | ext load before→after | draft accept |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 0 | 4122 | 357.3 | 37.00 | 26.118 | 39.207 | 0.0→0.0 | 0.598 |
| 1 | 4096 | 1 | 4121 | 348.9 | 35.50 | 25.201 | 40.633 | 0.0→0.0 | 0.564 |
| 1 | 4096 | 2 | 4123 | 353.1 | 35.80 | 25.43 | 40.267 | 0.0→0.0 | 0.591 |

TTFT (streaming, 64 tokens): 3.706 s, 3.748 s
