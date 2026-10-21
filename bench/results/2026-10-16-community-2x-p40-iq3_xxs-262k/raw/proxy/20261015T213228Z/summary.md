# flash-next-strata-proxy — 20261015T213228Z

model `flash-next-strata` via `http://localhost:4000`; max_tokens 1024; temperature none; cache_prompt false; ignore_eos true. E2 BARE-METAL, long-decode pass: host-built Strata on the same ctx256k-fit config, sampled, via proxy, 2026-10-15

| conc | prompt | rep | prompt_n | prefill tok/s | decode tok/s (per stream) | aggregate decode tok/s (wall) | wall s | ext load before→after | draft accept |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 0 | 4124 | 319.5 | 16.60 | 13.727 | 74.595 | 0.0→0.0 | 0.574 |
| 1 | 4096 | 1 | 4120 | 320.2 | 17.00 | 13.992 | 73.183 | 0.0→0.0 | 0.583 |
| 1 | 4096 | 2 | 4124 | 318.6 | 16.80 | 13.842 | 73.976 | 0.0→0.0 | 0.593 |

TTFT (streaming, 64 tokens): 4.46 s, 4.382 s
