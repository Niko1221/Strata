# flash-next-strata-proxy — 20261016T142027Z

model `flash-next-strata` via `http://localhost:4000`; max_tokens 1024; temperature none; cache_prompt false; ignore_eos true. 0.1.40.3 same-session CONTROL for the v0.1.40.4 upgrade A/B (plan step 3), stack 50 restarted 14:19 UTC on image 69dca0eca2b5

| conc | prompt | rep | prompt_n | prefill tok/s | decode tok/s (per stream) | aggregate decode tok/s (wall) | wall s | ext load before→after | draft accept |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 0 | 4120 | 314.1 | 17.50 | 14.278 | 71.716 | 0.0→0.0 | 0.559 |
| 1 | 4096 | 1 | 4118 | 313.3 | 18.00 | 14.646 | 69.916 | 0.0→0.0 | 0.604 |
| 1 | 4096 | 2 | 4118 | 317.4 | 18.10 | 14.696 | 69.678 | 0.0→0.0 | 0.597 |

TTFT (streaming, 64 tokens): 4.588 s, 4.509 s
