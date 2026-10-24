# flash-next-strata-proxy — 20261015T210525Z

model `flash-next-strata` via `http://localhost:4000`; max_tokens 256; temperature none; cache_prompt false; ignore_eos true. E2 bare-metal CONTROL, standard pass: container (stack 50) on ctx256k-fit, sampled, via proxy, 2026-10-15

| conc | prompt | rep | prompt_n | prefill tok/s | decode tok/s (per stream) | aggregate decode tok/s (wall) | wall s | ext load before→after | draft accept |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 256 | 0 | 356 | 156.0 | 17.70 | 15.233 | 16.805 | 0.0→0.0 | 0.565 |
| 1 | 256 | 1 | 355 | 150.8 | 26.70 | 21.421 | 11.951 | 0.0→0.0 | 0.910 |
| 1 | 1024 | 0 | 1117 | 259.7 | 18.50 | 14.09 | 18.169 | 0.0→0.0 | 0.638 |
| 1 | 1024 | 1 | 1116 | 261.3 | 16.80 | 13.097 | 19.546 | 0.0→0.0 | 0.530 |
| 1 | 4096 | 0 | 4125 | 321.2 | 17.80 | 9.395 | 27.249 | 0.0→0.0 | 0.614 |
| 1 | 4096 | 1 | 4125 | 320.7 | 17.20 | 9.222 | 27.76 | 0.0→0.0 | 0.553 |
| 1 | 16384 | 0 | 16511 | 505.0 | 18.90 | 5.522 | 46.356 | 0.0→0.0 | 0.668 |
| 1 | 32768 | 0 | 33256 | 593.0 | 17.00 | 3.585 | 71.412 | 0.0→0.0 | 0.563 |

TTFT (streaming, 64 tokens): 4.709 s, 4.325 s
