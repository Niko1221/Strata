# flash-next-strata-proxy — 20261015T210945Z

model `flash-next-strata` via `http://localhost:4000`; max_tokens 1024; temperature none; cache_prompt false; ignore_eos true. E2 bare-metal CONTROL, long-decode pass: container (stack 50) on ctx256k-fit, sampled, via proxy, 2026-10-15

| conc | prompt | rep | prompt_n | prefill tok/s | decode tok/s (per stream) | aggregate decode tok/s (wall) | wall s | ext load before→after | draft accept |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 0 | 4124 | 320.3 | 17.80 | 14.517 | 70.538 | 0.0→0.0 | 0.590 |
| 1 | 4096 | 1 | 4123 | 320.7 | 17.80 | 14.556 | 70.35 | 0.0→0.0 | 0.590 |
| 1 | 4096 | 2 | 4125 | 320.5 | 18.00 | 14.663 | 69.834 | 0.0→0.0 | 0.614 |

TTFT (streaming, 64 tokens): 4.333 s, 4.355 s
