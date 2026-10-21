# flash-next-strata-direct-concurrency — 20261014T125520Z

model `flash-next-strata` via `http://localhost:8080`; max_tokens 256; temperature 0; cache_prompt false; ignore_eos true. A/B #4 CONTROL (FIFO, no batch slots): 2 streams DIRECT :8080, D8 part 2 protocol; trim config (explicit split 25 + STRATA_STAGE_TRIM=1).

| conc | prompt | rep | prompt_n | prefill tok/s | decode tok/s (per stream) | aggregate decode tok/s (wall) | wall s | ext load before→after | draft accept |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 256 | 0 | 353, 355 | 146.5, 157.5 | 17.00, 18.60 | 15.268 | 33.533 | 0.0→0.0 | 0.604, 0.696 |
| 2 | 4096 | 0 | 4124, 4120 | 323.1, 321.7 | 18.70, 18.10 | 9.58 | 53.444 | 0.0→0.0 | 0.730, 0.684 |

TTFT (streaming, 64 tokens): 2.339 s, 2.367 s
