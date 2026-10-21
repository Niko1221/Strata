# flash-next-strata-direct-concurrency — 20261016T194933Z

model `flash-next-strata` via `http://localhost:8080`; max_tokens 256; temperature 0.0; cache_prompt false; ignore_eos true. §5.4 row 2 re-measure 2026-10-16 on v0.1.44.1: held config + --batch 2 --batch-groups 2 (strata-iq3_xxs.v0.1.44.1-held.batch2.json), 2 streams DIRECT :8080, A/B #4 / A2 protocol; gate = aggregate >= 16.8 short / 10.3 at 4k (A2 qwen38-dense-q6 2 direct streams)

| conc | prompt | rep | prompt_n | prefill tok/s | decode tok/s (per stream) | aggregate decode tok/s (wall) | wall s | ext load before→after | draft accept |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 256 | 0 | 353, 357 | 187.5, 182.5 | 27.00, 22.00 | 37.003 | 13.837 | 0.0→0.0 | — |
| 2 | 256 | 1 | 357, 354 | 193.4, 190.4 | 22.30, 27.30 | 37.763 | 13.558 | 0.0→0.0 | — |
| 2 | 4096 | 0 | 4121, 4121 | 351.8, 354.4 | 11.90, 26.80 | 15.335 | 33.389 | 0.0→0.0 | — |
| 2 | 4096 | 1 | 4122, 4122 | 349.5, 354.3 | 12.00, 26.90 | 15.318 | 33.425 | 0.0→0.0 | — |

TTFT (streaming, 64 tokens): 2.054 s, 1.816 s
