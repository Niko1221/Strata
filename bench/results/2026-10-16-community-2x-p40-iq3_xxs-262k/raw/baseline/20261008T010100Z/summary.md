# baseline-qwen38-dense-q6-proxy — 20261008T010100Z

model `qwen38` via `http://localhost:4000`; max_tokens 256; temperature 0; cache_prompt false; ignore_eos true. A2 part 1: proxy path, alias qwen38 = qwen38-dense-q6. Box: load 1.1 at start, GPUs idle, qwen38-dense-q6 bounced 21:22 UTC 2026-10-07; gateway traffic may appear (see ext load).

| conc | prompt | rep | prompt_n | prefill tok/s | decode tok/s (per stream) | aggregate decode tok/s (wall) | wall s | ext load before→after |
|---|---|---|---|---|---|---|---|---|
| 1 | 256 | 0 | 355 | 177.5 | 23.25 | 19.307 | 13.26 | 0.0→0.0 |
| 1 | 256 | 1 | 357 | 179.3 | 14.43 | 12.733 | 20.104 | 0.0→1.0 |
| 1 | 4096 | 0 | 4122 | 318.5 | 14.68 | 7.249 | 35.313 | 1.0→0.0 |
| 1 | 4096 | 1 | 4120 | 317.2 | 14.68 | 8.296 | 30.858 | 0.0→0.0 |
| 1 | 16384 | 0 | 16514 | 322.2 | 13.54 | 3.62 | 70.715 | 0.0→0.0 |

TTFT (streaming, 64 tokens): 2.97 s, 2.293 s
