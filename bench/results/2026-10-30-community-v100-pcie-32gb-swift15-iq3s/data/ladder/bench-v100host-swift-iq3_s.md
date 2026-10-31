## v100host-swift-iq3_s (2026-10-29T16:40:10Z)

v100host V100 Strata strata-swift-iq3_s-serve.json, 80 threads, scaling_max_freq 2700000 kHz; v41bench prompt set v1 (Qwen copy, enable_thinking false); 127.0.0.1

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 30.24 | 40.75 | 1.486 |
| C2 | 30.96 | 20.82 | 2.572 |
| C4 | 33.98 | 22.33 | 7.665 |
| C8 | 36.27 | 23.01 | 16.742 |
| C16 | 37.88 | 23.4 | 33.776 |

### Per-stream tok/s by category

| category | C1 | C2 | C4 | C8 | C16 |
|---|---|---|---|---|---|
| coding | 43.84 | 20.76 | 21.81 | 21.91 | 22.88 |
| json | 43.66 | 21.56 | 21.09 | 23.31 | 23.34 |
| narrative | 40.43 | 22.31 | 23.3 | 26.63 | 26.28 |
| prose | 42.46 | 24.18 | 26.2 | 26.05 | 24.97 |
| math | 39.22 | 19.66 | 21.69 | 21.91 | 22.21 |
| reasoning | 45.49 | 21.96 | 23.18 | 23.52 | 23.79 |
| summary | 32.83 | 18.01 | 21.04 | 19.39 | 20.79 |
| format | 38.07 | 18.14 | 20.31 | 21.37 | 22.97 |
| ceiling_count | 62.63 | 30.06 | 31.08 | 32.11 | 31.12 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 5075 | 4.221 | 1202.3 |
| 8000 | 20117 | 14.574 | 1380.3 |
| 32000 | 80629 | 57.262 | 1408.1 |
| 64000 | 161257 | 130.07 | 1239.8 |
