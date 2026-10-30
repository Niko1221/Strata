## v100host-swift-serve-s2 (2026-10-30T00:33:40Z)

v100host V100 Strata dev rel build, serve-s2 (serve-s2.json), scaling_max_freq 2700000 kHz; v41bench prompt set v1 (Qwen copy, enable_thinking false); 127.0.0.1:8081

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 42.46 | 58.11 | 1.091 |
| C2 | 38.8 | 25.99 | 1.989 |
| C4 | 41.99 | 26.74 | 5.988 |

### Per-stream tok/s by category

| category | C1 | C2 | C4 |
|---|---|---|---|
| coding | 62.73 | 26.96 | 25.7 |
| json | 63.53 | 26.31 | 27.7 |
| narrative | 55.47 | 24.28 | 28.99 |
| prose | 52.63 | 29.86 | 26.57 |
| math | 56.55 | 25.31 | 27.93 |
| reasoning | 65.76 | 28.65 | 29.64 |
| summary | 47.65 | 22.66 | 21.84 |
| format | 60.52 | 23.87 | 25.53 |
| ceiling_count | 83.59 | 32.48 | 33.44 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
