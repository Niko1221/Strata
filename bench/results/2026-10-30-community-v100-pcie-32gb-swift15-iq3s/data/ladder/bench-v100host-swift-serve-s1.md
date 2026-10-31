## v100host-swift-serve-s1 (2026-10-30T00:38:55Z)

v100host V100 Strata dev rel build, serve-s1 (serve-s1.json), scaling_max_freq 2700000 kHz; v41bench prompt set v1 (Qwen copy, enable_thinking false); 127.0.0.1:8081

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 58.07 | 81.6 | 0.868 |
| C2 | 60.08 | 86.78 | 2.403 |
| C4 | 63.0 | 92.12 | 5.112 |

### Per-stream tok/s by category

| category | C1 | C2 | C4 |
|---|---|---|---|
| coding | 86.25 | 100.36 | 111.15 |
| json | 90.77 | 93.99 | 101.24 |
| narrative | 65.23 | 63.62 | 66.89 |
| prose | 72.97 | 70.78 | 70.63 |
| math | 88.51 | 100.09 | 110.03 |
| reasoning | 87.15 | 95.68 | 98.49 |
| summary | 72.04 | 71.73 | 73.68 |
| format | 89.88 | 97.98 | 104.89 |
| ceiling_count | 114.49 | 124.74 | 128.15 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
