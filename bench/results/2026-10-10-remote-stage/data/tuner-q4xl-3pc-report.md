# Layer-split tuning

Model: 48 layers, 75.6 GiB of layer tensors. Max context 131072. Searched with 32000-token prompts and 500-token answers (score: a request's seconds).

## Search

| stages (layers) | chunk | prefill tok/s | decode tok/s | short decode | request s | per stage: prompt chunk ms / window ms |
|---|---|---|---|---|---|---|
| 0-23 24-39 40-47 | 5888  | 950 | 35.9 | 37.1 | 45.7 | 3333 / 41.4; 2611 / 14.0; 1246 / 6.5 |
| 0-17 18-33 34-47 | 5888  | 926 | 36.9 | 40.4 | 48.0 | 2424 / 29.2; 2418 / 14.5; 2122 / 14.7 |
| 0-21 22-39 40-47 | 5888  | 960 | 35.7 | 38.6 | 47.4 | 3000 / 38.5; 2867 / 16.3; 1158 / 6.9 |
| 0-25 26-39 40-47 | 5888  | 1007 | 32.4 | 36.9 | 47.3 | 3622 / 46.3; 2259 / 11.8; 1139 / 6.5 |
| 0-23 24-37 38-47 | 5888  | 984 | 33.9 | 37.1 | 47.3 | 3327 / 42.1; 2132 / 11.8; 1598 / 8.8 |
| 0-23 24-41 42-47 | 5888  | 979 | 33.6 | 36.8 | 47.6 | 3330 / 40.1; 2841 / 15.4; 980 / 4.7 |
| 0-22 23-39 40-47 | 5888  | 959 | 34.3 | 35.4 | 48.0 | 3163 / 37.0; 2697 / 15.2; 1187 / 6.5 |
| 0-24 25-39 40-47 | 5888  | 988 | 33.5 | 37.3 | 47.4 | 3484 / 42.2; 2413 / 12.8; 1197 / 6.5 |

## Workload profile [(8000, 0.3), (32000, 0.5), (100000, 0.2)]

| stages | chunk | 8K prefill / decode | 32K prefill / decode | 100K prefill / decode | weighted request s |
|---|---|---|---|---|---|
| 0-23 24-39 40-47 | 5888 | 552 / 33.1 | 975 / 35.2 | 1383 / 33.6 | 49.1 |
| 0-25 26-39 40-47 | 5888 | 592 / 34.7 | 1016 / 31.6 | 1332 / 34.5 | 49.3 |

## Near the max context (131072)

- 0-23 24-39 40-47, chunk 5888: 125230 tokens read at 1410 tok/s, decode 33.8

**Pick**: stages 0-23 24-39 40-47, chunk 5888 

Run it: `python3 tools/stage_tune.py apply cluster.json OUTDIR` starts the workers, then serve `OUTDIR/main.json` (serve/server.py --config).
