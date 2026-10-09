# Layer-split tuning

Goal: a 32000-token prompt and a 500-token answer (score: the request's seconds).

| stages (layers) | chunk | prefill tok/s | decode tok/s | short decode | request s | per stage: prompt chunk ms / window ms |
|---|---|---|---|---|---|---|
| 0-21 22-47 | 4096 | 1035 | 49.9 | 51.6 | 40.9 | 1964 / 23.3; 3132 / 20.1 |
| 0-27 28-47 | 4096 | 1174 | 41.5 | 46.1 | 39.3 | 2686 / 32.0; 2123 / 15.1 |
| 0-27 28-47 | 5888 | 1221 | 43.7 | 49.1 | 37.6 | 3031 / 31.7; 2630 / 15.3 |
| 0-27 28-47 | 8192 | - | - | - | does not run | |
| 0-27 28-47 | 8192 | 1235 | 42.2 | 46.2 | 37.8 | 3495 / 36.5; 3195 / 16.4 |
| 0-25 26-47 **best** | 5888 | 1230 | 48.7 | 49.3 | 36.3 | 2753 / 28.6; 2857 / 15.3 |
| 0-23 24-47 | 5888 | 1230 | 44.0 | 49.7 | 37.4 | 2533 / 26.4; 3200 / 17.1 |
| 0-24 25-47 | 5888 | 1199 | 47.8 | 50.4 | 37.2 | 2657 / 28.5; 3235 / 16.1 |
| 0-26 27-47 | 5888 | 1239 | 45.9 | 50.2 | 36.7 | 2862 / 31.2; 2813 / 15.1 |

Run it: `python3 tools/stage_tune.py apply cluster.json OUTDIR` starts the workers, then serve `OUTDIR/main.json` (serve/server.py --config).
