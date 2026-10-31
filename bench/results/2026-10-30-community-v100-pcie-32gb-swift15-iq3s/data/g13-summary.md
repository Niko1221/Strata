| arm | ctx | task | n | decode tok/s median (min-max) | range % | prompt tok/s median | VRAM share | drafts acc. | win ms | CPU-expert ms | hashes |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| bench-prod | 4096 | code | 3 | 81.62 (81.34-81.98) | 0.8 | 1128.4 | 0.96 | 0.78 | - | - | same 0855b916 |
| bench-prod | 4096 | prose | 3 | 57.28 (56.9-57.32) | 0.7 | 1089.3 | 0.95 | 0.48 | - | - | DIFFER 0050567a,14d9f0f3 |
| bench-prod | 16384 | code | 3 | 87.21 (85.32-87.81) | 2.9 | 1504.7 | 0.96 | 0.86 | - | - | DIFFER aec0fddb,c5b7c06d |
| bench-prod | 16384 | prose | 3 | 59.97 (59.58-60.0) | 0.7 | 1476.7 | 0.98 | 0.48 | - | - | DIFFER 8689b79e,dbd90842 |
| bench-prod-repro | 4096 | code | 1 | 68.9 (68.9-68.9) | 0.0 | 1285.5 | None | 0.77 | - | - | same db0feac5 |
| bench-prod-repro | 4096 | prose | 1 | 41.67 (41.67-41.67) | 0.0 | 1300.5 | None | 0.42 | - | - | same c75b0986 |
| bench-prod-repro | 16384 | code | 1 | 74.04 (74.04-74.04) | 0.0 | 1586.0 | None | 0.86 | - | - | same 1d7076cc |
| bench-prod-repro | 16384 | prose | 1 | 44.14 (44.14-44.14) | 0.0 | 1596.2 | None | 0.5 | - | - | same eb9bb5da |
| bench-prof | 4096 | code | 1 | 76.11 (76.11-76.11) | 0.0 | 1132.3 | 0.96 | 0.78 | - | - | same 0855b916 |
| bench-prof | 4096 | prose | 1 | 52.18 (52.18-52.18) | 0.0 | 1101.9 | 0.95 | 0.48 | - | - | same 0050567a |
| bench-rel | 4096 | code | 3 | 81.26 (81.05-81.38) | 0.4 | 1126.0 | 0.96 | 0.78 | - | - | same 0855b916 |
| bench-rel | 4096 | prose | 3 | 57.05 (57.0-57.31) | 0.5 | 1103.3 | 0.95 | 0.47 | - | - | DIFFER 0050567a,14d9f0f3 |
| bench-rel | 16384 | code | 3 | 85.18 (85.18-87.49) | 2.7 | 1500.6 | 0.96 | 0.83 | - | - | DIFFER aec0fddb,c5b7c06d |
| bench-rel | 16384 | prose | 3 | 59.55 (59.4-60.01) | 1.0 | 1483.8 | 0.97 | 0.49 | - | - | DIFFER 8689b79e,dbd90842 |
| bench-rel-repro | 4096 | code | 2 | 68.58 (68.27-68.89) | 0.9 | 1285.8 | None | 0.77 | - | - | same db0feac5 |
| bench-rel-repro | 4096 | prose | 2 | 41.86 (41.66-42.06) | 1.0 | 1299.8 | None | 0.42 | - | - | same c75b0986 |
| bench-rel-repro | 16384 | code | 2 | 73.81 (73.57-74.04) | 0.6 | 1582.25 | None | 0.86 | - | - | same 1d7076cc |
| bench-rel-repro | 16384 | prose | 2 | 44.38 (44.15-44.62) | 1.1 | 1592.95 | None | 0.49 | - | - | DIFFER 48a7a15d,eb9bb5da |
| serve-prof | 16384 | code | 1 | 52.2 (52.2-52.2) | 0.0 | 1484.3 | 0.8 | 0.81 | - | - | same 869a7594 |
| serve-prof | 16384 | prose | 1 | 39.56 (39.56-39.56) | 0.0 | 1438.8 | 0.83 | 0.47 | - | - | same 015a235a |
| serve-rel | 4096 | code | 3 | 55.29 (54.54-55.54) | 1.8 | 1005.0 | 0.8 | 0.83 | - | - | same 15c04957 |
| serve-rel | 4096 | prose | 3 | 41.03 (40.95-41.4) | 1.1 | 981.4 | 0.81 | 0.48 | - | - | same c462f799 |
| serve-rel | 16384 | code | 3 | 49.74 (49.28-50.27) | 2.0 | 1471.0 | 0.74 | 0.8 | - | - | same 27342b1d |
| serve-rel | 16384 | prose | 3 | 43.62 (43.61-44.06) | 1.0 | 1446.5 | 0.85 | 0.47 | - | - | same 8d6dfadc |
| serve-rel | 131072 | code | 3 | 49.93 (49.85-50.57) | 1.4 | 1317.8 | 0.74 | 0.83 | - | - | DIFFER 6abec92e,e2065bdd |
| serve-rel | 131072 | prose | 3 | 45.67 (43.79-46.14) | 5.1 | 1278.2 | 0.86 | 0.54 | - | - | DIFFER 2ead263d,4984b2c3 |

Against `bench-prod` (median decode, and the larger of the two arms' ranges as the MDE):

| arm | ctx | task | delta % | MDE % | reading | prompt delta % |
|---|---:|---|---:|---:|---|---:|
| bench-prod-repro | 16384 | code | -15.1 | 2.9 | slower | 5.4 |
| bench-prod-repro | 16384 | prose | -26.4 | 0.7 | slower | 8.1 |
| bench-prod-repro | 4096 | code | -15.6 | 0.8 | slower | 13.9 |
| bench-prod-repro | 4096 | prose | -27.3 | 0.7 | slower | 19.4 |
| bench-prof | 4096 | code | -6.8 | 0.8 | slower | 0.3 |
| bench-prof | 4096 | prose | -8.9 | 0.7 | slower | 1.2 |
| bench-rel-repro | 16384 | code | -15.4 | 2.9 | slower | 5.2 |
| bench-rel-repro | 16384 | prose | -26.0 | 1.1 | slower | 7.9 |
| bench-rel-repro | 4096 | code | -16.0 | 0.9 | slower | 13.9 |
| bench-rel-repro | 4096 | prose | -26.9 | 1.0 | slower | 19.3 |
| bench-rel | 16384 | code | -2.3 | 2.9 | within noise | -0.3 |
| bench-rel | 16384 | prose | -0.7 | 1.0 | within noise | 0.5 |
| bench-rel | 4096 | code | -0.4 | 0.8 | within noise | -0.2 |
| bench-rel | 4096 | prose | -0.4 | 0.7 | within noise | 1.3 |
| serve-prof | 16384 | code | -40.1 | 2.9 | slower | -1.4 |
| serve-prof | 16384 | prose | -34.0 | 0.7 | slower | -2.6 |
| serve-rel | 16384 | code | -43.0 | 2.9 | slower | -2.2 |
| serve-rel | 16384 | prose | -27.3 | 1.0 | slower | -2.0 |
| serve-rel | 4096 | code | -32.3 | 1.8 | slower | -10.9 |
| serve-rel | 4096 | prose | -28.4 | 1.1 | slower | -9.9 |
