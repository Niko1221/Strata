| arm | ctx | task | n | decode tok/s median (min-max) | range % | prompt tok/s median | VRAM share | drafts acc. | win ms | CPU-expert ms | hashes |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| bench-nsys | 4096 | code | 1 | 73.64 (73.64-73.64) | 0.0 | 1100.6 | 0.95 | 0.83 | - | - | same 844182a1 |
| bench-ref | 4096 | code | 3 | 80.96 (80.79-81.12) | 0.4 | 1128.0 | 0.96 | 0.78 | - | - | same 0855b916 |
| bench-ref | 4096 | prose | 3 | 56.5 (56.47-56.61) | 0.2 | 1089.1 | 0.95 | 0.47 | - | - | same 14d9f0f3 |
| bench-ref | 16384 | code | 3 | 84.84 (84.8-85.0) | 0.2 | 1498.3 | 0.96 | 0.83 | - | - | same aec0fddb |
| bench-ref | 16384 | prose | 3 | 59.36 (59.3-59.53) | 0.4 | 1471.4 | 0.97 | 0.49 | - | - | same 8689b79e |
| bench-ref-repro | 4096 | code | 2 | 68.41 (68.39-68.42) | 0.0 | 1288.85 | None | 0.78 | - | - | same db0feac5 |
| bench-ref-repro | 4096 | prose | 2 | 41.45 (41.43-41.47) | 0.1 | 1302.95 | None | 0.42 | - | - | same c75b0986 |
| bench-ref-repro | 16384 | code | 2 | 73.48 (73.45-73.52) | 0.1 | 1592.1 | None | 0.86 | - | - | same 1d7076cc |
| bench-ref-repro | 16384 | prose | 2 | 43.95 (43.91-43.99) | 0.2 | 1590.05 | None | 0.48 | - | - | same 48a7a15d |
| bench-v1 | 4096 | code | 3 | 88.82 (87.93-89.08) | 1.3 | 1162.7 | 0.96 | 0.78 | - | - | same 0855b916 |
| bench-v1 | 4096 | prose | 3 | 61.38 (61.35-61.49) | 0.2 | 1122.7 | 0.95 | 0.47 | - | - | same 14d9f0f3 |
| bench-v1 | 16384 | code | 3 | 93.72 (93.43-93.79) | 0.4 | 1532.1 | 0.96 | 0.83 | - | - | same aec0fddb |
| bench-v1 | 16384 | prose | 3 | 65.15 (64.99-65.18) | 0.3 | 1507.3 | 0.97 | 0.49 | - | - | same 8689b79e |
| bench-v1-repro | 4096 | code | 2 | 73.09 (73.06-73.13) | 0.1 | 1329.25 | None | 0.79 | - | - | DIFFER 9c8886df,db0feac5 |
| bench-v1-repro | 4096 | prose | 2 | 43.19 (42.72-43.66) | 2.2 | 1344.1 | None | 0.42 | - | - | same c75b0986 |
| bench-v1-repro | 16384 | code | 2 | 78.53 (77.72-79.33) | 2.1 | 1616.65 | None | 0.86 | - | - | same 1d7076cc |
| bench-v1-repro | 16384 | prose | 2 | 45.99 (45.55-46.43) | 1.9 | 1616.65 | None | 0.49 | - | - | DIFFER 48a7a15d,eb9bb5da |
| bench-v1x | 4096 | code | 3 | 88.81 (88.67-89.0) | 0.4 | 1166.2 | 0.96 | 0.78 | - | - | same 0855b916 |
| bench-v1x | 4096 | prose | 3 | 61.31 (61.2-61.53) | 0.5 | 1137.7 | 0.95 | 0.47 | - | - | same 14d9f0f3 |
| bench-v1x | 16384 | code | 3 | 93.69 (93.69-93.75) | 0.1 | 1535.6 | 0.96 | 0.83 | - | - | same aec0fddb |
| bench-v1x | 16384 | prose | 3 | 65.17 (65.11-65.18) | 0.1 | 1511.3 | 0.97 | 0.49 | - | - | same 8689b79e |
| bench-v1x-repro | 4096 | code | 2 | 72.98 (72.94-73.03) | 0.1 | 1329.85 | None | 0.78 | - | - | same db0feac5 |
| bench-v1x-repro | 4096 | prose | 2 | 43.47 (43.34-43.6) | 0.6 | 1345.15 | None | 0.42 | - | - | same c75b0986 |
| bench-v1x-repro | 16384 | code | 2 | 78.69 (78.63-78.74) | 0.1 | 1617.9 | None | 0.86 | - | - | same 1d7076cc |
| bench-v1x-repro | 16384 | prose | 2 | 46.35 (46.2-46.5) | 0.6 | 1622.5 | None | 0.48 | - | - | same 48a7a15d |
| serve-pc0 | 4096 | code | 3 | 58.23 (57.92-58.98) | 1.8 | 1010.3 | 0.8 | 0.83 | - | - | same 15c04957 |
| serve-pc0 | 4096 | prose | 3 | 42.55 (42.45-43.01) | 1.3 | 982.7 | 0.81 | 0.48 | - | - | same c462f799 |
| serve-pc0 | 16384 | code | 3 | 51.95 (51.63-52.68) | 2.0 | 1460.2 | 0.74 | 0.8 | - | - | same 27342b1d |
| serve-pc0 | 16384 | prose | 3 | 44.63 (44.54-44.91) | 0.8 | 1437.6 | 0.85 | 0.47 | - | - | same 8d6dfadc |
| serve-pf32k | 4096 | code | 3 | 56.18 (56.01-56.81) | 1.4 | 1006.0 | 0.8 | 0.83 | - | - | same 15c04957 |
| serve-pf32k | 4096 | prose | 3 | 41.64 (41.53-41.93) | 1.0 | 972.9 | 0.81 | 0.48 | - | - | same c462f799 |
| serve-pf32k | 16384 | code | 3 | 52.1 (52.08-52.94) | 1.7 | 1527.0 | 0.74 | 0.83 | - | - | same 3ad97074 |
| serve-pf32k | 16384 | prose | 3 | 45.56 (45.53-45.78) | 0.5 | 1499.6 | 0.86 | 0.47 | - | - | same 51627589 |
| serve-pf32k | 131072 | code | 3 | 51.78 (51.72-52.47) | 1.4 | 1436.8 | 0.75 | 0.84 | - | - | same e0d24dfd |
| serve-ref | 4096 | code | 3 | 57.12 (55.0-57.33) | 4.1 | 1003.8 | 0.8 | 0.83 | - | - | same 15c04957 |
| serve-ref | 4096 | prose | 3 | 41.99 (40.84-42.2) | 3.2 | 970.9 | 0.81 | 0.48 | - | - | same c462f799 |
| serve-ref | 16384 | code | 3 | 51.54 (49.47-51.84) | 4.6 | 1460.4 | 0.74 | 0.8 | - | - | same 27342b1d |
| serve-ref | 16384 | prose | 3 | 44.39 (43.51-44.59) | 2.4 | 1438.9 | 0.85 | 0.47 | - | - | same 8d6dfadc |
| serve-ref | 131072 | code | 3 | 52.2 (50.35-52.79) | 4.7 | 1322.1 | 0.75 | 0.82 | - | - | same c0c470f0 |
| serve-v1 | 4096 | code | 3 | 59.84 (55.99-60.09) | 6.9 | 1019.7 | 0.8 | 0.83 | - | - | same 15c04957 |
| serve-v1 | 4096 | prose | 3 | 44.0 (41.99-44.11) | 4.8 | 977.2 | 0.81 | 0.48 | - | - | same c462f799 |
| serve-v1 | 16384 | code | 3 | 53.92 (50.53-54.25) | 6.9 | 1493.9 | 0.74 | 0.8 | - | - | same 27342b1d |
| serve-v1 | 16384 | prose | 3 | 47.27 (46.91-48.12) | 2.6 | 1468.7 | 0.85 | 0.47 | - | - | DIFFER 238a9bbf,8d6dfadc |
| serve-v2 | 4096 | code | 3 | 68.77 (68.42-69.02) | 0.9 | 1114.1 | 0.91 | 0.79 | - | - | same 7137c924 |
| serve-v2 | 4096 | prose | 3 | 50.6 (50.34-51.15) | 1.6 | 1097.2 | 0.91 | 0.49 | - | - | same 6309e3e9 |
| serve-v2 | 16384 | code | 3 | 70.19 (69.92-70.34) | 0.6 | 1477.3 | 0.9 | 0.84 | - | - | same 143cae4e |
| serve-v2 | 16384 | prose | 3 | 54.83 (54.67-55.0) | 0.6 | 1453.3 | 0.94 | 0.48 | - | - | same 14246264 |
| serve-v2 | 131072 | code | 3 | 69.17 (68.99-69.24) | 0.4 | 1315.2 | 0.91 | 0.83 | - | - | same 6abec92e |
| serve-v3 | 4096 | code | 3 | 80.6 (80.6-80.99) | 0.5 | 1115.7 | 0.93 | 0.82 | - | - | same 844182a1 |
| serve-v3 | 4096 | prose | 3 | 55.45 (55.33-55.68) | 0.6 | 1091.9 | 0.93 | 0.47 | - | - | same f4130dd9 |
| serve-v3 | 16384 | code | 3 | 80.93 (80.83-81.33) | 0.6 | 1506.3 | 0.93 | 0.85 | - | - | same 09df732a |
| serve-v3 | 16384 | prose | 3 | 59.02 (58.86-59.09) | 0.4 | 1478.5 | 0.96 | 0.51 | - | - | same 54856440 |
| serve-v3 | 131072 | code | 3 | 74.9 (74.72-75.17) | 0.6 | 1357.0 | 0.92 | 0.81 | - | - | same 6ea43c2a |
| serve-v4 | 4096 | code | 3 | 58.07 (55.99-58.94) | 5.1 | 1003.7 | 0.79 | 0.8 | - | - | same 2aad366f |
| serve-v4 | 4096 | prose | 3 | 42.87 (41.78-43.34) | 3.6 | 972.8 | 0.8 | 0.46 | - | - | same a4f8d78b |
| serve-v4 | 16384 | code | 3 | 53.83 (52.09-54.71) | 4.9 | 1464.5 | 0.72 | 0.84 | - | - | same 143cae4e |
| serve-v4 | 16384 | prose | 3 | 47.42 (46.48-47.84) | 2.9 | 1443.8 | 0.85 | 0.49 | - | - | same a583dd31 |

Against `bench-ref` (median decode, and the larger of the two arms' ranges as the MDE):

| arm | ctx | task | delta % | MDE % | reading | prompt delta % |
|---|---:|---|---:|---:|---|---:|
| bench-nsys | 4096 | code | -9.0 | 0.4 | slower | -2.4 |
| bench-ref-repro | 16384 | code | -13.4 | 0.2 | slower | 6.3 |
| bench-ref-repro | 16384 | prose | -26.0 | 0.4 | slower | 8.1 |
| bench-ref-repro | 4096 | code | -15.5 | 0.4 | slower | 14.3 |
| bench-ref-repro | 4096 | prose | -26.6 | 0.2 | slower | 19.6 |
| bench-v1-repro | 16384 | code | -7.4 | 2.1 | slower | 7.9 |
| bench-v1-repro | 16384 | prose | -22.5 | 1.9 | slower | 9.9 |
| bench-v1-repro | 4096 | code | -9.7 | 0.4 | slower | 17.8 |
| bench-v1-repro | 4096 | prose | -23.6 | 2.2 | slower | 23.4 |
| bench-v1x-repro | 16384 | code | -7.2 | 0.2 | slower | 8.0 |
| bench-v1x-repro | 16384 | prose | -21.9 | 0.6 | slower | 10.3 |
| bench-v1x-repro | 4096 | code | -9.9 | 0.4 | slower | 17.9 |
| bench-v1x-repro | 4096 | prose | -23.1 | 0.6 | slower | 23.5 |
| bench-v1x | 16384 | code | +10.4 | 0.2 | faster | 2.5 |
| bench-v1x | 16384 | prose | +9.8 | 0.4 | faster | 2.7 |
| bench-v1x | 4096 | code | +9.7 | 0.4 | faster | 3.4 |
| bench-v1x | 4096 | prose | +8.5 | 0.5 | faster | 4.5 |
| bench-v1 | 16384 | code | +10.5 | 0.4 | faster | 2.3 |
| bench-v1 | 16384 | prose | +9.8 | 0.4 | faster | 2.4 |
| bench-v1 | 4096 | code | +9.7 | 1.3 | faster | 3.1 |
| bench-v1 | 4096 | prose | +8.6 | 0.2 | faster | 3.1 |
| serve-pc0 | 16384 | code | -38.8 | 2.0 | slower | -2.5 |
| serve-pc0 | 16384 | prose | -24.8 | 0.8 | slower | -2.3 |
| serve-pc0 | 4096 | code | -28.1 | 1.8 | slower | -10.4 |
| serve-pc0 | 4096 | prose | -24.7 | 1.3 | slower | -9.8 |
| serve-pf32k | 16384 | code | -38.6 | 1.7 | slower | 1.9 |
| serve-pf32k | 16384 | prose | -23.2 | 0.5 | slower | 1.9 |
| serve-pf32k | 4096 | code | -30.6 | 1.4 | slower | -10.8 |
| serve-pf32k | 4096 | prose | -26.3 | 1.0 | slower | -10.7 |
| serve-ref | 16384 | code | -39.3 | 4.6 | slower | -2.5 |
| serve-ref | 16384 | prose | -25.2 | 2.4 | slower | -2.2 |
| serve-ref | 4096 | code | -29.4 | 4.1 | slower | -11.0 |
| serve-ref | 4096 | prose | -25.7 | 3.2 | slower | -10.9 |
| serve-v1 | 16384 | code | -36.4 | 6.9 | slower | -0.3 |
| serve-v1 | 16384 | prose | -20.4 | 2.6 | slower | -0.2 |
| serve-v1 | 4096 | code | -26.1 | 6.9 | slower | -9.6 |
| serve-v1 | 4096 | prose | -22.1 | 4.8 | slower | -10.3 |
| serve-v2 | 16384 | code | -17.3 | 0.6 | slower | -1.4 |
| serve-v2 | 16384 | prose | -7.6 | 0.6 | slower | -1.2 |
| serve-v2 | 4096 | code | -15.1 | 0.9 | slower | -1.2 |
| serve-v2 | 4096 | prose | -10.4 | 1.6 | slower | 0.7 |
| serve-v3 | 16384 | code | -4.6 | 0.6 | slower | 0.5 |
| serve-v3 | 16384 | prose | -0.6 | 0.4 | slower | 0.5 |
| serve-v3 | 4096 | code | -0.4 | 0.5 | within noise | -1.1 |
| serve-v3 | 4096 | prose | -1.9 | 0.6 | slower | 0.3 |
| serve-v4 | 16384 | code | -36.6 | 4.9 | slower | -2.3 |
| serve-v4 | 16384 | prose | -20.1 | 2.9 | slower | -1.9 |
| serve-v4 | 4096 | code | -28.3 | 5.1 | slower | -11.0 |
| serve-v4 | 4096 | prose | -24.1 | 3.6 | slower | -10.7 |
