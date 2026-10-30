| arm | ctx | task | n | decode tok/s median (min-max) | range % | prompt tok/s median | VRAM share | drafts acc. | win ms | CPU-expert ms | hashes |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| serve-ref | 4096 | code | 2 | 56.76 (55.77-57.74) | 3.5 | 1004.45 | 0.8 | 0.83 | - | - | same 15c04957 |
| serve-ref | 4096 | prose | 2 | 41.78 (41.05-42.52) | 3.5 | 975.9 | 0.81 | 0.48 | - | - | same c462f799 |
| serve-ref | 16384 | code | 2 | 51.2 (50.35-52.04) | 3.3 | 1466.85 | 0.74 | 0.8 | - | - | same 27342b1d |
| serve-ref | 16384 | prose | 2 | 44.17 (43.58-44.76) | 2.7 | 1438.95 | 0.85 | 0.47 | - | - | same 8d6dfadc |
| serve-ref | 131072 | code | 2 | 51.75 (50.98-52.52) | 3.0 | 1323.25 | 0.75 | 0.82 | - | - | same c0c470f0 |
| serve-s1 | 4096 | code | 2 | 88.19 (87.96-88.41) | 0.5 | 1157.9 | 0.95 | 0.78 | - | - | same d1db0030 |
| serve-s1 | 4096 | prose | 2 | 62.71 (61.91-63.51) | 2.6 | 1133.2 | 0.95 | 0.5 | - | - | DIFFER 392186b6,4973364a |
| serve-s1 | 16384 | code | 2 | 93.11 (92.64-93.58) | 1.0 | 1522.95 | 0.96 | 0.84 | - | - | DIFFER 143cae4e,6e3d3cdf |
| serve-s1 | 16384 | prose | 2 | 66.13 (65.42-66.85) | 2.2 | 1503.75 | 0.97 | 0.51 | - | - | DIFFER 31642cd0,a890be50 |
| serve-s1 | 131072 | code | 2 | 86.61 (86.43-86.78) | 0.4 | 1360.7 | 0.96 | 0.8 | - | - | DIFFER 14069414,86f913d6 |
| serve-s2 | 4096 | code | 2 | 76.55 (76.13-76.97) | 1.1 | 1150.65 | 0.91 | 0.84 | - | - | same dd272b27 |
| serve-s2 | 4096 | prose | 2 | 52.32 (51.91-52.73) | 1.6 | 1119.35 | 0.91 | 0.46 | - | - | same e3412343 |
| serve-s2 | 16384 | code | 2 | 74.87 (74.45-75.29) | 1.1 | 1504.9 | 0.89 | 0.85 | - | - | DIFFER 79b7dd6c,c5b7c06d |
| serve-s2 | 16384 | prose | 2 | 60.0 (59.98-60.02) | 0.1 | 1484.6 | 0.94 | 0.49 | - | - | DIFFER 60e4d5a1,c9cf29b3 |
| serve-s2 | 131072 | code | 2 | 73.5 (73.35-73.64) | 0.4 | 1335.6 | 0.9 | 0.82 | - | - | same 6ea43c2a |

Against `serve-ref` (median decode, and the larger of the two arms' ranges as the MDE):

| arm | ctx | task | delta % | MDE % | reading | prompt delta % |
|---|---:|---|---:|---:|---|---:|
| serve-s1 | 131072 | code | +67.4 | 3.0 | faster | 2.8 |
| serve-s1 | 16384 | code | +81.9 | 3.3 | faster | 3.8 |
| serve-s1 | 16384 | prose | +49.7 | 2.7 | faster | 4.5 |
| serve-s1 | 4096 | code | +55.4 | 3.5 | faster | 15.3 |
| serve-s1 | 4096 | prose | +50.1 | 3.5 | faster | 16.1 |
| serve-s2 | 131072 | code | +42.0 | 3.0 | faster | 0.9 |
| serve-s2 | 16384 | code | +46.2 | 3.3 | faster | 2.6 |
| serve-s2 | 16384 | prose | +35.8 | 2.7 | faster | 3.2 |
| serve-s2 | 4096 | code | +34.9 | 3.5 | faster | 14.6 |
| serve-s2 | 4096 | prose | +25.2 | 3.5 | faster | 14.7 |
