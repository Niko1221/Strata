### Speed (Strata 0.2.1, 262,144-token context, one request at a time)

| Model | GGUF size (GiB) | Cold start (s) | Experts in GPU cache | Decode tok/s 8K / 32K / 128K | Prompt tok/s 8K / 32K / 128K | strata_bench short chat / agent decode | strata_bench 4K / 16K prompt |
|---|---:|---:|---|---|---|---|---|
| UD-Q2_K_XL | 73.5 | 38 | 19415 (33.91 GiB) | 130 / 126 / 118 | 1428 / 3434 / 3417 | 136.9 / 149.8 | 2736.4 / 3464.6 |
| UD-Q3_K_XL | 83.8 | 43 | 15570 (32.96 GiB) | 111 / 114 / 110 | 1471 / 3485 / 3416 | 125.5 / 125.5 | 2709.5 / 3481.6 |
| UD-Q4_K_XL | 103.7 | 33 | 11204 (32.71 GiB) | 103 / 99 / 92 | 1324 / 2825 / 2734 | 96.8 / 106.6 | 1728.2 / 2894.9 |
| UD-Q5_K_XL | 147.4 | 68 | 8758 (32.66 GiB) | 94 / 87 / 86 | 1208 / 2308 / 2179 | 87.6 / 96.8 | 1258.2 / 2297.3 |
| UD-Q6_K_XL | 157.5 | 69 | 7878 (32.64 GiB) | 83 / 86 / 78 | 1173 / 2043 / 1926 | 81.9 / 85.4 | 1093.9 / 2025.8 |
| Q8_0 | 175.3 | 43 | 6627 (32.23 GiB) | 74 / 74 / 69 | 1017 / 1708 / 1615 | 69.8 / 81.3 | 900.6 / 1687.2 |

### Quality against Q8_0 on the same engine (first-token logits)

| Model | KL mean, 60 prompts | KL median / p90 | same top-1 | KL mean, 160 prompts by domain | same top-1 | needles at 128K |
|---|---:|---|---|---:|---|---|
| UD-Q2_K_XL | 0.221 | 0.173 / 0.365 | 51/60 | 0.406 | 118/160 | 3/3 |
| UD-Q3_K_XL | 0.116 | 0.068 / 0.291 | 54/60 | 0.144 | 135/160 | 3/3 |
| UD-Q4_K_XL | 0.031 | 0.014 / 0.080 | 54/60 | 0.057 | 143/160 | 3/3 |
| UD-Q5_K_XL | 0.016 | 0.009 / 0.049 | 57/60 | 0.029 | 150/160 | 3/3 |
| UD-Q6_K_XL | 0.010 | 0.006 / 0.026 | 59/60 | 0.020 | 151/160 | 3/3 |
| Q8_0 | - | - / - | - | - | - | 3/3 |

### Task accuracy (lm-evaluation-harness 0.4.13 and EvalPlus over the HTTP API, greedy)

| Model | HumanEval+ base / plus (164) | GSM8K strict, 4096 tokens (100) |  IFEval prompt / instruction strict (100) | MATH geometry, math_verify (100) | BBH geometric_shapes (250) | BBH navigate (250) |
|---|---|---|---|---|---|---|
| UD-Q2_K_XL | 93.3 / 89.6 | 97 | 83 / 87.7 | 96 | - | - |
| UD-Q3_K_XL | 91.5 / 87.8 | 98 | 83 / 87.7 | 93 | - | - |
| UD-Q4_K_XL | 92.7 / 89.0 | 99 | 82 / 87.1 | 96 | - | - |
| UD-Q5_K_XL | 93.9 / 90.9 | 99 | 84 / 89.0 | 96 | - | - |
| UD-Q6_K_XL | 93.9 / 90.9 | 99 | 83 / 88.3 | 97 | - | - |
| Q8_0 | 93.3 / 90.9 | 99 | 82 / 87.7 | 95 | - | - |

### Image input (synthetic images, vision encoder on the GPU)

| Model | texto9px | texto12px | texto16px | texto24px | contar | tabela | cor-posicao |
|---|---|---|---|---|---|---|---|
| UD-Q4_K_XL | 0/2 | 1/2 | 2/2 | 2/2 | 2/6 | 6/6 | 4/4 |
| Q8_0 | 0/2 | 1/2 | 2/2 | 2/2 | 2/6 | 6/6 | 4/4 |

### 3D tasks in Blender 5.2.1 (26 tasks, one attempt, temperature 0)

| Model | passed | medidas | relacoes | camera_luz | material_cena | hierarquia_mod | animacao | failed tasks | median s per task |
|---|---|---|---|---|---|---|---|---|---:|
| UD-Q2_K_XL | **24/26** | 5/5 | 8/8 | 2/2 | 5/5 | 4/5 | 0/1 | espelho, animacao | 30.2 |
| UD-Q3_K_XL | **25/26** | 5/5 | 8/8 | 2/2 | 4/5 | 5/5 | 1/1 | colecao | 27.2 |
| UD-Q4_K_XL | **25/26** | 5/5 | 8/8 | 2/2 | 5/5 | 4/5 | 1/1 | espelho | 42.6 |
| UD-Q5_K_XL | **26/26** | 5/5 | 8/8 | 2/2 | 5/5 | 5/5 | 1/1 | - | 30.9 |
| UD-Q6_K_XL | **26/26** | 5/5 | 8/8 | 2/2 | 5/5 | 5/5 | 1/1 | - | 35.4 |
| Q8_0 | **25/26** | 5/5 | 8/8 | 2/2 | 4/5 | 5/5 | 1/1 | texto | 45.0 |

### Same model and config, 0.2.0 -> 0.2.1 (single runs)

| Model | Decode 8K / 32K / 128K | Prompt 32K / 128K | strata_bench chat / agent decode | HumanEval+ plus | GSM8K | IFEval prompt | MATH geometry |
|---|---|---|---|---|---|---|---|
| UD-Q2_K_XL | 126 / 122 / 111 → 130 / 126 / 118 | 3194 / 3425 → 3434 / 3417 | 138.9 / 136.5 → 136.9 / 149.8 | 89.0 → 89.6 | 98 → 97 | 88 → 83 | 95 → 96 |
| UD-Q3_K_XL | 107 / 108 / 105 → 111 / 114 / 110 | 3484 / 3417 → 3485 / 3416 | 116.5 / 128.1 → 125.5 / 125.5 | 87.2 → 87.8 | 99 → 98 | 83 → 83 | 96 → 93 |
| UD-Q4_K_XL | 98 / 99 / 96 → 103 / 99 / 92 | 2828 / 2739 → 2825 / 2734 | 104.4 / 113.0 → 96.8 / 106.6 | 91.5 → 89.0 | 99 → 99 | 84 → 82 | 95 → 96 |
| UD-Q5_K_XL | 89 / 90 / 83 → 94 / 87 / 86 | 2288 / 2167 → 2308 / 2179 | 86.9 / 98.4 → 87.6 / 96.8 | 89.6 → 90.9 | 98 → 99 | 82 → 84 | 96 → 96 |
| UD-Q6_K_XL | 82 / 82 / 78 → 83 / 86 / 78 | 2037 / 1928 → 2043 / 1926 | 77.9 / 90.7 → 81.9 / 85.4 | 89.0 → 90.9 | 99 → 99 | 86 → 83 | 96 → 97 |
| Q8_0 | 75 / 73 / 72 → 74 / 74 / 69 | 1703 / 1613 → 1708 / 1615 | 69.9 / 76.3 → 69.8 / 81.3 | 89.0 → 90.9 | 99 → 99 | 86 → 82 | 97 → 95 |

### Interleaved starts, UD-Q4_K_XL: 0.2.0, 0.2.1, 0.2.0, 0.2.1, ... (3 pairs, warm file cache)

| Version | Decode 8K, runs 1 / 2 / 3 | Decode 32K, runs 1 / 2 / 3 | Prompt 8K / 32K, mean |
|---|---|---|---|
| 0.2.0 | 91.3 / 90.7 / 87.6 (mean 89.9) | 102.1 / 99.3 / 106.1 (mean 102.5) | 1678 / 2797 |
| 0.2.1 | 93.9 / 97.8 / 90.9 (mean 94.2) | 101.0 / 103.0 / 100.9 (mean 101.6) | 1825 / 2817 |

### First-token KL across versions

| Comparison | KL, 60 prompts | same top-1 | KL, 160 prompts | same top-1 |
|---|---:|---|---:|---|
| Q8_0: 0.2.1 against 0.2.0 | 0.012 | 56/60 | 0.018 | 153/160 |
| UD-Q4_K_XL 0.2.1 against Q8_0 0.2.0 | 0.034 | 56/60 | - | - |
| UD-Q2_K_XL 0.2.1 against Q8_0 0.2.0 | 0.217 | 52/60 | - | - |
