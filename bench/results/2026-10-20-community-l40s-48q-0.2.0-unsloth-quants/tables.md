### Speed (Strata 0.2.0, 262,144-token context, one request at a time)

| Model | GGUF size (GiB) | Cold start (s) | Experts in GPU cache | Decode tok/s 8K / 32K / 128K | Prompt tok/s 8K / 32K / 128K | strata_bench short chat / agent decode | strata_bench 4K / 16K prompt |
|---|---:|---:|---|---|---|---|---|
| UD-Q2_K_XL | 73.5 | 39 | 19414 (33.91 GiB) | 126 / 122 / 111 | 1340 / 3194 / 3425 | 138.9 / 136.5 | 2730.7 / 3471.9 |
| UD-Q3_K_XL | 83.8 | 43 | 15568 (32.95 GiB) | 107 / 108 / 105 | 1435 / 3484 / 3417 | 116.5 / 128.1 | 2692.3 / 3464.4 |
| UD-Q4_K_XL | 103.7 | 118 | 11203 (32.71 GiB) | 98 / 99 / 96 | 1345 / 2828 / 2739 | 104.4 / 113.0 | 1723.2 / 2908.1 |
| UD-Q5_K_XL | 147.4 | 159 | 8758 (32.66 GiB) | 89 / 90 / 83 | 1166 / 2288 / 2167 | 86.9 / 98.4 | 1249.2 / 2288.9 |
| UD-Q6_K_XL | 157.5 | 148 | 7878 (32.64 GiB) | 82 / 82 / 78 | 1091 / 2037 / 1928 | 77.9 / 90.7 | 1086.2 / 2020.0 |
| Q8_0 | 175.3 | 198 | 6627 (32.23 GiB) | 75 / 73 / 72 | 977 / 1703 / 1613 | 69.9 / 76.3 | 898.1 / 1676.6 |
| Q8_0 on 0.1.45 (reference) | - | 78 | 6629 (32.24 GiB) | 74 / 70 / 70 | 972 / 1712 / 1616 | 66.5 / 75.5 | 902.2 / 1682.0 |
| UD-Q4_K_XL, tail skip at the engine default | - | 53 | 11203 (32.71 GiB) | 97 / 103 / 96 | 1275 / 2778 / 2726 | 104.2 / 111.8 | 1729.0 / 3052.5 |

### Quality against Q8_0 on the same engine (first-token logits)

| Model | KL mean, 60 prompts | KL median / p90 | same top-1 | KL mean, 160 prompts by domain | same top-1 | needles at 128K |
|---|---:|---|---|---:|---|---|
| UD-Q2_K_XL | 0.202 | 0.146 / 0.429 | 53/60 | 0.405 | 115/160 | 3/3 |
| UD-Q3_K_XL | 0.108 | 0.063 / 0.311 | 54/60 | 0.158 | 133/160 | 3/3 |
| UD-Q4_K_XL | 0.037 | 0.012 / 0.096 | 57/60 | 0.065 | 143/160 | 3/3 |
| UD-Q5_K_XL | 0.023 | 0.008 / 0.070 | 55/60 | 0.034 | 152/160 | 3/3 |
| UD-Q6_K_XL | 0.015 | 0.006 / 0.033 | 56/60 | 0.037 | 148/160 | 3/3 |
| Q8_0 | - | - / - | - | - | - | 3/3 |
| Q8_0 on 0.1.45 (reference) | - | - / - | - | - | - | - |
| UD-Q4_K_XL, tail skip at the engine default | - | - / - | - | - | - | - |

### Task accuracy (lm-evaluation-harness 0.4.13 and EvalPlus over the HTTP API, greedy)

| Model | HumanEval+ base / plus (164) | GSM8K strict, 4096 tokens (100) |  IFEval prompt / instruction strict (100) | MATH geometry, math_verify (100) | BBH geometric_shapes (250) | BBH navigate (250) |
|---|---|---|---|---|---|---|
| UD-Q2_K_XL | 93.3 / 89.0 | 98 | 88 / 91.4 | 95 | 80.0 | 99.6 |
| UD-Q3_K_XL | 89.6 / 87.2 | 99 | 83 / 89.0 | 96 | 80.0 | 100.0 |
| UD-Q4_K_XL | 93.9 / 91.5 | 99 | 84 / 89.0 | 95 | 80.0 | 100.0 |
| UD-Q5_K_XL | 92.7 / 89.6 | 98 | 82 / 87.7 | 96 | 80.0 | 100.0 |
| UD-Q6_K_XL | 92.1 / 89.0 | 99 | 86 / 90.2 | 96 | 80.0 | 100.0 |
| Q8_0 | 92.7 / 89.0 | 99 | 86 / 90.2 | 97 | 80.0 | 100.0 |
| Q8_0 on 0.1.45 (reference) | 93.3 / 90.2 | 99 | 85 / 89.6 | 95 | 80.0 | 99.2 |
| UD-Q4_K_XL, tail skip at the engine default | 93.9 / 90.2 | 99 | - / - | - | - | - |

### Image input (synthetic images, vision encoder on the GPU)

| Model | texto9px | texto12px | texto16px | texto24px | contar | tabela | cor-posicao |
|---|---|---|---|---|---|---|---|
| UD-Q4_K_XL | 0/2 | 1/2 | 2/2 | 2/2 | 2/6 | 6/6 | 4/4 |
| Q8_0 | 0/2 | 1/2 | 2/2 | 2/2 | 2/6 | 6/6 | 4/4 |
| Q8_0 on 0.1.45 (reference) | 0/2 | 1/2 | 2/2 | 2/2 | 3/6 | 6/6 | 4/4 |
