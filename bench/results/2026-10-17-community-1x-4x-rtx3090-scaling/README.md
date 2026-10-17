# 1-4x RTX 3090 scaling — Qwen3.8-Flash-Next UD-IQ4_XS

Controlled scaling run on one host. Each row used a fresh Strata process with the same model and settings; only the GPU list changed. Values are medians from `tools/strata_bench.py --runs 3 --long`.

## Results

| GPUs | Chat decode tok/s | Chat TTFT s | Agent decode tok/s | 4K prompt tok/s | 16K prompt tok/s | 32K prompt tok/s | 32K TTFT s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 62.9 | 0.40 | 66.9 | 1,427.5 | 1,927.8 | 1,906.8 | 16.86 |
| 2 | 92.4 | 0.37 | 95.7 | 1,645.2 | 2,451.2 | 2,885.3 | 11.15 |
| 3 | 107.3 | 0.34 | 102.7 | 1,772.6 | 2,752.9 | 3,583.8 | 8.99 |
| 4 | 118.6 | 0.23 | 109.8 | 1,828.8 | 2,887.2 | 3,982.2 | 8.11 |

From 1 to 4 GPUs, median chat decode improved 1.89x and 32K prompt processing improved 2.09x.

## Hardware and software

- 4x NVIDIA GeForce RTX 3090 24 GiB, each limited to 220 W
- AMD EPYC 7702 (64 cores / 128 threads), 247.3 GiB RAM
- Ubuntu 26.04.1 LTS, Linux 7.0.0-38, NVIDIA 595.91.07, CUDA 13.2
- Strata 0.1.45, commit `477ac13c`; locally built engine 0.1.45 for `sm_86`
- Qwen3.8-Flash-Next Unsloth Dynamic UD-IQ4_XS, 262,144-token context
- INT8 KV, 32,768 resident KV tokens, 55 GiB resident expert budget, automatic prefill and layer split, MTP with `--spec 4 --spec-min-p 0.5`

GPU order was `[2]`, `[2,3]`, `[2,3,1]`, then `[2,3,1,0]`.

## Method and artifacts

The built-in long suite ran three repetitions per configuration. Model load time was excluded. No other inference workload ran during the matrix.

- Raw per-request data: [`results/`](results/)
- Exact launch configurations: [`configs/`](configs/)
- Matrix runner: [`run-matrix.sh`](run-matrix.sh)

## Limitations

Speed suite only; no correctness or long-context recall test was run. Results cover one host, one model quantization, and one 220 W power limit.
