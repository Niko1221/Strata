# 1–4× RTX 3090 scaling on EPYC 7702 — UD-IQ4_XS at 262K

Controlled scaling run on one bare-metal host. Each row used a fresh Strata process with the same model and settings; only the GPU list changed. Values are medians from `tools/strata_bench.py --runs 3 --long`; raw JSON contains the full per-request data and ranges.

## Results

### Decode

| GPUs | Chat tok/s | Chat TTFT s | Agent tok/s | Agent TTFT s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 62.9 | 0.401 | 66.9 | 2.587 |
| 2 | 92.4 | 0.365 | 95.7 | 2.171 |
| 3 | 107.3 | 0.339 | 102.7 | 2.031 |
| 4 | 118.6 | 0.230 | 109.8 | 2.030 |

### Prefill

| GPUs | 4K tok/s | 4K TTFT s | 16K tok/s | 16K TTFT s | 32K tok/s | 32K TTFT s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,427.5 | 2.837 | 1,927.8 | 8.363 | 1,906.8 | 16.856 |
| 2 | 1,645.2 | 2.470 | 2,451.2 | 6.585 | 2,885.3 | 11.155 |
| 3 | 1,772.6 | 2.289 | 2,752.9 | 5.866 | 3,583.8 | 8.990 |
| 4 | 1,828.8 | 2.219 | 2,887.2 | 5.594 | 3,982.2 | 8.113 |

From 1 to 4 GPUs, median chat decode improved 1.89x and 32K prompt processing improved 2.09x.

## Hardware and software

- Bare-metal ASRock Rack ROMED8-2T host
- 4× ASUS TUF Gaming GeForce RTX 3090 OC 24 GiB (`TUF-RTX3090-O24G-GAMING`), each on a PCIe 4.0 ×16 link and limited to 220 W
- AMD EPYC 7702, 1 socket, 64 cores / 128 threads, 1.50–3.35 GHz, AVX2
- 256 GB installed / 247.3 GiB usable: 8× 32 GB Micron dual-rank DDR4-2666 ECC RDIMMs, all eight memory channels populated
- Model files on a local WD_BLACK SN7100 1 TB NVMe; storage throughput was not separately measured
- 8 GiB swap configured, 0 used at final verification; 55 GiB resident expert budget
- Ubuntu 26.04.1 LTS, Linux 7.0.0-38, NVIDIA 595.91.07, CUDA 13.2
- Strata 0.1.45, commit `477ac13c`; locally built engine 0.1.45 for `sm_86`
- Qwen3.8-Flash-Next Unsloth Dynamic UD-IQ4_XS, 262,144-token context
- INT8 KV, 32,768 resident KV tokens, automatic prefill and layer split, MTP with `--spec 4 --spec-min-p 0.5`

GPU order was `[2]`, `[2,3]`, `[2,3,1]`, then `[2,3,1,0]`.

## Method and artifacts

The built-in long suite ran three repetitions per configuration. Model load time was excluded. No other inference workload ran during the matrix.

- Raw per-request data: [`results/`](results/)
- Exact launch configurations: [`configs/`](configs/)
- Matrix runner: [`run-matrix.sh`](run-matrix.sh)

## Limitations

Speed suite only; no correctness or long-context recall test was run. Results cover one host, one model quantization, and one 220 W power limit.
