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

### Extended 128K prefill

| GPUs | Actual prompt tokens | Prompt tok/s median (range) | TTFT s median (range) | Scaling vs. 1 GPU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 127,975 | 1,820.0 (1,819.1–1,824.2) | 70.460 (70.293–70.484) | 1.00× |
| 2 | 127,975 | 3,481.0 (3,478.4–3,491.8) | 36.901 (36.791–36.930) | 1.91× |
| 3 | 127,975 | 4,837.3 (4,821.9–4,855.2) | 26.597 (26.495–26.676) | 2.66× |
| 4 | 127,975 | 5,925.9 (5,923.9–5,943.5) | 21.734 (21.686–21.735) | 3.26× |

From 1 to 4 GPUs, median chat decode improved 1.89×, 32K prompt processing improved 2.09×, and 128K prompt processing improved 3.26×.

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

The built-in long suite ran three repetitions per configuration. The separate 128K extension reused Strata's prompt generator, calibration, timing parser, and unique fresh-prompt tags for three repetitions per configuration with a 24-token output cap. Model load and warm-up time were excluded, every measured 128K request reported zero reused tokens, and no other inference workload ran during either matrix.

- Raw per-request data: [`results/`](results/)
- Exact launch configurations: [`configs/`](configs/)
- Stock matrix runner: [`run-matrix.sh`](run-matrix.sh)
- 128K companion and matrix runner: [`bench-128k.py`](bench-128k.py), [`run-128k-matrix.sh`](run-128k-matrix.sh)

## Limitations

Speed suite only; no correctness or long-context recall test was run. The 128K extension measures ingestion speed, not answer quality. Results cover one host, one model quantization, and one 220 W power limit.
