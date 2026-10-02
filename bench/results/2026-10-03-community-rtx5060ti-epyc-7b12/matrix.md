# Matrix: unsloth UD-Q4_K_XL on one RTX 5060 Ti 16 GB, 262,144-token context

See `README.md` for hardware, build, hashes and method. All throughput values are engine-reported; TTFT is derived from the non-streaming prompt-processing time. Model loading is excluded; vision encoding is included in the image cell. `matrix.json` holds every run record.

## Per-run values

### A1-多模态GPU (GPU vision)

| Cell | Run | Prompt tok | Reused tok | Generated tok | Prompt tok/s | Decode tok/s | TTFT s | Latency s | Draft accepted | cache_n |
|---|---|---|---|---|---|---|---|---|---|---|
| short | 1 | 25 | 0 | 18 | 33.4 | 20.7 | 0.748 | 1.7 | 13/17 | 0 |
| short | 2 | 25 | 0 | 18 | 39.0 | 28.4 | 0.642 | 1.3 | 11/17 | 0 |
| short | 3 | 25 | 0 | 18 | 43.2 | 26.3 | 0.579 | 1.3 | 13/17 | 0 |
| long | 1 | 10381 | 0 | 62 | 392.6 | 23.2 | 26.441 | 29.5 | 31/50 | 0 |
| long | 2 | 10381 | 0 | 62 | 618.3 | 27.1 | 16.791 | 19.1 | 30/51 | 0 |
| long | 3 | 10383 | 0 | 61 | 555.9 | 22.3 | 18.678 | 21.5 | 33/51 | 0 |
| long | 4 | 10383 | 0 | 62 | 622.9 | 25.8 | 16.669 | 19.1 | 30/53 | 0 |
| deep | 1 | 144610 | 0 | 53 | 795.3 | 25.8 | 181.833 | 184.7 | 27/33 | 0 |
| deep | 2 | 144618 | 0 | 46 | 795.5 | 23.5 | 181.8 | 184.5 | 21/29 | 0 |
| solve | 1 | 369 | 0 | 1502 | 86.9 | 32.7 | 4.246 | 72.2 | 971/1355 | unavailable |


### A2-多模态CPU-系统内存 (CPU vision)

| Cell | Run | Prompt tok | Reused tok | Generated tok | Prompt tok/s | Decode tok/s | TTFT s | Latency s | Draft accepted | cache_n |
|---|---|---|---|---|---|---|---|---|---|---|
| short | 1 | 25 | 0 | 18 | 38.8 | 22.1 | 0.644 | 1.5 | 10/17 | 0 |
| short | 2 | 25 | 0 | 18 | 39.9 | 30.1 | 0.626 | 1.2 | 11/15 | 0 |
| short | 3 | 25 | 0 | 18 | 43.1 | 26.7 | 0.58 | 1.3 | 13/17 | 0 |
| long | 1 | 10381 | 0 | 62 | 594.6 | 23.6 | 17.458 | 20.1 | 31/51 | 0 |
| long | 2 | 10381 | 0 | 61 | 630.9 | 30.0 | 16.455 | 18.5 | 31/47 | 0 |
| deep | 1 | 144610 | 0 | 58 | 806.6 | 26.9 | 179.279 | 182.1 | 29/33 | 0 |
| solve | 1 | 369 | 0 | 1615 | 88.8 | 34.8 | 4.155 | 51.2 | 1115/1345 | unavailable |


## Engine log counters

### A1-多模态GPU (GPU vision)

| Cell | Run | KV VRAM hit % | KV RAM read MiB | GPU expert hits | RAM blobs | File blobs | File read MB | Resident RAM GiB |
|---|---|---|---|---|---|---|---|---|
| short | 1 | 96.13 | 0.6 | 1436 | 12956 | 0 | 6496.2 | 65.68 |
| short | 2 | 96.28 | 0.5 | 3850 | 23825 | 0 | 6496.2 | 65.68 |
| short | 3 | 96.31 | 0.5 | 4401 | 34201 | 0 | 6496.2 | 65.68 |
| long | 1 | 95.41 | 101.4 | 16973 | 98973 | 1371 | 19291.6 | 65.68 |
| long | 2 | 95.49 | 101.9 | 26064 | 158077 | 2742 | 32163.9 | 65.68 |
| long | 3 | 95.36 | 101.4 | 13522 | 66406 | 1371 | 19250.8 | 65.68 |
| long | 4 | 95.6 | 101.5 | 25417 | 127157 | 2742 | 32123.1 | 65.68 |
| deep | 1 | 87.24 | 212.2 | 13733 | 577431 | 4113 | 113688.5 | 65.68 |
| deep | 2 | 87.7 | 189.3 | 12842 | 545546 | 4113 | 113647.6 | 65.68 |
| solve | 1 | 99.91 | 22.6 | 543750 | 913510 | 4315 | 114794.7 | 65.68 |

### A2-多模态CPU-系统内存 (CPU vision)

| Cell | Run | KV VRAM hit % | KV RAM read MiB | GPU expert hits | RAM blobs | File blobs | File read MB | Resident RAM GiB |
|---|---|---|---|---|---|---|---|---|
| short | 1 | 96.42 | 0.5 | 1887 | 12434 | 0 | 8279.7 | 64.02 |
| short | 2 | 96.07 | 0.5 | 3870 | 22372 | 0 | 8279.7 | 64.02 |
| short | 3 | 96.31 | 0.5 | 4817 | 32327 | 0 | 8279.7 | 64.02 |
| long | 1 | 95.45 | 101.6 | 18540 | 95073 | 1369 | 21077.1 | 64.02 |
| long | 2 | 95.22 | 100.9 | 26348 | 150506 | 2738 | 33955.2 | 64.02 |
| deep | 1 | 87.53 | 216.6 | 16520 | 558621 | 4107 | 115516.7 | 64.02 |
| solve | 1 | 99.91 | 24.0 | 507523 | 907161 | 4309 | 116631.4 | 64.02 |

## Vision encode A/B

| Mode | Image tokens | 1st-call wall s | 1st prompt_n | 1st prefill tok/s | Prompt-prefill part s | Implied encode s (upper bound) | 2nd-call wall s (embedding hit) |
|---|---|---|---|---|---|---|---|
| GPU | 273 | 3.92 | 280 | 74.1 | 3.779 | 0.141 | 0.177 |
| CPU | 273 | 5.398 | 280 | 62.8 | 4.459 | 0.939 | 0.172 |

## Single-variable sweep (ctx=8192, no vision)

| # | Config | ctx | Load s | Decode tok/s | Prompt tok/s | Accept % | Expert slots | VRAM free MiB | Arena MiB | GPU hits | File blobs | File MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 29 | S2-256k-best | 262144 | 74 | 32.85 | 237.05 | 80.5 | 2042 | 328 | 73450 |  |  |  |
| 3 | S0c-arena | 8192 | 60 | 29.45 | 247.3 | 54.7 | 2809 | 314 | 73450 |  |  |  |
| 4 | S0d-arena-shm | 8192 | 88 | 29.45 | 244.65 | 54.7 | 2809 | 314 | 73450 |  |  |  |
| 28 | S0e-arena-shm-2nd | 8192 | 80 | 29.45 | 248.05 | 54.7 | 2809 | 314 | 73450 |  |  |  |
| 15 | S1-spec2 | 8192 | 76 | 26.05 | 245.6 | 73.25 | 2857 | 202 | 64921 | 20935 | 794 | 13563.4 |
| 18 | S1-shortread0 | 8192 | 77 | 25.8 | 236.05 | 62.05 | 2856 | 190 | 64924 | 20505 | 794 | 13751.3 |
| 5 | S1-base | 8192 | 309 | 25.35 | 241.35 | 62.2 | 2856 | 190 | 64924 | 22068 | 794 | 13560.3 |
| 9 | S1-pool112 | 8192 | 76 | 25.25 | 246.6 | 62.2 | 2856 | 190 | 64924 | 22068 | 794 | 13560.3 |
| 22 | S1-affinity-auto | 8192 | 76 | 25.1 | 247.15 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 12 | S1-ple-mmap | 8192 | 93 | 25.05 | 245.9 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 13 | S1-prefill16k | 8192 | 77 | 25.05 | 246.3 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 14 | S1-prefill32k | 8192 | 76 | 25.05 | 247.3 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 19 | S1-shortread128 | 8192 | 77 | 25.05 | 247.1 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 20 | S1-kvres20480 | 8192 | 76 | 25.05 | 247 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 21 | S1-stager96 | 8192 | 76 | 25.05 | 244 | 58.2 | 2856 | 188 | 64924 | 24024 | 794 | 13560.3 |
| 23 | S1-nohostworker | 8192 | 77 | 25.05 | 247.2 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 11 | S1-ple-ram | 8192 | 181 | 25 | 245.15 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 8 | S1-pool96 | 8192 | 77 | 24.95 | 247.5 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 7 | S1-pool48 | 8192 | 81 | 24.7 | 242.25 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 1 | S0a-budget71 | 8192 | 84 | 24.7 | 241.9 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 17 | S1-suffix0 | 8192 | 76 | 24.6 | 245.85 | 56 | 2857 | 202 | 64921 | 24405 | 794 | 13563.4 |
| 6 | S1-pool32 | 8192 | 101 | 24.55 | 240.8 | 58.2 | 2856 | 190 | 64924 | 24024 | 794 | 13560.3 |
| 27 | S1-vramres1400 | 8192 | 101 | 22.95 | 240.3 | 57.65 | 2623 | 882 | 65619 | 22931 | 794 | 12871.5 |
| 10 | S1-pool127 | 8192 | 76 | 22.85 | 246.8 | 62.2 | 2856 | 190 | 64924 | 22068 | 794 | 13560.3 |
| 16 | S1-spec8 | 8192 | 77 | 20.85 | 247.95 | 40.3 | 2856 | 178 | 64924 | 32745 | 794 | 13560.3 |
| 2 | S0b-mmap | 8192 | 12 | 7.5 | 116.7 | 58.35 | 2856 | 318 |  | 22883 | 0 | 249029 |

- **FAILED** `S1-expertcache-perlayer-3600` (ctx=8192, load=s): INFEASIBLE: needs 13.39 GiB for 3600 slots x 3993600 B but only 9.29 GiB VRAM free on 16 GB; engine refused to start (work/strata-sweep/S1-expertcache-perlayer-engine.log)
- **FAILED** `S1-expertcache-2400-perlayer` (ctx=8192, load=342s): FAILED at first request, not at startup: engine line 'strata serve: the prompt path borrows 1037 CUDA0 cache slots (3.86 GiB)' then 'strata serve: prefill gemm: cublasCreate: cuBLAS status 3' (CUBLAS_STATUS_ALLOC_FAILED, src/prefill/gemm.cu:310-311). Startup succeeded: 'expert cache 2400 slots, 8.93 GiB of VRAM', 'pre-filled 2400 of 2400 slots from the profile; slot 0 verified'. So this is neither a rejected slot count nor an invalid --expert-cache/--expert-profile combination; the prefill path over-commits VRAM. log work/strata-sweep/S1-expertcache-2400-perlayer-engine.log
