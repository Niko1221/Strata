# Raw measurements (2026-10-23)

## strata-concurrent (CPU<->GPU DRAM contention), source build
Command: `build/strata-concurrent.exe --file experts_dummy.bin --cpu-threads 6 --gpu-roles 200000`
```
arena 1.56 GiB pinned; 6 CPU threads; 200000 GPU roles
  mode                     wall   CPU read          GPU
  CPU only             3.012 s   CPU read  73.54 GB/s   GPU     0.0 roles/s
  GPU only            12.829 s   CPU read   0.00 GB/s   GPU 15589.5 roles/s
  BOTH                15.360 s   CPU read  59.09 GB/s   GPU 13020.6 roles/s
  independence check:
    CPU throughput lost when the GPU also runs :  19.6%
    GPU throughput lost when the CPU also runs :  16.5%
```

## Production telemetry (strata_perf.csv, n=1492)
- decode_tok_s median 53.8 (p10 43.5 / p90 69.7)
- fresh-ingest median ~153 new tok/s; draft acceptance 68.2 %
- expert cache hit ~93 %; routed-over-PCIe ~0.5 %
- corr(decode, pcie_share) = -0.06

## GPU utilization during a sustained localhost decode (nvidia-smi, 2s samples)
- GPU0: avg 39 %, median 52 %, max 100 %; >80 % in 8 % of samples
- GPU1: avg 39 %, median 50 %, max 84 %; >80 % in 3 % of samples

## Method scripts
- `bench/bottleneck_probe.py` (HOT/COLD decode probe; note: cache stayed ~93 % warm in both
  phases, so the hot-vs-cold gap was prompt-complexity confounded, reported for completeness).
- Reconnaissance: `--pipeline-windows` capped at 2 in source (`min(2, atoi(...))`); AVX1
  kernels (`kq_avx1`, `bf16_rows_dot_multi_avx1`, `q2_0_gguf_rows_multi_avx1`) + `cpu_avx1_ok()`
  routing already ship upstream.