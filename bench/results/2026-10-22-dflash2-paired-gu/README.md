# Rejected paired gate/up AVX2 kernel — 2026-10-22

Implementation experiment at fixed settings; no launch-parameter search. For one- and two-token native NVFP4 expert groups, gate/up dot products share each Q8 activation load and interleave independent accumulators. Each gate/up accumulator retains the original 01 -> 23 FMA sequence and final lane reduction. Wider groups retain the original kernel. Weights, shared head and target verification policy are unchanged.

CUDA/CPU build passed. With `STRATA_NVFP4_PAIRED_GU=1`, the existing `nvfp4_avx512_parity` passed on random data and the local pack's first real expert, covering dot products and gate/up with scales for 1..9 tokens. The AVX2 checks require bit equality with ggml's own AVX2 reference.

Separate existing DRAM microbenchmark on Ryzen 5900X, 11 threads, 1536 distinct synthetic expert blobs (~4.2 GB), unchanged prefetch 2048, best of the benchmark's three passes:

| Tokens per expert | Original | Paired | Change in GB/s |
|---|---:|---:|---:|
|1|43.2 GB/s|42.2 GB/s|−2.3%|
|2|42.2 GB/s|42.6 GB/s|+0.9%|

These are mixed microbenchmark results, not an end-to-end gain. No full-model throughput was claimed or measured for this paired kernel. The prototype was rejected and removed from active source. It does not establish the requested +32% DFlash2 advantage. The ~43 GB/s measurements are consistent with substantial DRAM pressure; they are not a proof of a strict hardware or algorithmic throughput ceiling.

Commands actually run, sequentially without a model benchmark:

```
cmake --build build-dflash2 --target strata nvfp4_avx512_parity -j4
STRATA_NVFP4_PAIRED_GU=1 build-dflash2/nvfp4_avx512_parity ../Strata-data/dflash2-acceptance-20261022/expert0.bin
build-dflash2/nvfp4_avx512_parity --bw 11 1 avx2
STRATA_NVFP4_PAIRED_GU=1 build-dflash2/nvfp4_avx512_parity --bw 11 1 avx2
build-dflash2/nvfp4_avx512_parity --bw 11 2 avx2
STRATA_NVFP4_PAIRED_GU=1 build-dflash2/nvfp4_avx512_parity --bw 11 2 avx2
```

Logs are beside this report. `tested-source.patch` records the actual combined working tree: it contains the earlier CPU scheduling prototype as well as the paired kernel, but its pipeline switch was unset during these kernel checks. Apply to parent `649c8e06` to reproduce. The real expert is the first 2,764,816 bytes of the recorded NVFP4 pack; weights and binaries are not included. No HIP/SYCL/Windows, other CPUs/models, or production-concurrency validation was performed for these rejected changes.
