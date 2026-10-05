# ARM64 / NVIDIA GB10 status

This branch records the work needed to run Strata on NVIDIA GB10 systems such as
DGX Spark and GX10.

## What works

* CUDA 13 and `sm_121` configure successfully on an ARM64 host.
* `strata-device --list-devices` reports the GB10 and its unified-memory pool.
* The setup hardware probe now accepts the `[N/A]` `memory.total` value returned
  by `nvidia-smi` for GB10 and uses host physical memory for model-fit checks.

## Current blocker

The CPU expert fallback is currently x86-specific. It includes `immintrin.h` and
`cpuid.h` and compiles several translation units with AVX/AVX-512 flags. An ARM64
build therefore stops before the Strata server is linked. The CUDA kernels are
not the blocker; the next step is an ARM/NEON CPU expert path (or a supported
GPU-only path) followed by a real model parity and throughput test.

Do not download the 70–100 GB model until that engine build is complete.
