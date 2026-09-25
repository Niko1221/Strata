# V100 Stage 1 — Baseline

Recorded 2026-09-25, before any code change.

## Strata

- Repo: `Niko1221/Strata`, commit `1ee8b666c13e0a22e28013223995a1e9d9be3498` (2026-09-24, upstream main).
- Work branch: `feature/v100-moe`.
- Working copy: `/mnt/ssd/strata/Strata` (NVMe). A read-only reference clone also exists at
  `/home/noorazman/dsh/strata/Strata` (same commit, clean tree).

## Host

| Item | Value |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 6.8.0-138-generic (x86_64, PREEMPT_DYNAMIC) |
| CPU | 2× Intel Xeon E5-2680 v4 @ 2.40 GHz (14 cores/28 threads each), 56 logical CPUs, 2 NUMA nodes |
| CPU ISA | AVX2, FMA, F16C, BMI2 — **no AVX-512** (affects Strata's AVX-512 CPU-expert fast path) |
| RAM | 125 GiB total |
| Disk | `/` ext4 on /dev/sdb, 457 GB (130 GB free at baseline); `/mnt/ssd` ext4 on /dev/nvme0n1, 916 GB (132 GB free at baseline) |
| NVIDIA driver | 580.178.04 (driver API level CUDA 13.0) |
| CUDA toolkits | 12.9.86 (`/usr/local/cuda`, used for all builds), 12.8.61 (`/usr/local/cuda-12.8`) |
| Compilers | gcc/g++ 11.4.0 (system), nvcc 12.9.86, CMake 4.4.3 (venv; system CMake 3.22.1 is below the required 3.24), Ninja |

## GPUs (verified with `nvidia-smi` + direct `cudaGetDeviceProperties`)

| idx | Model | VRAM | Compute cap | Slot | Notes |
| --- | --- | --- | --- | --- | --- |
| 0 | Tesla V100-PCIE-32GB | 32768 MiB | 7.0 | 02:00.0 | **Stage 1 development GPU** |
| 1 | Tesla V100-SXM2-16GB | 16384 MiB | 7.0 | 03:00.0 | occupied by :8080 llama service |
| 2 | Tesla V100-SXM2-16GB | 16384 MiB | 7.0 | 82:00.0 | router service (3 GB used) |
| 3 | Tesla V100-PCIE-32GB | 32768 MiB | 7.0 | 83:00.0 | occupied (agent LLM via ninfer) |
| 4 | Tesla V100-SXM2-16GB | 16384 MiB | 7.0 | 84:00.0 | **Stage 1 validation GPU** |

All five report cc = 7.0, 96 KB (98304 B) max opt-in shared memory per block.

## Original-project build result (unmodified source)

- `cmake -S . -B build-orig -DSTRATA_ENABLE_CUDA=ON` **without** an explicit
  `CMAKE_CUDA_ARCHITECTURES` **fails at configure**: CMake's `native` architecture detection reports
  `52` on this machine (quirk of CMake 4.4.3 + driver; all five GPUs really are 7.0), and Strata's
  arch guard rejects anything < 80.
- With `-DCMAKE_CUDA_ARCHITECTURES=120` (the project's own default target) the unmodified tree
  **builds cleanly**: `BUILD_OK`, 31 compiler warnings, no errors. Artifacts: `build-orig/strata`
  (13.5 MB) and `build-orig/strata-device`.
- Original test suite not run at baseline: the published tree ships without `tests/` / `bench/micro/`,
  so `STRATA_BUILD_TESTS` defaults OFF; and `strata-device --selftest` from the sm_120 binary is
  expected to refuse a cc 7.0 device at runtime (that refusal is the point of the Stage 1 patch).

## Environment quirks found (relevant to later phases)

1. `nvidia-smi --query-gpu=...ecc.mode...` segfaults on driver 580.178.04; valid field lists work.
2. CMake `CMAKE_CUDA_ARCHITECTURES` "native" detection returns 52 here → always pass the arch explicitly.
3. Shell `ulimit -l` is 8192 KB (soft); `/etc/security/limits.conf` grants noorazman unlimited memlock
   (login sessions). Strata pins a ~34 GB expert arena — runtime launch may need `prlimit`/systemd scope.
4. The agent LLM for this session moved twice: first :8000 (llama.cpp, GPU 0+3), then :8001
   (ninfer-serve, GPU 3). Keep GPU 0 and GPU 4 clear for Strata; do not kill the LLM service.
5. CPU is Broadwell (no AVX-512): Strata's AVX-512 CPU-expert kernels are probe-gated at runtime and
   fall back to the AVX2 Q2 kernel / ggml-cpu for i-quants — verify this at first run (Phase 5).
