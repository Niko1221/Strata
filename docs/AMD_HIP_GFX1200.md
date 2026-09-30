# Experimental AMD HIP backend on gfx1200 (RDNA4, RX 9060 XT)

The HIP backend ([docs/AMD_HIP.md](AMD_HIP.md), [docs/AMD_HIP_PERFORMANCE.md](AMD_HIP_PERFORMANCE.md)) runs
unchanged on RDNA4 gfx1200 (RX 9060 XT, 16 GiB) with the deltas below. Engine 0.1.29, ROCm 7.2 with
hipBLASLt 100202, Linux.

What is gfx1200-specific:

1. `cmake/hip_backend.cmake`: the backend builds for one arch from `STRATA_HIP_ARCHS` (gfx1100, gfx1200) and
   records it as `STRATA_HIP_ARCH`.
2. `src/core/device.cu`: the startup check compares the running card's `gcnArchName` against the arch the
   binary was compiled for (and still requires wave32), so a gfx1100 build carried to a gfx1200 card fails
   with a clear message instead of later with an `invalid device function`; rebuild with
   `-DCMAKE_HIP_ARCHITECTURES=<card arch>`.
3. `setup.py`: `AMD_ARCHS` / `AMD_NAMES` list the RX 9060 XT; setup compiles with the detected arch and finds
   the hipBLASLt table below by `{arch}-hipblaslt-*.txt`.
4. `tools/hip/gfx1200-hipblaslt-100202.txt` (new): the solution table for the dense prompt GEMMs, calibrated
   on the card with `tools/hip/tune_hipblaslt` (67 rows, bf16 + f16). The engine refuses a table made for
   another arch or hipBLASLt version and falls back to plain hipBLAS as before.
5. `tools/hip/gfx1200-run.sh`: the launcher used for the measurements (IQ1_M Coder pack, MTP).

Nothing else changes: the RDNA4 cards are wave32 like gfx1100, and the same kernels (including `sudot4`
dp4a) compile as-is. On this ROCm 7.2 install the build needs
`-DCMAKE_HIP_ARCHITECTURES=gfx1200` (or `./setup.sh --backend hip` with the card present).

Hand build:

```sh
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_PREFILL_MMQ=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1200 -DSTRATA_GGML_DIR=<pinned llama.cpp>
cmake --build build-hip -j
STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1200-hipblaslt-100202.txt" \
  ctest --test-dir build-hip --timeout 120 -E '^(ple_parity|platform_memory_test)$'
```

Measured on RX 9060 XT 16 GiB + Ryzen 9 7950X + 64 GiB RAM (Qwen3.8-Flash-Next IQ1_M Coder, greedy,
MTP `--spec 4 --spec-min-p 0.5`, `tools/hip/gfx1200-run.sh`; decode speed scales with MTP draft
acceptance, which depends on the prompt):

| prompt | prefill | decode |
|---|---|---|
| 2,374 tokens | 540 tok/s | 27.1 tok/s (0.69 acceptance on this prompt; 31.0 at 1.00 acceptance on the earlier code prompt) |
| 65K (`--kv int8 --kv-resident 65536 --prefill 16384`) | 748-753 tok/s | 26-40 tok/s by acceptance |
| 130K (same flags) | 725 tok/s | - |

ctest (excluding `ple_parity` and `platform_memory_test`, as in docs/AMD_HIP.md): 32/32, including
`hip_prefill_hipblaslt_gemm` against the table above. Deterministic greedy output across runs; the
launcher keeps 1792 MiB of VRAM reserved (`--vram-reserve-mib`).

One observation for upstream: at 0.1.27, raising the decode-side attention query batch (`attn_batch` in
`src/prefill/prefill.cpp`) from 32 to 64 overflowed the prompt path's borrowed buffer region at any chunk
size ("prefill: device buffers for a chunk of 256 tokens do not fit"). On 0.1.29 it fits again and
measures 554 vs 540 tok/s short / 763 vs 748 tok/s at 64K on this card - within a few percent - so this
PR keeps the stock 32.
