# V100 Stage 1 — SM70 build

How the V100-capable Strata binaries are produced on this machine (commit `6bb6b94`, branch
`feature/v100-moe`).

## Requirements

- CUDA toolkit 12.9 (`/usr/local/cuda`; sm_70 still compiles, with a deprecation warning).
- CMake ≥ 3.24 (system CMake 3.22.1 is too old) — the repo's `.venv` provides 4.4.3.
- Ninja, gcc 11.4, driver 580.x.

## Commands

```sh
cd /mnt/ssd/strata/Strata
.venv/bin/cmake -S . -B build-sm70 -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DFETCHCONTENT_BASE_DIR=/mnt/ssd/strata/fetch
.venv/bin/cmake --build build-sm70 -j 40
```

- `CMAKE_CUDA_ARCHITECTURES` must be passed **explicitly**: CMake's `native` detection reports 52 on
  this machine (quirk, see baseline), and the project's own 120 default would build the wrong fatbin.
- `FETCHCONTENT_BASE_DIR` is shared with `build-orig` so the pinned llama.cpp checkout is cloned once.
- The unmodified tree's sm_120 baseline build lives in `build-orig/` (recorded in the baseline doc).

## Verification performed (Phase 4)

1. `cuobjdump --list-elf build-sm70/strata` → fatbin contains **only sm_70 cubins**.
2. `cuobjdump -sass` on `native_qsa_score.cu.o` → 0 `ldmatrix`, 0 `MMA`, 448 `FFMA.FTZ`, 338 `SHFL`,
   0 shared-memory ops in the fallback scorer.
3. GPU runtime (GPU 0, V100-PCIE-32GB, `CUDA_VISIBLE_DEVICES=0`):
   - `strata-device --selftest` → OK (cc 7.0 accepted; 20 480-ctx plan fits).
   - 16/16 kernel parity selftests OK (dequant_s2, s2_gemv, s2_gemv_q8, shared_expert, gr, gdn,
     sampler, rope, quantize_act, router_top10, s_gemv, s_gemv_q8k, elementwise, bf16_gemv, qsa, kv_q8).
   - `qsa_score_hostref_test` (new, in-tree) → OK, worst rel err 2.6e-6 vs CPU reference.
   - `./strata` usage path exits 0.

## Rebuild / test one-liners

```sh
.venv/bin/cmake --build build-sm70 -j 40          # incremental rebuild
cd build-sm70 && CUDA_VISIBLE_DEVICES=0 ctest --output-on-failure -R "parity|selftest|hostref"
```

`ctest` status on a clean tree (no model yet), sm_70 build, GPU 0: **19/20 pass** —
all 16 kernel parities, `qsa_score_hostref_test`, `ple_reader_selftest`, `cuda_device_selftest`.
The one red test is `ple_parity`, which hard-errors by design on its missing required fixtures
(`bench/micro/` and the Q2_0 GGUF shard are absent from the published tree); re-run it once the
model is in `models/`.

Note: `enable_testing()` is now called unconditionally (build commit), so the unconditional
`add_test()` registrations are visible to ctest even with `STRATA_BUILD_TESTS=OFF`.
