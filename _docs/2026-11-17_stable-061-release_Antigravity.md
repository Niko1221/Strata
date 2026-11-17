# Hakua / Strata 0.6.1 Implementation & Release Log

## Overview
Built, benchmarked, packaged, published (`v0.6.1` / `latest`), and overwrite-installed Hakua / Strata `0.6.1` (App `0.6.1`, Native Q38 `0.6.1` SM120, Control `0.6.0`), along with 8-language README updates, King's English license/upstream acknowledgements, and CI/CD fixes on `main` (`bbdcf3fdc`, `fdb8f6727`, `f4a9ef39c`).

## Implemented Changes
- **Native Q38 Engine (`0.6.1`, `src/qwen38/`, `include/strata/qwen38/`)**:
  - `include/strata/qwen38/quant_tensor.hpp`: Per-tensor mixed-quantization polymorphism (`StrataQuantTensor`) supporting 11 GGML/Strata types (`F32`, `F16`, `BF16`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ3_S`, `IQ2_S`, `IQ4_XS`, `PQ2_0=142`).
  - `include/strata/qwen38/universal_layer.hpp`: Metadata-driven `UniversalTransformerLayer` table.
  - `src/qwen38/prefill_gemm.cuh`, `src/qwen38/prefill_gemm.cu`: Double-buffered FP16 `cublasGemmEx` (`prefill_gemm=cublas_tensor_op`) + opt-in SM120 FP8 `cublasLtMatmul` (`STRATA_FP8_PREFILL=1`).
  - `src/qwen38/spin8_triality.cuh`, `src/qwen38/kv.hpp`, `src/qwen38/kv.cu`: `KvCache::enqueue_batch` (`spin8_store_q_batch_kernel`) and `KvCache::rewind(int)`.
  - `include/strata/qwen38/admission.hpp`, `src/qwen38/engine.cu`: `blk.64.nextn.*` MTP `K=3` draft/verify/rollback loop and `--spec-draft-n-max 0..3`.
- **Packaging, Versioning & Docs**:
  - Updated `VERSION`, `apps/desktop/package.json`, `apps/desktop/src-tauri/Cargo.toml`, `apps/desktop/src-tauri/Cargo.lock`, `apps/desktop/src-tauri/tauri.conf.json`, `apps/host/package.json` to `0.6.1` (while keeping `CONTROL_VERSION` and `CMakeLists.txt` base version at `0.6.0` for signed Control/base `strata.exe` compatibility).
  - Updated `docs/data/release.json` (`v0.6.1` current + `v0.6.0` historical), `data/settings-register.json`, and `docs/SETTINGS.md` (`STRATA_FP8_PREFILL`).
  - Rewrote all 8 language READMEs (`README.md`, `README.ja.md`, `README.de.md`, `README.es.md`, `README.fr.md`, `README.pl.md`, `README.pt-BR.md`, `README.zh-CN.md`) for `0.6.1` and added a King's English **Acknowledgements & Gratitude** section to `README.md`.

## Verification & Benchmark Results
- **140-token Prefill + 8-token Decode (`qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf`, `--ctx-size 4096 --kv spin8_k8v4 --serve`, RTX 5060 Ti SM120)**:
  - Sequential baseline (`--conservative-sync`, `prefill_gemm=off`): `pp = 36,760.7 ms` (`3.81 tok/s`), `tg = 1,905.6 ms` (`4.20 tok/s`).
  - Tensor Core + `KvCache::enqueue_batch` (`prefill_gemm=cublas_tensor_op`): `pp = 11,387.4 ms` (`12.29 tok/s`), `tg = 1,614.7 ms` (`4.95 tok/s`) — **3.23x prefill speedup**.
- **Local Docs Check**: `tools/check --only docs` passed (`6 passed, 0 failed`).
- **Release Packaging (`H:\Strata\releases\stable-061\assets\`) & GitHub Release**:
  - `Strata-0.6.1-windows-x64-setup.exe` (`126,662,410` bytes)
  - `strata-engine-0.6.1-windows-x86_64-cuda-cuda13.tar.gz` (`1,479,650,087` bytes)
  - `Strata-0.6.1-windows-x64.tar.7z` (`1,670,163,256` bytes)
  - `SHA256SUMS.txt` (`834` bytes)
  - Published at https://github.com/zapabob/Strata/releases/tag/v0.6.1 (`--latest`).
- **Local Overwrite-Installation**:
  - Installed `Strata-0.6.1-windows-x64-setup.exe /S /D=C:\Users\downl\AppData\Local\Programs\Strata` (`strata-desktop.exe` ProductVersion: `0.6.1`).
  - Installed `0.6.1-windows-x86_64-cuda-cuda13` (`strata-q38.exe --version`: `strata-q38 own-native 0.6.1 CUDA13 SM120`).
- **CI/CD (`zapabob/Strata`)**:
  - `Docs` (`44078374127`, `44078126433`), `Deploy GitHub Pages` (`44078126229`), `Config snapshot` (`44077224799`), `Backend compile checks` (`44077224644`), `Native preview CPU checks` (`44077224658`, `44080689298`), `Portable monitor client` (`44080689328`): **success**.
  - `Checks` (`44077224929`) and `Minimal serving runtime` (`44080689284`): `in_progress` at log write time.
  - **140-token Prefill + 8/32-token Decode Stage 1–5 Progression (CUDA 13.5, `qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf`, `--kv spin8_k8v4 --ctx-size 4096`)**:
    - Stage 1 (separated `TIMING` + `GEMM_STATS` + 16-aligned FP8 tiling + `cublasLt` plan cache + `estimate_bytes` accounting): CUDA 13.3 cold `pp = 11,220.474 ms` vs CUDA 13.5 cold `pp = 11,209.687 ms`, warm `pp = 2,734.626 ms`.
    - Stage 2 (packed-block GEMV `mv_q8_0`, `mv_qk45`, `mv_q6_k`, `mv_iq2_s`, `mv_iq3_s`): `GEN 8` `decode_forward_ms` `1,445.019 ms` (`206.4 ms/step`) -> `636.682 ms` (`90.95 ms/step`, **2.27x** faster); `GEN 32` steady-state `3,204.240 ms` (`103.36 ms/step`).
    - Stage 3–5 (`fused_gdn_conv_l2_batch` + `fused_gdn_step_norm_silu_batch` + `attention_normalize_batch` + 2-stream GEMM overlap + `spin8_attend_batch_kernel`): warm `140`-token `prefill_forward_ms` improved to **`1,558.299 ms` (`89.8 tok/s`, 23.6x faster than `36,760.7 ms` sequential baseline)** with exact token parity (`198,206,207,208,209,210,211,212`).
    - Stage 6 (CUDA Graph Decode preserving 8-layer STOP/NaN fences: `gdn_graphs[16]` + `ffn_graphs[16]` + `head_graph`): `GEN 32` `decode_forward_ms` improved to **`2,639.048 ms` (`85.13 ms/step`, `11.75 step/s`, 2.42x faster than `206.4 ms/step`)**; with `--spec-draft-n-max 3` (MTP `K=3`), `GEN 32` completed in `2,623.952 ms` (`~12.2 tok/s`).
- **Limits**: `releaseQualified=false`; component/feed signing was skipped for `0.6.1` per user instruction (`署名関係はいいよ`).
