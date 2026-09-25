# V100 Stage 1 — CUDA compatibility audit (sm_70 / Volta)

Scope: every GPU-specific component of Strata at commit `1ee8b66`, audited for Tesla V100 (cc 7.0).
Method: static grep/inventory of the 44 `.cu` files + kernel headers for architecture-gated features,
plus empirical runs of the built sm_70 binaries on GPU 0 (V100-PCIE-32GB, driver 580.178.04, CUDA 12.9.86).

## Verdicts

| Feature | Where | sm_70 status | Action |
| --- | --- | --- | --- |
| CMake arch floor (`_base LESS 80`) | `CMakeLists.txt:52` | build rejected | **patched**: floor lowered to 70, message updated |
| Runtime device check `cc_major != 12` | `src/core/device.cu:62` | run rejected | **patched**: `cc_major < 7` now throws |
| `mma.sync...f32.tf32.tf32.f32` (sm_80) | `src/kernels/cuda/native_qsa_score.cu:47` (only PTX MMA in the tree) | needs fallback | **patched**: `#if __CUDA_ARCH__ >= 800` keeps the MMA path; sm_70 uses a warp-FMA path with tf32-truncated inputs (see below) |
| `ldmatrix.sync.m8n8.x4/.x2.b16` (sm_75) | same file, `load_a`/`load_b` | not on Volta | **patched**: helpers compiled out for `__CUDA_ARCH__ < 800` |
| `cp.async` (sm_80) | tree-wide | unused | works unchanged |
| bf16 hardware intrinsics / `__nv_bfloat16` | tree-wide | none (bf16 is pure `uint16` bit tricks, `bf16_bits.hpp`) | works unchanged |
| FP8 (`e4m3`/`e5m2`), `stmatrix`, `cluster.*`, `redux`, `setmaxnreg`, `tcgen` | tree-wide | none | works unchanged |
| `__CUDA_ARCH__` guards | tree-wide | none existed | added in the two patched spots only |
| cuBLAS GEMM, `CUBLAS_COMPUTE_32F` with `CUDA_R_16BF` inputs | `src/prefill/gemm.cu` | **verified OK on V100** (empirical 64³ GEMM, cuBLAS 12.9) | works unchanged |
| cuBLAS `CUDA_R_16F` GEMM | `src/prefill/gemm.cu` | verified OK (same test) | works unchanged |
| Dynamic shared memory, `cudaFuncSetAttribute` opt-in | `fused_gr.cu` (80 KB), `qsa.cu` (runtime limit query) | V100 opt-in limit 96 KB | fits; `qsa_attend` already queries `cudaDevAttrMaxSharedMemoryPerBlockOptin` at runtime |
| CUDA graphs, `cudaHostAlloc(Mapped)`, `cudaHostRegister` (pinned expert arena) | `core/*`, `platform/*` | standard, no arch gate | works unchanged; see baseline note on `ulimit -l` |
| `--use_fast_math` on the 14 native kernels | `CMakeLists.txt` | fine on Volta (FTZ) | works unchanged |
| CPU AVX-512 expert kernels | `src/kernels/cpu/{expert,iq_avx512}.cpp` | host has AVX2 only | probe-gated at runtime; AVX2 Q2 kernel + ggml-cpu fallbacks exist — to confirm at first model run |

## The one real kernel: QSA scorer fallback

`native_qsa_score.cu` computes, per 4-cell KV block row, `sum_h ReLU(sum_d pooled[row][d] * query[h][d])`
plus block bias and a `1e9` bonus on the incomplete tail block. The sm_80 path stages through shared memory
and uses `ldmatrix` + tf32 MMA. The sm_70 path:

- each of the 2 warps owns 16 consecutive rows; each lane accumulates 4 of the 128 dims per head with
  `fmaf` on inputs truncated to tf32 (`& 0xFFFFE000`) — the MMA consumes raw F32 bits that the unit
  interprets as tf32 (low 13 mantissa bits ignored), so truncation matches the reference input exactly;
- 5-step `__shfl_xor_sync` butterfly per (row, head); per-row `fmaxf` ReLU; left-associative head adds,
  then `__fadd_rn` bias and tail bonus — same operation order as the MMA path.

Verified (all on GPU 0, sm_70 build):
- SASS of `native_qsa_score.cu.o`: 0 `ldmatrix`, 0 `MMA`, 448 `FFMA.FTZ`, 338 `SHFL`, 0 shared-memory ops.
- `build-sm70/strata` fatbin: **only sm_70 cubins** (no other arch present).
- Standalone unit test vs an independent CPU reference (15 `n_kv` values incl. 1, 3, 5, 63, 65, 2048–2051):
  worst relative error **2.6e-6**, no cell mismatches, cells `[n_kv, max_cells)` left untouched.
  (Test kept in-tree as `src/kernels/qsa_score_hostref_test.cpp`, CMake target `qsa_score_hostref_test`.)

Note: in this commit the scorer is dormant in the engine (`native_qsa_score_set_enabled` default false, no
caller) — the live decode path is `qsa_decode_attn.cu` (pure FMA/shuffle, already sm_70-safe). The fallback
therefore matters for forward compatibility and any flag-enable, not for today's hot path.

## Empirical evidence (sm_70 build on GPU 0)

- `strata-device --selftest`: OK — cc 7.0 accepted, memory plan at 20 480 ctx fits (KV+indexer 0.27 GB,
  expert cache 5.68 GB / 4105 slots, VRAM pool 5.94 GB).
- 16/16 kernel parity selftests pass: dequant_s2, s2_gemv, s2_gemv_q8, shared_expert, gr, gdn, sampler,
  rope, quantize_act, router_top10, s_gemv, s_gemv_q8k, elementwise, bf16_gemv, qsa, kv_q8.
- `./strata --help` (usage path) initializes and exits 0.

## Known residual risks (to close in Phases 5–7)

1. Pinned-expert-arena `cudaHostRegister` under a low `ulimit -l` (baseline note 3).
2. AVX-512-free CPU expert path (AVX2 fallback) — first real-model run will exercise it.
3. `qsa_attend` shared-memory sizing at large `max_ids` (96 KB V100 limit vs 164 KB sm_80) — context-length
   dependent; the runtime check aborts loudly rather than mislaunching.
4. Long-context prefill bandwidth over PCIe 3.0 x16 (V100 PCIe card) vs the NVLink-less setup the model was
   tuned for — a performance question, not a correctness one.
