# 2026-11-22 RTX 5060 Ti (`sm_120a`) / CUDA 13.5 ネイティブエンジン高速化ログ (Antigravity)

## 環境
- GPU: NVIDIA GeForce RTX 5060 Ti (16,311 MiB VRAM, Compute Capability 12.0, Driver 621.24)
- CUDA Toolkit: `13.5.37` (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin\nvcc.exe`)
- ビルド: `H:/Strata/build/engine-cuda135` (Ninja, `-j 5`, `CMAKE_CUDA_ARCHITECTURES=120`)

## 実装した変更点
1. **LingMini2 (`engine/src/inkling2/cuda/`)**
   - `ink_runner.cuh`: Dense `FMT_Q8ROW` の `cublasGemmEx(CUDA_R_8I, CUDA_R_32I)` 経路を `m.sm_cc == 61` 限定から `(m.sm_cc == 61 || m.sm_cc >= 75)` へ拡張し、`sm_120` で INT8 Tensor Core 直接演算を有効化。
   - `ink_moe_cuda.cuh`: `sm_120a` (`mxf4nvf4` / `mxf8f6f4`) MoE Tensor Core および `use_fp16_tc` の適用閾値を `M >= 24` から `M >= 16` へ引き下げ。
2. **Qwen3.8-27B / 9B (`engine/q38/src/`)**
   - `engine.cu`: `cudaHostAlloc` による `pinned_error` を導入し、非 `conservative` 時の 8 層ごとの `LayerFence` 内 `cudaStreamSynchronize`（WDDM 同期ストール）を解消。`mm_batch()` の Folded Hadamard 変換を `Q::hadamard_batch` へ統合。
   - `ops.cuh` / `ops.cu`:
     - `had_batch` / `hadamard_batch` を追加（2D グリッド一括起動）。
     - `PQ2_0` (type 142) 向けワープ並列カーネル `mv_pq2_0` を追加。
     - llama.cpp (`mmvq.cu`, MIT) を参考に、`Q8_0` / `Q4_K` / `Q5_K` / `Q6_K` 向けマルチトークン MMVQ（`mv_q8_q8_0_batch`, `mv_q8_qk45_batch`, `mv_q8_q6k_batch`, `NCOLS=2/4/8`）および `quantize_q8_1_batch` を実装。
   - `prefill_gemm.cu`:
     - `prepare_act_scaled_fp16_reg_kernel` に 9B 向け `K=4096, 12288` 特殊化を追加。
     - `n_tokens = 2..16` の `Q8_0` / `Q4_K` / `Q5_K` / `Q6_K` 投影を `matvec_q8_1_batch` へディスパッチ。
3. **ビルド修正 (`engine/cmake/strata_gpu_engine.cmake`, `engine/q38/CMakeLists.txt`, `engine/src/prefill/prefill.cpp`)**
   - `q38/include` パス修正、`enable_language(C)` 追加、`Prefill::arm_cpu_share_split` 定義補完。

## 実機計測結果（RTX 5060 Ti）
- **`DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf`** (`--ctx-size 4096 --tokens 1..16 --n 64`):
  - Prefill (16 tok): `259.565 ms` -> **`91.258 ms`** (**2.84x 高速化**, `GEMM_STATS fp16=0`)
  - Decode (63 steps): `1097.814 ms` (`57.4 tok/s`) -> **`967.942 ms` (`65.1 tok/s`)** (**+13.4%**)
  - E2E: `1417.800 ms` -> **`1095.802 ms`** (**+22.7%**)
  - 出力トークン完全一致を確認。
- **`Hikari-LoRA-fixed-scale-PQ2_0.gguf`** (`--ctx-size 4096 --tokens 1..16 --n 32`):
  - Prefill (16 tok): `955.621 ms` -> **`783.204 ms`** (**-18.0%**)
  - Decode (31 steps): `865.143 ms` (`35.8 tok/s`) -> **`841.977 ms` (`36.8 tok/s`)**
  - E2E: `1845.952 ms` -> **`1645.004 ms`** (**-10.9%**)
  - 出力トークン完全一致を確認。
- **`strata-inkling.exe`**: CUDA 13.5.37 にてビルド成功を確認（`.strata` モデル未配置のため実機推論ベンチマークは未実施）。
- **`strata-inkling2.exe` (`Inkling-2-Mini-UD-Q4_K_M.gguf`, `--max-context 16384 --tokens 1..16 --max-new 32 --bench`)**:
  - `profile_rules.hpp`: `Sm120Blackwell` (`sm120`) 専用プロファイル追加。
  - `inkling2_kernels.cuh`: SakanaAI ShinkaEvolve 式ハーネス (`evolve_inkling2.py`) で検証済みの `uint64_t` Branchless Key-Pack Router (`k_route_inkling2`) + 4-Warp `__ballot_sync` 並列 Prefix-Scan Grouping (`k_group_experts`) を実装（ビット完全一致 & 1.16x 高速化確認）。
  - `inkling2_engine.cu`: `try_wmma` 閾値を `T >= 16` -> `T >= 48` へ最適化、`ev_route` ポーリングの `std::this_thread::yield()` 追加。
  - Prefill (16 tok): `484.5 ms` (`33.02 t/s`) -> **`463.7 ms` (`34.51 t/s`)**
  - Decode (31 tok): `1294.8 ms` (`23.94 t/s`, `cpu_ms=109.2`) -> **`1256.7 ms` (`24.67 t/s`, `cpu_ms=220.0`)**（GPU 実行時間は `1185.6 ms` -> **`1036.7 ms`**, **-12.6%**）


## 2026-11-23 追記: Sakura-Inkling-2-Mini 対応・WDDM 4階層最適化・CPU MoE 進化・Spin(8) Triality + 最新論文導入
- **Sakura-Inkling-2-Mini (`inkling2_loop`)**:
  - `engine/src/inkling2/inkling2_engine.cu`: 24物理層→48仮想層展開、`blk.L.loop_msg_query.weight` → `loop_res_query`/`loop_kv_res_query` 再構築（UD とコサイン類似度 `1.000000`）、`loop_{q,k,v,ffn}_ctrl_*` フォールバック、Windows WDDM 向け `2048 MiB` 自動リザーブ（`STRATA_VRAM_RESERVE_MB` 上書き可）を実装。
  - 実測（RTX 5060 Ti / CUDA 13.5.37, 16K ctx, 31 tok）: `tg = 13.54 t/s (2290.0 ms)` → **`19.34 t/s (1602.6 ms, +42.8%)`**、`moe_gpu(nv) = 861.4 ms` → **`52.4 ms (16.4x)`**、32/32 トークン完全一致。
- **GPU / CPU MoE カーネル進化**:
  - `engine/src/kernels/cuda/iq_kernels.cu`: `vec_dot_q5_0_q8_1` の `qh` ロードをループ外へホイスト + `SITU` 活性化へ PTX `tanh.approx.f32` (`__CUDA_ARCH__ >= 750`) と逆数乗算を適用。
  - `engine/src/kernels/cpu/native_expert.cpp`: `SITU` へ Padé `[7/6]` 有理関数近似 `fast_tanh_f32` + `STRATA_ROW_PF=2048` + `STRATA_KQ256=1` を導入。
- **FLA (ICLR 2025) GDN + Spin(8) Triality + FlashAttention-2/3 Base-2 `exp2f`**:
  - `engine/src/kernels/cuda/fused_gdn.cu`: デコード版 `gdn_step_norm_vec4_kernel` およびプリフィル版 `gdn_step_norm_vec4_batch_kernel` を接続。
  - `engine/q38/src/kv.cu`: `spin8_attend_decode_reduce_kernel` の 8-lane サブワープ `__shfl_sync` 逆回転化、`log2(e)` 畳み込み + `exp2f` 化、`spin8_store_q_batch_kernel` の `float4` `__ldg` 化。
  - 実測（`Qwen3.5-9B`）: `f16` Prefill `259.565 ms` → **`53.368 ms (4.86x)`**; `--kv spin8_k8v4 --spin8-view plus` で KV 59.4% 圧縮 (`134.2 MB → 54.5 MB`) かつ Prefill **`54.303 ms`** / Decode **`975.945 ms (64.6 tok/s)`**。
  - `Qwen3.8-27B-...-MTP-IQ3_M.gguf`: `IQ3_S (type=21)` 409 テンソルを確認（`strata-q38` ローダは `IQ3_S` 未接続）。
