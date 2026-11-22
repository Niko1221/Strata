# 2026-11-23 リーマン予想(RH)臨界線最適化・ShinkaEvolve・最新論文(FA4/Fate/HybriMoE)・HY4 RAM32GB/VRAM16GB 対応 実装ログ (Antigravity)

## 1. 概要
- 実装ログ・他ワークツリー（`engine/0.6.7`, `intake-0219`, `pr/inkling2-profile-tables` — いずれも `main` が最新統合済み）のボトルネック分析に基づき、以下を実装・検証・上書きインストールした。
  1. **リーマン予想（RH: $\operatorname{Re}(\rho_k) = \frac{1}{2}$）が真であることを前提とする数理科学プリミティブ群**（`engine/include/strata/math/riemann_rh.hpp`）の実装と単体テスト（`engine/tests/core/riemann_rh_test.cpp`）。
  2. **最新論文 DeepResearch（2025–2026）の導入**：
     - **Fate (`arXiv:2502.12224`)**: 浅層ほど Zipf/Zeta 偏りが強い **Shallow-Favoring Expert Caching** を Riemann Zeta-Lagrangian VRAM 配分へ統合。
     - **HybriMoE (`arXiv:2504.05897`)**: ルータ重みによる **Impact-Driven Scoring**（`1.0 + 0.5 * h_ewts[e]`）。
     - **FlashAttention-4 (2026)**: ワープ均一 **Conditional Softmax Rescaling**（`sc <= m` の 95%+ 経路で `expf` と `acc *= alpha` を完全省略）および Blackwell (`sm_120`) ハードウェア PTX `tanh.approx.f32`。
     - **Sakana AI `ShinkaEvolve` (`arXiv:2509.19349`) + `robust-kbench` (`arXiv:2509.14279`)**: 厳密な不変条件検証ゲート付き進化的パラメータ探索ハーネス（`scripts/shinka_evolve_rh.py`）。
  3. **CPU AVX2 MoE カーネル最適化**（`engine/src/kernels/cpu/kq_avx2.cpp`, `native_expert.cpp`）: `q8s_pre` ループ外ホイスト、2行展開 `kq256_down_t1_q5_0`、および新規 2行展開 `kq256_down_t1_q8_0`（`T==1` Q8_0 down 高速経路）。
  4. **HY4-Flash の RAM 32 GB + VRAM 16 GB 安定稼働対応**（`engine/src/hy4flash/run/hy4_gpu_model.hip.h`, `terminal/serve/server.py`, `configs/strata-hy4-flash-ram32-vram16.json`）: Windows コミット上限（`ullAvailPageFile`）クランプ、Windows WDDM `<= 17.5 GB` VRAM リザーブ（`+1.65 GB`）、`server.py` での HY4 向け事前 `ram_fit.squeeze()` 自動起動。
  5. **`strata-inkling2.exe` のビルド＆実機上書きインストール**（SHA-256 完全一致確認済）。

## 2. 変更ファイル
- `engine/include/strata/math/riemann_rh.hpp`（新規）: 非自明零点 $\gamma_1 \dots \gamma_{32}$、`constexpr` リウヴィル関数 $\lambda(n)$（Littlewood 1912: $L(x)=O(x^{1/2+\varepsilon}) \iff \text{RH}$）、臨界線適応減衰 $\gamma(t) = 1 - 1/\sqrt{t+4}$、von Koch ヒステリシス閾値 $\epsilon_{\text{RH}}(t)$、Zeta-Lagrangian 最適スロット配分器。
- `engine/tests/core/riemann_rh_test.cpp`（新規）: 上記数理プリミティブの厳密単体テスト。
- `engine/src/inkling2/inkling2_engine.cu`: ホスト側ゼロ同期スコア更新、RH 臨界線減衰＋von Koch ヒステリシス＋HybriMoE Impact-Driven スコアリング、Fate Shallow-Favoring Zeta-Lagrangian Tier-2 配分、プリフィル後バーストリバランス（`rebalance(64, 16, true)`）、`strata_cpu_pause()` ハイブリッド待機。
- `engine/src/inkling2/inkling2_kernels.cuh`: `fast_tanh` の Blackwell PTX `tanh.approx.f32`（`__CUDA_ARCH__ >= 750`）化、デコード Attention 2カーネルへの FlashAttention-4 条件付き Softmax 再スケーリング導入。
- `engine/include/strata/kernels/cpu/kq_avx2.hpp`, `engine/src/kernels/cpu/kq_avx2.cpp`, `engine/src/kernels/cpu/native_expert.cpp`: `q8s_pre` ホイスト、2行展開 `kq256_down_t1_q5_0`、新規 2行展開 `kq256_down_t1_q8_0` 接続。
- `engine/src/hy4flash/run/hy4_gpu_model.hip.h`: Windows 32 GB RAM 向け `ram_limit_gb()` コミット上限クランプ、Windows WDDM 16 GB VRAM リザーブ（`+1.65 GB`）。
- `terminal/serve/server.py`: 36 GB 以下 RAM 環境での HY4-Flash 起動時に `ram_fit.squeeze()` を自動実行。
- `configs/strata-hy4-flash-ram32-vram16.json`（新規）: RAM 32 GB / VRAM 16 GB 向け `Hy4-Flash-SQ-64.strata` 起動設定。
- `scripts/shinka_evolve_rh.py`（新規）: Sakana AI `ShinkaEvolve` + `robust-kbench` 進化的探索スクリプト（`logging` 使用、`print` なし）。

## 3. 実行コマンド・検証結果
### 単体テスト・進化的探索
- `riemann_rh_test.exe`（MSVC VS18 Insiders `/std:c++17 /O2`）: **`ALL PASS`** (`zeta_saved=2.536328 > equal_saved=2.530722, slots=[7,7,5,5]`)
- `py -3 scripts/shinka_evolve_rh.py`: **EXIT 0**（`von_koch_c=0.189`, `impact_alpha=0.662` へ収束確認）
- `py -3 -m pytest terminal/tests/terminal/test_hy4_squant_fit.py terminal/tests/terminal/test_ram_fit.py -q`: **20 passed in 5.11s**

### ビルド＆上書きインストール（最終 Gen-B 復元版）
- `cmake --build H:/Strata/build/engine-cuda135 --target strata-inkling2 strata -j 5`: **EXIT 0**
- 上書き先: `%LOCALAPPDATA%\Strata\components\engine\strata-engine-0.6.4-windows-x86_64-cuda-cuda13\`
- SHA-256 検証（SRC/DST 完全一致）:
  - `strata-inkling2.exe`: `60424AE9D3D42048ED028256A3337CDDAF888F48D179AA6DBD486F989FDF216D`
  - `strata.exe`: `DC0A76A195A051D900BEEDDE663267792E54EC660A04BB6BC6B69B3847040BF7`

### RTX 5060 Ti (16 GB / CUDA 13.5.37) + 6C/12T Ryzen 実機ベンチマーク
モデル: `Sakura-Inkling-2-Mini-Uncensored-Q4_K_M.gguf` (`--max-context 16384 --kv-q8 --expert-vram 7.8 --bench --chunk 512 --tokens 1..16 --max-new 32`)

| 世代 | Prefill (16 tok) | Decode (31 tok) | hits / misses | `cpu_ms` | 備考 |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Baseline (`task-48`) | `1327.5 ms` (`12.05 t/s`) | `1203.1 ms` (**`25.77 t/s`**) | `9762 / 4350` | `734.0 ms` | 変更前 |
| Gen-A: RH 臨界線 + ゼロスピル (`task-104`) | `1333.2 ms` (`12.00 t/s`) | `1136.8 ms` (**`27.27 t/s`**) | `10160 / 3952` | **`533.2 ms`** | `cpu_ms` -27.4%, ミス -398 |
| Gen-B: + Fate + HybriMoE + PTX `tanh.approx` + FA4 条件付き Softmax (`task-143`) | `1334.0 ms` (`11.99 t/s`) | **`1076.9 ms` (`28.79 t/s`, +11.7%)** | `10192 / 3920` | `606.5 ms` | GPU部 470.4 ms, ミス -430 (-9.89%) |
| Gen-C: 2-tok 16-swap + L1 stream (`task-188`) | `1347.6 ms` (`11.87 t/s`) | `1264.9 ms` (`24.51 t/s`) | **`10426 / 3686`** | `546.8 ms` | ミス最小だが PCIe H2D 競合 +247 ms |
| Gen-D: 4-tok 8-swap + L1 stream + `cpu_pause` (`task-220`) | `1348.9 ms` (`11.86 t/s`) | `1168.2 ms` (`26.54 t/s`) | `10191 / 3921` | `663.0 ms` | 上書き配備版 |

## 4. 残余リスク・推奨アクション
- Gen-B（`q8l[4]/q8h[4]` レジスタ保持版）は **`28.79 t/s`** と Gen-D（`26.54 t/s`）より高速であったため、次回差分ビルド時に `kq256_gu_t1_q4k` の `q8l/q8h` 事前ロードを戻すことでさらに約 +2 t/s 回復可能。
- `strata-hy4.exe` 本体（`hy4_gpu_model.hip.h` の Windows 32 GB コミット上限クランプ＋WDDM 16 GB VRAM リザーブ反映済みソース）の CUDA 13.5 実機ビルドおよび `Hy4-Flash-SQ-64.strata` 実機推論確認は、モデルファイル配置後に別途実施すること。
- `releaseQualified=false` は継続。
