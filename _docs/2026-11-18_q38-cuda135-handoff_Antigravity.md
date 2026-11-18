# Qwen3.8-27B CUDA 13.5 / SM120 最適化・引き継ぎ書

- **対象リポジトリ / ブランチ**: `C:\Users\downl\Documents\ChatGPT\Strata` (`main`, 基準 `1f393b027cfdf68e342264fcbaa231501fa9b74b`)
- **検証済み対象モデル**: `H:\Strata\models\qwen38-27b-soya-mixed-iq\qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf`
  - SHA-256: `17ef7767533efd039d13594603bb0d9a787965759ec653b42f9221b50318e4d7`
  - 27.32B params (`IQ2_S` 14.261B, `Q4_K` 5.536B, `Q6_K` 3.397B, `IQ3_S` 2.852B, `Q5_K` 1.248B, `Q8_0` 0.024B, `F32` 0.003B)
- **実行環境**: Windows 11 x64 / RTX 5060 Ti 16GB (SM120) / 32GB RAM / CUDA 13.5 (`H:\Strata\cuda-13.5`, `nvcc 13.5.37`)
- **隔離ワークスペース・ビルド・計測補助**:
  - 隔離 worktree: `H:\Strata\worktrees\q38-cuda135-wmma-1f393b0`
  - クリーン `main` ビルド: `H:\Strata\builds\q38-135-main-check`
  - 参考ビルド / 旧計測ビルド: `H:\Strata\builds\q38-135-reference-1f393b0`, `H:\Strata\builds\q38-135-candidate-1f393b0`
  - 独立 `llama.cpp` 参照バイナリ: `C:\Users\downl\.hermes\llama\prism-b10826-c9c34bc\bin\` (`llama-bench.exe`, `llama-tokenize.exe`, `llama-completion.exe`)
  - 計測・資格判定補助スクリプト: `H:\Strata\stage\strata-q38-cuda135\tools\` (`bench_strata.py`, `compare_logits.py`, `qualification.py`)

---

## 1. 今回のコミットで `main` に反映した変更一覧

| ファイル | 種別 | 主な変更内容 |
| :--- | :--- | :--- |
| `src/qwen38/CMakeLists.txt` | 変更 | `STRATA_Q38_REQUIRE_CUDA135` (`OFF`), `STRATA_Q38_EXPERIMENTAL_FUSED_WMMA` (`OFF`), `q38-fused-wmma-test` (`fused_wmma`) 追加 |
| `src/qwen38/iq_tables.hpp` | 変更 | `Q38_IQ_TABLE` を `__device__ __constant__` から `__device__ const` (`__ldg`) へ変更し定数キャッシュのワープ直列化を解消 |
| `src/qwen38/ops.cuh`, `src/qwen38/ops.cu` | 変更 | FP32 `Q::matvec` の共有メモリ + `uint16_t` コアレスドロード化（`atol=2e-4, rtol=2e-5` 維持）。`BlockQ8_1`, `Q::quantize_q8_1`, `Q::matvec_q8_1`（`__dp4a` パック内積: `Q4_K`/`Q5_K`/`Q6_K`/`IQ2_S`/`IQ3_S`）、および共有 LUT `sgrid[1024]` + unroll-2 の `mv_q8_iq2s_8rows` を実装 |
| `src/qwen38/prefill_gemm.cuh`, `src/qwen38/prefill_gemm.cu` | 変更 | `dequant_tile_kernel`（types `12,13,14,21,22`）の `uint16_t` 共有メモリ・コアレスドロード化、およびオプトイン `matmul_fused_wmma_fp16` / `fused_wmma_calls` 統計追加 |
| `src/qwen38/prefill_wmma.cuh`, `src/qwen38/prefill_wmma_layout.hpp`, `src/qwen38/prefill_wmma_test.cu` | 新規 | 実験用オプトイン fused WMMA テスト・レイアウト定義（通常デフォルトは無効、Stage 2 単体 83 ケース PASS） |
| `src/qwen38/engine.cu` | 変更 | `kPrefillChunk = 512`, `kPrefillTileBytes = 64 MiB`, `xq8` + `rms_h` + `h_q8_valid` による同一入力 `quantize_q8_1(h)` 共有化、環境変数 `STRATA_Q38_PROFILE` による詳細タイミング計測を追加 |
| `src/qwen38/kv.cu` | 変更 | `spin8_store_q_batch_kernel`, `BT=4 / HG=3` の `spin8_attend_batch_kernel`, Split-KV Decode (`spin8_attend_decode_split_kernel` + `spin8_attend_decode_reduce_kernel`), 融合 NaN 検査、CTA 内 warp 均一 `early_stop` ブロードキャスト、および `KvCache::enqueue()` で `conservative=true/false` 間の bit-exact 同一カーネル実行を保証 |
| `src/qwen38/kv_numerics.cu` | 変更 | `spin8_batch_and_split` テスト（チャンク `{1,2,3,4,5,63,64,65,127}` + Split-KV Decode + NaN 拒否 + reset 再利用）を追加 |

---

## 2. 検証済みテスト・Sanitizer・実測ベンチマーク

### 2.1 `main` ソースからのビルド & 単体テスト（最新）
- **ビルドコマンド**:
  - `cmake -S C:\Users\downl\Documents\ChatGPT\Strata\src\qwen38 -B H:\Strata\builds\q38-135-main-check -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=H:/Strata/cuda-13.5/bin/nvcc.exe`
  - `cmake --build H:\Strata\builds\q38-135-main-check --parallel 5` → 全 50 ターゲット成功（終了コード 0）
- **`ctest --test-dir H:\Strata\builds\q38-135-main-check --output-on-failure`**（順次実行、最終コード確認）:
  - **17/17 PASSED (100%, 13.87 sec)**: `input_reader`, `language_cpu`, `protocol`, `numerics`, `iq_cpu`, `iq_gpu`, `reader_regression`, `geometry_regression`, `family_regression`, `reader_metadata`, `kv_numerics`, `kv_cpu`, `hardware`, `prefill_gemm`, `fused_wmma`, `spin8_triality`, `dialect_cuda_regression`
- **CUDA Sanitizer (`q38-kv-numerics.exe`, 最終 `kv.cu` 修正後の `main` ビルド)**:
  - `compute-sanitizer --tool synccheck`: **0 errors**
  - `compute-sanitizer --tool memcheck`: **0 errors**
  - 参考（`kv.cu` の `early_stop` / `!conservative_` 4 行調整直前の `q38-135-reference-1f393b0\q38-kv-numerics.exe`）: `racecheck` **0 hazards displayed (0 errors, 0 warnings)**（`q38-135-main-check\q38-kv-numerics.exe` での再 `racecheck` は時間都合上未実施のため必要なら再実行）

### 2.2 実モデルベンチマーク（`--kv spin8_k8v4 --spec-draft-n-max 0`, 非プロファイル計測）
> 注: 以下は `kv.cu` 最終 4 行調整直前（同一 `BT=4` + `Split-KV` + `sgrid` + `dp4a` カーネル、`H:\Strata\builds\q38-135-reference-1f393b0\strata-q38.exe`）での `bench_strata.py` 計測結果です。`main` 最終コミット後の `strata-q38.exe` での再取得・`llama.cpp` との teacher-forced 資格比較は未実施（残課題）です。

| 構成・プロンプト長 | Prefill 速度 (中央値) | Decode 128-step (中央値 / 最速) | 記録ファイル |
| :--- | :--- | :--- | :--- |
| `llama-bench.exe` (`p=140`) | **715.02 tok/s** (196.68 ms) | **33.55 tok/s** (3816.18 ms) | 独立参照基準 (90%目標: 643.5 / 30.2) |
| 初期 `1f393b0` Reference (`p=140`) | 72.67 tok/s | 5.23 step/s | `H:\Strata\q38-135-reference-140.json` |
| **`strata-q38` (`p=140`)** | **196.94 tok/s** (710.87 ms) | **23.26 step/s** (5502.36 ms / 最速 23.86) | `H:\Strata\q38-135-reference-140-sgrid.json` |
| `llama-bench.exe` (`p=1024`) | **943.04 tok/s** (1085.85 ms) | — | 独立参照基準 (90%目標: 848.7) |
| **`strata-q38` (`p=1024`)** | **354.49 tok/s** (2888.62 ms) | **23.63 step/s** (5416.24 ms / 最速 23.70) | `H:\Strata\q38-135-candidate-1024-bt4.json` |
| `llama-bench.exe` (`p=3072`) | **932.11 tok/s** (3295.74 ms) | — | 独立参照基準 (90%目標: 838.9) |
| **`strata-q38` (`p=3072`)** | **288.83 tok/s** (10635.84 ms) | **22.47 step/s** (5697.39 ms / 最速 22.62) | `H:\Strata\q38-135-candidate-3072-bt4-r2.json` |

### 2.3 プロファイル内訳・精度・長文 Admission の観測事実
- **Prefill 内訳 (`STRATA_Q38_PROFILE=1`, `n=512` chunk 2 @ 1024 prompt, 同期直列計測参考値)**:
  `ffn=864.79 ms`, `gdn_mm=251.80 ms`, `attn_mm=82.24 ms` (`mm_batch` 合計 **1198.8 ms ≒ 79%**), `attn_kv=233.72 ms`, `gdn_core=72.91 ms`, `fence=10.01 ms`。
- **Decode 内訳 (`STRATA_Q38_PROFILE=1`, `pos=1025`)**:
  `gdn=26.75 ms`, `ffn=6.13 ms`, `fence=4.35 ms` (`LayerFence(8)` WDDM `cudaMemGetInfo` + `GlobalMemoryStatusEx`), `attn_mm=3.42 ms`, `head=2.35 ms` (`442 GB/s`), `kv=2.15 ms` (Split-KV 導入前 `6.35 ms` から短縮)。
- **Logits 比較 (`dp4a + Q8_1` vs 初期 `1f393b0` Reference, `--kv f16`, `p=140` / 9行×248,320語彙)**:
  - `H:\Strata\q38-135-dp4a-logit-comparison.json`: `max_abs_error=0.2740`, `normalised_rmse=0.01057`, `min_cosine=0.999881`, `kl_mean=0.000430`, `kl_max=0.001040`, `top1_agreement=1.0`。
- **長文 Context Admission (`spin8_k8v4`, `--tokens 100,101`)**:
  - `ctx=65536`: PASS (`required_with_reserve=14586769900`, `available=15906897920`, `reserve=805306368`, `scratch_bytes=458650092`)
  - `ctx=98304`: PASS (`required_with_reserve=15463379436`, `available=15906897920`, `scratch_bytes=461795820`)
  - `ctx=131072`: FAIL (`required=16337891820` > `available=15906897920`, 約 411 MiB 超過)
- **補足事象**:
  - `H:\Strata\q38-135-candidate-3072-bt4.json`（初回試行）は一時的なホスト RAM 残量低下により重みロード中に `requires available RAM >= 3 GiB and commit >= 4 GiB` で安全停止（仕様通りの動作）。ホスト RAM `7.67 GiB` / Commit `21.26 GiB` 確認後の再試行（`-r2.json`）で正常完走。

---

## 3. 次担当への優先タスクリスト（90% 目標・完全検証まで）

1. **Prefill `mm_batch`（全時間の約 79%）の分解・高速化（最優先）**
   - `src/qwen38/prefill_gemm.cu` の `matmul_batch` 内を `STRATA_Q38_PROFILE=1` + 非同期 `cudaEvent` で分解計測し、`f32_to_fp16` / `dequant_rows_fp16` / `cublasGemmEx` / ストリーム待ち、および行列形状・量子化種別（`IQ2_S`, `IQ3_S`, `Q4_K`, `Q5_K`, `Q6_K`, `17408x5120`, `5120x17408` 等）ごとの実時間を特定する。
   - 支配的な箇所に応じて `dequant_tile_kernel` の multi-row CTA + `sgrid` 共有 LUT + `half2` ストア化、入力キャッシュによる同一 `bh` 変換回数の削減、および（精度ゲート付きで）INT8 Tensor Core / 量子化融合 GEMM（MMQ 方式、FP8/NVFP4 は不使用）を検討する。
   - **KV Batch Attention 比較**: `spin8_attend_batch_kernel` で `BT=2, HG=6`（GQA-6 共有で K/V ロード半減・レジスタ数維持）と `BT=4, HG=3` / `BT=1` を比較検証する。

2. **Decode の 30+ step/s 到達**
   - `gdn=26.75 ms`・`ffn=6.13 ms` のうち `IQ3_S` / `Q4_K` / `Q6_K` GEMV への共有 LUT・2ブロック先読み ILP 適用と、`ffn_gate` + `ffn_up` 等の融合、残存 KV カーネル起動削減を進める（`head` `output.weight` は 2.35 ms / 442 GB/s で帯域上限に近いため変更不要）。

3. **65K / 96K / 131K 実長文 & 131072 収容対応**
   - `engine.cu` で `3 GiB RAM / 4 GiB Commit / 768 MiB GPU_RESERVE / LayerFence(8)` 等の安全フロアは**一切下げず**、KV 予算確保後の残余 VRAM に応じて `kPrefillChunk` / `kPrefillTileBytes` を適応縮小することで `ctx=131072` を収容し、`65536` / `98304` / `131072` の実長文推論・状態・STOP・ピークメモリを検証する。

4. **独立 `llama.cpp` との自然文（日/英/コード）teacher-forced 比較 & 最終資格確認**
   - `llama-tokenize.exe` で日本語・英語・コード（チャンク境界 31/32/33, 63/64/65, 127/128/129/140 を含む）をトークン化し、`strata-q38`（`--dump-full-logits`）と `llama.cpp` の teacher-forced logits 比較（`compare_logits.py` + `qualification.py`）を実施する。
   - リリース / 資格確認用には `q38-kv-numerics.exe` に対する `racecheck` 再実行および `main` ビルド `strata-q38.exe` でのベンチ再取得を行う。

---

## 4. 開発・運用上の注意点（ハマりポイント）

- **CUDA 13.5 & VS2022 環境**: `H:\Strata\cuda-13.5` を使用するため、`vcvars64.bat` 後に `$env:PATH='H:\Strata\cuda-13.5\bin;' + $env:PATH` と `-DCMAKE_CUDA_COMPILER='H:/Strata/cuda-13.5/bin/nvcc.exe'` を明示すること。並列ビルド上限は `--parallel 5`。
- **Python**: サンドボックス内では Windows Store エイリアス回避のため `C:\Users\downl\AppData\Local\Programs\Python\Python312\python.exe` を直接指定すること。
- **GPU バイナリの並列実行禁止**: `cudaMemGetInfo` ベースの予約・アカウント検査があるため、`ctest` は並列化せず 1 つずつ実行すること（別プロセスとの同時 GPU 実行禁止）。
- **空き容量フロア**: `resources.hpp` 等により `C:/ >= 5 GiB`・`H:/ >= 10 GiB` が要求される（確認時点: `C:/ 6.32 GiB`, `H:/ 18.58 GiB`）。
- **変更・出力の非資格表記**: 現行の高速化は単体・Sanitizer・`1f393b0` 対比で検証済みだが、`llama.cpp` 対比による `releaseQualified=true` 判定は未達のため、既存 `AGENTS.md` / `releaseQualified=false` 方針を維持すること。
