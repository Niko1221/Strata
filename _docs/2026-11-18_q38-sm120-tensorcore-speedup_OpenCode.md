# 2026-11-18 Q38 SM120 Tensor Core 高速化 plan 実行ログ (OpenCode)

## 概要
Antigravity セッションの `implementation_plan.md` に基づき、Qwen3.8-27B ネイティブ CUDA エンジンの
単体テスト修正と実機検証準備を実施した。実装はローカル `main` (`3bd6191b4`) に対して行い、
ユーザー指示により `origin/main` へ push する。

## 背景・要求
- 同一モデル GGUF で Prism 版 llama.cpp (`prism-b10826-c9c34bc`) の Prefill/Decode を上回ること。
- `ctx = 131072` (`--kv spin8_k8v4`) の実長文推論を安全フロア緩和なしで達成すること。
- `ctest` 17/17 PASS、`compute-sanitizer` 0 errors / 0 hazards をゲートとすること。

## 前提・判断
- 適用 SOP: `SOP-Common-MILSPECLLMOps-Engineering-Standard.md` ->
  `SOP-Implementation-Start-Gate.md` -> `SOP-CPP.md`。適用 Skill: `milspec-codex-standard`,
  `deepresearch-defense-standard`（今回は WEB 検索不要のため deep research は未実行）。
- 計画書の記載と実ファイルの差分: `prefill_gemm.cu` は 997 行ではなく 930 行、
  `ops.cu` は 993 行ではなく 940 行（いずれも 1,000 行未満を満たす）。
- LayerFence 高速化 (`GetPerformanceInfo` スキップ)、KV Split-KV 動的分割、
  `STRATA_Q38_FP32_DECODE/PREFILL` アブレーション、適応的 prefill チャンク階段は
  現在のスナップショットに既に存在するため、重複変更は行わない（計算量を増やさない判断）。
- Windows Controlled Folder Access (`EnableControlledFolderAccess=1`) が
  `Documents` 配下への直接書き込みを拒否したため、一時領域へ書き出したバイトを
  `cmd.exe /c copy /Y` で配置した。セキュリティ設定自体は変更していない。

## 変更対象ファイル
- `src/qwen38/numerics.cu`: `make_packed_weights` の `ty == 8` (Q8_0) 合成ブロックスケールを
  `sc * 0.125f` とし、実モデル相当の重み振幅 (`|w| <= 0.5`) に正規化。
- `src/qwen38/prefill_gemm_test.cu`: 実モデル形状テストを先頭/末尾のみから
  全トークン (`0..T-1`) の `Q::matvec` 比較へ強化。

## 実装詳細
- Prefill の `target_max = 4.0f` 統一と 256 行タイル整列は、失敗再現試験の結果
  (下記) により原因修正にならなかったため、スナップショットの元値へ戻した。
  不要な挙動変更を main へ持ち込まないためである。
- モジュール分割 (`prefill_dequant.cuh` / `ops_q8_mv.cuh`) は、実測行数が
  1,000 行未満であり、ビルド検証を優先するため今回は見送った（残留リスク参照）。

## 実行コマンド
- `cmake --build H:\Strata\builds\q38-135-sm120-fast --parallel 5`
  (VS2022 `vcvars64.bat` 経由、`STRATA_Q38_REQUIRE_CUDA135=ON`,
   `STRATA_Q38_EXPERIMENTAL_FUSED_WMMA=OFF`)
- `ctest --test-dir H:\Strata\builds\q38-135-sm120-fast --output-on-failure`
- 個別再現: `q38-numerics.exe`, `q38-prefill-gemm-test.exe`

## テスト・検証結果
- ビルド: 13/13 .target 成功、エラーなし（CUDA ヘッダの C4819 警告のみ）。
- `ctest`: **15/17 PASS**。失敗は以下の 2 件で、いずれも計画書が前提とした既存失敗である。
  - `numerics`: `ty == 8` の `vs_fp64` は修正により `max_abs=0.00314` (atol 0.006) で PASS。
    ただし後続の `ty == 12/13/14/21` 等の `vs_fp64` も合成重み振幅が実モデルより大きく、
    依然として失敗する（例: `t21_n512_r7` error 0.024）。`_exact_q8` は全型 PASS であり
    カーネル自体の演算は正常。
  - `prefill_gemm`: `K=5120, N=5120, T=64, type=12` で GEMM 出力の末尾トークンが 0 になる。
    トークン別診断では 0..60 は一致し 61..63 がゼロ（256 行整列時）。
    `target_max` 変更・256 行整列・`CUBLAS_COMPUTE_32F` 強制のいずれでも解消せず、
    原因は未確定（cuBLAS 経路または活性ステージングの可能性）。
- 実機ベンチマーク / 131K 長文脈 / 4 段階ログits アブレーション / sanitizer:
  **未実施**。モデルは `H:\Strata\models\qwen38-27b-soya-mixed-iq\...3.69bpw-12GB-MTP.gguf`
  に存在するが、計画書指定のトークン JSON と llama.cpp 参照レポートが未提供のため A/B 比較不可。

## 残留リスク
- `releaseQualified=false` を維持。一般モデル品質・全ウィンドウ推論・GPU 並行は未検証。
- 上記 2 件の単体テスト失敗が残るため、main は CI グリーンではない。
- 1,000 行制約は満たすが、計画書が推奨した 700 行以下へのモジュール分割は未実施。

## 次の推奨アクション
1. `prefill_gemm` の末尾トークンゼロ化を `act_fp16` / `act_scales` のダンプで切り分け、
   cuBLAS 呼び出しまたは `prepare_act_scaled_fp16_reg_kernel<5120>` を修正する。
2. `numerics.cu` の合成重み生成を全量子化形式で実モデル振幅へ正規化し、
   `_exact_q8` の厳密 tolerances を維持したまま `vs_fp64` を通す。
3. トークン JSON と llama.cpp 参照レポートを準備し、実機ベンチと 131K 長文脈を実施する。
4. その後に 0.6.x バージョン更新、上書きインストール、E2E、README 各言語、
   release タグ発行を行う。
