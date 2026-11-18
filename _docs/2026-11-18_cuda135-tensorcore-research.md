# CUDA 13.5 / Tensor Core / Qwen3.8 アーキテクチャ調査 (2026-11-18)

一次資料:
- NVIDIA Technical Blog, "NVIDIA CUDA 13.5 Brings NVIDIA Vera Rubin GA Support" (2026-11-04)
- NVIDIA, "CUDA Toolkit 13.5 Release Notes" (cuBLAS / cuBLASLt の項目)
- Qwen Team, "On the Design of Qwen3.8-Next Architecture", arXiv:2608.30320v1 (2026-08-31)

## CUDA 13.5 で Q38 に直結する点

1. **cudaMemcpyBatchAsync / CUDA Graph 内のバッチ 1D コピー**
   13.5 で CUDA Graph キャプチャに対応。同一バッチ内のコピーは順序保証なし
   (独立先なら安全)。`run_prefill_chunk` の 140 回の Embedding H2D を
   1 回のバッチコピーに集約できる。ただし現在の `embrow` は 1 行分
   (`emb_stride`) しかなく、複数行分のデバイスバッファ追加と admission
   の再計算が必要。効果は起動オーバーヘッド程度 (想定 1〜3 ms/140 tokens)
   で、p=140 の 76 ms 差を単独で埋めるものではない。
2. **TMA out-of-bounds をカーネル内で設定 (`tensormap.replace`)**
   ragged tile の Illegal Address 回避 Workaround を除去できる。
   実験的 WMMA 経路 (32 token tiles) を本格化する際の前提。
3. **cuBLASLt の既知の正确性問題**
   13.5 リリースノートは Tensor Memory と並行する algo config ID 66 の
   誤結果を修正。Q38 は 20 MiB の L2 persisting window を使うため、
   cuBLASLt に明示 algo を渡す際は 66 を除外するのが安全。
   既存 FP8 パスの `get_lt_plan` を FP16 にも広げる候補。
4. **CCCL 3.5 の Batched Top-K / deferred-size reduction**
   QSA (K=2048, r=4) のトークン選択を GPU 内で完結させる将来候補。
   現行の Spin8 KV は密な K8V4 なので、まずは密な経路を仕上げる。
5. **nvprune --precompile / Nsight Perf SDK**
   sm_120 向け SASS 事前コンパイルと常時プロファイリング。初回起動の
   JIT と間欠的な性能劣化の追跡に使える。

## Blackwell / SM120 Tensor Core の活用状況

- Q38 prefill は cuBLAS `CUBLAS_COMPUTE_16F` + FP16 一時出力 +
  1D Vec8 アンスケール。FP16 入力の Tensor Core 経路は利用済み。
- 実験的 fused WMMA (`STRATA_Q38_EXPERIMENTAL_FUSED_WMMA`) は
  32 token × 64 row tiles で、同じ重みをトークンタイルごとに再復号する。
  有効化するだけでは cuBLAS 経路を超えられない (実測 3/10 pass、
  性能も非改善)。トークンタイル内での重み再利用とパイプライン化が必要。
- FP8 経路 (`STRATA_FP8_PREFILL=1`) は実測 1737 ms / 140 tokens で
  通常経路 (272 ms) より大幅に遅く、現状は非採用。

## Qwen3.8 アーキテクチャから導かれる Q38 の方向

一次論文の構成: GDN と global attention を 3:1 で hybrid、
QSA は r=4 圧縮・K=2048 トークン予算、Gated Residual 4 ブランチ、
n-gram embedding を host メモリへ prefetch。GDN カーネルは
FlashQLA で forward 2-3x、backward 約 2x (Triton 比)。

Q38 の実装は GDN hybrid + Spin8 KV + MTP で方向は一致している。
未実装は QSA / GR / n-gram embedding だが、これらはモデル形式と
重みレイアウトの変更を伴うため、現行 27B GGUF の互換性を保つ範囲では
GDN 入力射影の結合 (小さな GEMM の削減) と Spin8 attention の
タイル単位 softmax 更新が、安全に試せる次の一手。

## 結論

CUDA 13.5 の新機能のうち、今すぐ効果が見込めるのはバッチコピーと
TMA OOB の 2 点。p=140 で llama.cpp f16 に追いつくには、
GDN の 672 回の GEMM タイル呼び出し (全体 2592 回) を減らす
構造変更が主因であり、ドライバ機能だけでは解消しない。
