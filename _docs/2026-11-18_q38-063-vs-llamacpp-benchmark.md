# 2026-11-18 Qwen3.8-27B ネイティブ CUDA vs llama.cpp 同日比較 (0.6.3)

## 条件
- GPU: NVIDIA GeForce RTX 5060 Ti 16 GB (SM120), ドライバ 621.14
- モデル: `qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf`
  SHA-256 `17ef7767533efd039d13594603bb0d9a787965759ec653b42f9221b50318e4d7`
- ネイティブ: `strata-q38.exe` 0.6.3 (`strata-q38 own-native 0.6.3 CUDA13 SM120`)、
  `--kv spin8_k8v4`、`--spec-draft-n-max 0`、`--context 4096/8192`、
  `bench_strata.py --new-tokens 129 --warmups 1 --repeats 5`、中央値を採用。
- llama.cpp: Prism `prism-b10826-c9c34bc` の `llama-bench.exe`、`-ngl 99 -r 5`、
  (A) 既定 KV `f16`、(B) KV `q8_0`/`q4_0` + `--flash-attn on`。
- トークン列: p=140 は `bench140.ids.json`、p=1024/3072 は同列を循環させた合成列
  （自然文ではないため絶対値は参考値、A/B 比較には使用可）。

## 実測 (prefill, 中央値)

| 入力長 | ネイティブ spin8_k8v4 | llama.cpp (A) f16 | llama.cpp (B) q8_0/q4_0+FA |
| ---: | ---: | ---: | ---: |
| 140 | **514.17 tok/s** (272.28 ms) | 671.36 tok/s (209.76 ms) | 378.68 tok/s (371.84 ms) |
| 1024 | **915.31 tok/s** (1118.74 ms) | 528.77 tok/s (2248.75 ms) | 586.80 tok/s (1748.65 ms) |
| 3072 | **770.99 tok/s** (3984.50 ms) | 594.03 tok/s (6604.40 ms) | 744.67 tok/s (4125.78 ms) |

## Decode (p=140, 128 steps)

| 実装 | ms/step | tok/s |
| --- | ---: | ---: |
| ネイティブ spin8_k8v4 | 32.93 | 30.37 |
| llama.cpp (A) f16 | 30.38 | 32.92 |

## 判定
- 同一の量子化 KV 契約 (native `spin8_k8v4` ≒ llama `q8_0`/`q4_0`) では、
  ネイティブは p=140 で +36%、p=1024 で +56%、p=3072 で +3% 上回る。
- 既定 f16 KV の llama.cpp に対しては p=140 で -23% だが、
  p=1024 で +73%、p=3072 で +30% 上回る。
- Decode はほぼ同等（ネイティブ -8%）。

## 補足
- 計画書の llama.cpp 基準値 (p=140 715、p=1024 943、p=3072 932 tok/s) は
  今回の同日ビルドでは再現せず、フラグ/KV 形式に強く依存する。
- `STRATA_Q38_PROFILE=1` のコールド計測では `gdn_mm` が 490/737 ms を占め、
  GDN 入力射影の結合が残る最適化候補。FP8 prefill 経路は 1737 ms で大幅に遅い。
- すべての計測で `qualified=false`（同一モデルの数列一致・full-window・GPU 並行は未検証）。
