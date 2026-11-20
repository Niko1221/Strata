# QSA architecture findings from the reference implementation (2026-11-20)

Repository: `kuraneko1/Qwen3.8-27B-QSA-llama.cpp`, cloned to
`C:\Users\downl\AppData\Local\Temp\opencode\qsa-llamacpp` (HEAD
`6c38fd0727da42233bca53c0394165cc8405870d`). The commit ID supplied by the
user (`864934c...`) is not present in this repository (fetch failed), so the
current HEAD was analysed instead.

## What QSA is

The reference fork implements **Qwen Sparse Attention** with a "lightning
indexer" (DeepSeek sparse-attention style): a small learned projector computes
query/key vectors for selecting the top-k KV blocks, and the full attention
then runs only on the selected sparse set. Relevant source:
- `src/llama-arch.cpp:289-294` — metadata keys `attention.indexer.head_count`,
  `key_length`, `top_k`, `block_size`, `local_blocks`, `types`.
- `src/llama-hparams.h:282-291` — `indexer_n_head`, `indexer_head_size`,
  `indexer_top_k`, `indexer_block_size`, `indexer_local_blocks`,
  `is_indexer_full_impl`.
- `src/llama-graph.cpp:1317` — `top_k` input to the sparse attention graph.
- `src/llama-context.cpp:331-337,1532,3449` — dense-vs-sparse planning,
  `QSA27B mode: sparse text / dense multimodal`, graph reuse limits.
- `src/llama-cparams.h:55` — `fused_lid` (fused lightning indexer).

## Tensor payload observed in the two HF checkpoints

- `Kiyo1130/Qwen3.8-27B-QSA-NVFP4`: arch qwen35, 64+1 layers, 24/4 heads,
  F32/BF16/NVFP4 (type 40), 1346 tensors; extra 400 `.scale` (type 0, [1])
  and 80 `.indexer.*` tensors per 16 attention layers
  (`q_proj [5120,512]`, `k_proj [5120,128]`, `q_norm/k_norm/k_norm.pre_weight [128]`).
  NVFP4 payload measured from offsets = 36 B per 64 values (scales inline).
  `--inspect` on the FP4-support build accepted this file.
- `higa2515/Qwen3.8-27B-QSA-GGUF` Q4_K_M: arch qwen35, 946 tensors,
  Q3_K (type 11) on `ffn_up`, Q4_K/Q6_K elsewhere; also indexer tensors.

## Current Hakua support status

- **Admission**: both checkpoints' tensor roles are now admitted; `.scale` and
  `.indexer.*` are treated as ignored auxiliary tensors with extent checks.
- **Formats**: type 40 dequant (E2M1 + inline E8M0/UE4M3 scales) and type 11
  dequant (Q3_K, ported from `src/quant/dequant.cpp`) were added to the
  dequantize-to-F16 prefill path.
- **QSA semantics**: NOT implemented. The indexer projections and top-k sparse
  KV selection are ignored, so this engine runs dense attention. Outputs will
  not be equivalent to the QSA reference fork. Model quality is unqualified.

## What full QSA support would require

1. Load the indexer projections (q/k, norms) from the extra tensors.
2. Compute indexer logits, top-k (with `block_size` / `local_blocks` rules) for
   each query token, matching the reference selection exactly.
3. Gather selected KV blocks into the attention kernel, preserving the
   dense-prefill / sparse-decode planning boundary.
4. Validate logits against the reference implementation on a fixed prompt.

This is a new attention family, not a format fix. It should be gated behind
an explicit `qwen38_dense_qsa` profile and kept opt-in until numerically
validated.

## Verified acceptance (both HF checkpoints)

Header-only sparse files were created from the remote GGUF headers (HTTP Range)
plus each pinned total size, allowing CPU `--inspect` without downloading the
weights. With the final QSA-support build (`q38-9b-fp4-20261120`, CTest 18/18):

- `Kiyo1130/Qwen3.8-27B-QSA-NVFP4`: accepted — `profile=q38-27b layers=64
  tensors=1346 hidden=5120 ffn=17408 vocab=248320 q_heads=24 kv_heads=4
  nextn=1 mtp_tensors=15 gpu_weight_bytes=16331667456 host_embedding_bytes=2542796800`.
- `higa2515/Qwen3.8-27B-QSA-GGUF` Q4_K_M: accepted — `tensors=946
  gpu_weight_bytes=15387273216 host_embedding_bytes=715161600`, same geometry.
- Regression on the existing 9B fixture: token `15217` unchanged.

Additional type support added for the QSA Q4_K_M quant mixture:
type 11 Q3_K (ported from `src/quant/dequant.cpp`) and type 20 IQ4_NL
(GGML 16-value codebook), both dequantized to FP16 before the cuBLAS GEMM.

## Note on VRAM

QSA-NVFP4 `--inspect` reported 16,331,667,456 GPU weight bytes (~15.2 GiB)
plus 768 MiB reserve and scratch: this checkpoint does not fit a 16 GB GPU
without offload or further quantization, regardless of QSA support.
