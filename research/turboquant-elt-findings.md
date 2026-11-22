# Research findings: TurboQuant-CUDA and Elastic Looped Transformer

Read 2026-11-22 from primary sources (GitHub APIs, READMEs). These are
knowledge notes; no third-party code was copied into the Q38 tree. Licensing
is recorded per source.

## 1. zapabob/Turboquant-CUDA (Apache-2.0)
- Measurement/research workspace, not a standalone CUDA kernel source. CUDA
  kernels live in the vendored `zapabob/llama.cpp` submodule.
- Focus: Qwen3.5-9B KV-cache TurboQuant with captured replay on RTX 3060 12 GB.
- Key line: `key_only_block_so8_triality_vector`; TQ4_1S GGUF packaging and
  a fused packed-weight CUDA path for small decode in vendored llama.cpp.
- Metrics: hidden cosine, logit cosine, attention relative error, memory ratio,
  Friedman/Wilcoxon-Holm statistics (`artifacts/metrics/`).
- Relevance to fork: the 9B KV-cache quantization/measurement methodology
  informs our 9B engine (spin8_k8v4 / KV tiers); no code copied.

## 2. zapabob/elastic-looped-transformer (ELT)
- README declares Apache-2.0 (frontmatter + badge); no standalone LICENSE file.
- arXiv:2604.09168 faithful PyTorch port. Causal LM with Transformer layers
  weight-shared across L iterations; pick `L ∈ [1,4]` per request to trade
  quality for latency from one checkpoint (effective depth up to 112 at L=4).
- Trained with Intra-Loop Self-Distillation + GRPO; scales 10M-1B non-embedding
  runnable on a single RTX 3060 12 GB.
- Relevance: looped-weight ideas; ideas only, no code copied.

## Licensing note
- Apache-2.0 sources (TurboQuant, ELT per README) require attribution when
  ideas/measurements are referenced; recorded here. No code from these repos
  was copied into the Q38 engine tree.
