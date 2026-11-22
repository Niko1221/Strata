# Research findings: Inkling-2, TurboQuant-CUDA, Elastic Looped Transformer

Read 2026-11-22 from primary sources (GitHub APIs, READMEs, Hugging Face GGUF
README). These are 知見ドキュメント; no third-party code was copied into the
Q38 tree. Licensing is recorded per source.

## 1. Thinking Machines Inkling-2 Mini (GGUF + separate DLL)
- Original: `thinkingmachines/Inkling-2-Mini` (Apache-2.0 + Model Acceptable Use Policy).
- GGUF ports: `ggml-org/Inkling-2-Mini-GGUF` (IQ4_NL 17.6 GB, Q8_0 27.8 GB,
  BF16 52.2 GB, mmproj 642/346 MB), also bartowski/unsloth/InterpreticAI.
- GGUF header (read directly, see H:/Strata/evidence/inkling-gguf-header.txt):
  `magic=GGUF, version=3, tensors=704, kv=63`,
  `general.architecture = inkling2_loop`.
- Architecture: looped MoE, 26.4B total / 3.4B active (5.4B effective),
  24 layers where layers 6-17 form a loop core run 3x (48 evals/token),
  128 experts with 8 active per token. Original weights store routed experts
  in NVFP4 (layers 0-19) and MXFP8 (20-23); loop controller F32.
- Engine: `zapabob/yasakani` (Apache-2.0) ships `yskn_inkling2.dll` with a C
  ABI (`ink2_step`/`ink2_feed`, INK2_ABI_VERSION 3, packed NVFP4/MXFP8 dense
  support). DLL SHA-256 pinned in BUILDING.md and verified:
  `b10248cddc7b2bceb9a26d0d28b9446db57bf71c24db4772a3829f8e66b1c078`.
- Hardware: 24 GB+ NVIDIA GPU required; cannot run on this 32GB/16GB PC.
- Fork integration: DLL bundled under `app/desktop/src-tauri/resources/inkling2/`
  with LICENSE/NOTICE/THIRD_PARTY_NOTICES (docs/third-party/inkling2/);
  `model_family.hpp` recognises `inkling2_loop` for metadata/inspect only;
  generation stays in the separate DLL.

## 2. zapabob/Turboquant-CUDA (Apache-2.0)
- Measurement/research workspace, not a standalone CUDA kernel source. CUDA
  kernels live in the vendored `zapabob/llama.cpp` submodule.
- Focus: Qwen3.5-9B KV-cache TurboQuant with captured replay on RTX 3060 12 GB.
- Key line: `key_only_block_so8_triality_vector`; TQ4_1S GGUF packaging and
  a fused packed-weight CUDA path for small decode in vendored llama.cpp.
- Metrics: hidden cosine, logit cosine, attention relative error, memory ratio,
  Friedman/Wilcoxon-Holm statistics (`artifacts/metrics/`).
- Relevance to fork: the 9B KV-cache quantization/measurement methodology
  informs our 9B engine (spin8_k8v4 / KV tiers); no code copied.

## 3. zapabob/elastic-looped-transformer (ELT)
- README declares Apache-2.0 (frontmatter + badge); no standalone LICENSE file.
- arXiv:2604.09168 faithful PyTorch port. Causal LM with Transformer layers
  weight-shared across L iterations; pick `L ∈ [1,4]` per request to trade
  quality for latency from one checkpoint (effective depth up to 112 at L=4).
- Trained with Intra-Loop Self-Distillation + GRPO; scales 10M-1B non-embedding
  runnable on a single RTX 3060 12 GB.
- Relevance to Inkling-2: both share the looped-weight idea (Inkling-2 loops
  layers 6-17 three times; ELT makes L selectable per request). Ideas only,
  no code copied.

## Licensing note
- Apache-2.0 sources (yasakani, TurboQuant, ELT per README) require NOTICE and
  attribution; kept in docs/third-party/inkling2/ and referenced here. No code
  from these repos was copied into the Q38 engine tree.
