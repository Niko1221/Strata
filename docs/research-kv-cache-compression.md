# KV Cache Compression Research — Strata Engine
*Compiled 2026-10-11. Sources: arXiv papers (linked), NVIDIA/AMD docs, llama.cpp/ik_llama.cpp/vLLM docs, Strata source.*

## 0. Where Strata is today

- Serving config (`strata-swift-iq3_xxs.json`): ctx 131072, **KV = q4_0**, kv-resident 32768.
- KV code in-tree:
  - `src/kernels/cuda/kv_q4.cu` — Q4_0 KV **with Fast Walsh–Hadamard rotation (fwht256)**, deterministic group max (PR #21 lineage).
  - `src/kernels/cuda/kv_stream.cu` — KV streaming formats: `kKvHybrid` (**K8V4**: int8 K codes + f16 scales + rotated q4_0 V), `kKvQ4`, `kKvInt8`.
  - `src/kernels/kv_q8_parity.cpp` — INT8 KV append/gather verified bitwise vs host reference (plan v0.3 P7).
  - `qsa_decode_attn.cu`, `kv_q8.hpp` (KV_Q8_GROUP per-token groups along head_dim).
- The bundled ggml fork already declares **TQ1_0 (34), TQ2_0 (35), MXFP4 (39), NVFP4 (40)** types — the TurboQuant and FP4 type IDs exist, unused for KV.
- Hardware: RTX 5080 = **SM120 consumer Blackwell** (gen-5 tensor cores, native FP4/FP8). CUDA 13.0 release builds.

## 1. Literature map (what actually works)

### Rotation-based (kills the outlier problem)
- **QuaRot** (arXiv 2404.00456): Hadamard rotations make 4-bit KV lossless; outliers are an artifact of the basis, not the data.
- **RotateKV** (2501.16383): outlier-aware adaptive rotations + **pre-RoPE grouped-head rotation** + attention-sink protection → **<0.3 PPL at 2-bit** (LLaMA-2-13B), 3.97× peak-memory cut.
- **OScaR** (2605.19660): canalized Hadamard rotation + omni-token L2 scaling; per-channel 2-bit K with a small FP16 residual flush block (R=128). Simpler than TurboQuant pipelines, strong CUDA kernels.
- **OptR** (2608.02691): output-aware rotation, consistently beats QuaRot/OSCAR at INT2, keeps paged-KV format.
- **TurboQuant** (2504.19874, Google, ICLR 2026): random/Hadamard rotation → coordinates are Beta-distributed → **Lloyd-Max scalar codebook per coordinate**, plus 1-bit QJL residual for unbiased inner products. **Quality-neutral at 3.5 bits, marginal at 2.5**; ~6× KV memory, up to 8× attention speedup (H100, 4-bit). Online, data-oblivious, no calibration. Shipped in vLLM; ggml types TQ1_0/TQ2_0.
- ik_llama.cpp ships `--k-cache-hadamard` / `--v-cache-hadamard` and reports Q4_0 KV already low-PPL, Q6_0 ≈ Q8_0.

### Distribution-matched quantization
- **KIVI** (2402.02750, ICML 2024): **K per-channel, V per-token** (keys have persistent channel outliers; values don't). 2-bit with FP16 residual window; 2.6× peak-memory cut.
- **KVQuant** (2401.18079, NeurIPS 2024): **pre-RoPE per-channel K quant**, **non-uniform nuqX** codebooks (k-means signposts weighted by Fisher sensitivity), dense-and-sparse outlier isolation (keep 1% outliers in FP16), Q-Norm. **<0.1 PPL at 3-bit**; 1M ctx on one A100.
- **SKVQ** (2405.06219): channel reorder + clipped dynamic quant + **sliding FP16 window for recent tokens** → 2-bit K / 1.5-bit V, 1M ctx on 80 GB.
- **Quantize What Counts** (2502.15075): spectral-gap theorems — **keys deserve more bits than values** (key-driven quantization).
- **MixKVQ** (2512.19206), **TaDA**, **MiniKV**, **PM-KVQ** (2505.18610): layer/head/token-adaptive mixed precision; reasoning models need more KV precision on CoT-critical tokens.
- **KVSink** (2508.04257): attention-sink tokens (first few) must stay high precision under quantization.
- **XQuant** (EMNLP 2025): <1.4 effective bits with grouped K/V + residual.
- **CommVQ / VQ-LLM** (2025): vector-quantization codebooks that commute with RoPE — 1–2 bit territory.
- Caveat paper: **Quantization Hurts Reasoning?** (2504.04823): 4-bit KV is safe, 3-bit is risky for small models; **pre-bias K quantization** (quantize before the K-projection bias) recovers ~7.8 pts at 3-bit.

### Native FP4 hardware formats
- **NVFP4** (NVIDIA, Blackwell gen-5 tensor cores): E2M1 4-bit values, **FP8 E4M3 scale per 16-element block + FP32 per-tensor scale** ≈ 4.5 b/elem. Dequant happens *inside* the tensor core — no dequant kernel. NVIDIA reports NVFP4 KV ≈ **50% smaller than FP8 KV with <1% accuracy loss, ~5% better than MXFP4 KV** (finer block + mantissa scale).
- **MXFP4** (OCP microscaling): E2M1 + power-of-two E8M0 scale per 32 elements ≈ 4.19 b/elem. Coarser scale dial → more quantization error (NVIDIA claims ~88% higher error than NVFP4's scaling).
- **ScaleSearchAttention** (2605.12464): runs **QKᵀ and PV matmuls directly in NVFP4, zero dequant**, KV stored 4.5-bit.
- **ReQAT** (2606.15682): full NVFP4 W4A4KV4 training; found **E1M2 FP4 layout beats E2M1 for KV** (lower loss) — a free accuracy lever where the hardware accepts it.
- SM120 (RTX 5080) specifics: native FP4 tensor cores exist, but kernels must use SM80-style `mma.sync` (no TMEM/UMMA — that's SM100); needs CUDA 13.0+; llama.cpp b8967+ dispatches native NVFP4 MMQ on sm_120. TensorRT-LLM/vLLM do **not** ship NVFP4 KV on SM120 yet — open lane.
- **FP8 KV caution** (vLLM blog 2026-04-22): on Hopper, FP8 attention accumulation past ~100K contraction dimension collapses long-context retrieval (needle 91%→13%); fixed with two-level accumulation. Blackwell/B200 unaffected — the 5080 is on the safe side.

### AMD RDNA4 (RX 9070 / R9700, gfx1200/1201)
- 2nd-gen AI accelerators: **native FP8 (E4M3 + E5M2) WMMA, INT8, INT4 tensor inputs, 4:2 structured sparsity** (2× sparse peak). FP8 matrix dense = 289 TFLOPS on RX 9070 (2× FP16 matrix).
- **No FP4 hardware** (confirmed negative — HotChips 2025 RDNA4 deck lists F16/F8/INT8/INT4 tensor types only). Any FP4 (MXFP4/NVFP4) on RDNA4 is software-emulated (dequant to FP16/INT8 WMMA).
- Ecosystem proof: `vllm-rocm-rdna4` fork runs gfx1201 with FP8/MXFP4/NVFP4 quant paths (MXFP4 linear kernel), FP8 KV included.
- Practical RDNA4 KV target: **FP8 E4M3 KV + FP8 WMMA attention** (2× compression, near-lossless); INT4-with-block-scales as the 4-bit option; FP4 only as a storage format with fast dequant.

## 2. Bits-per-byte ladder (KV, f16 = 16 bits baseline)

| Format | b/elem | Notes |
|---|---|---|
| q8_0 | 8.1 | near-lossless default (llama.cpp consensus) |
| FP8 E4M3 | 8.0 | near-lossless; relative step beats q8_0's absolute step on heavy-tailed KV; tensor-core attention on Blackwell/RDNA4 |
| q4_0 | 4.5 | Strata's current; safe with Hadamard rotation (already in kv_q4.cu) |
| IQ4_NL | 4.5 | non-linear grid, beats q4_0 on KV in llama.cpp tests |
| NVFP4 | 4.5 | native Blackwell; dequant-free attention; NVIDIA: <1% loss vs FP8 |
| TurboQuant 3.5 | 3.5 | paper-quality-neutral with outlier split |
| MXFP4 | 4.2 | coarser scale; fine for V, weaker for K |
| 2-bit (RotateKV/OScaR/SKVQ class) | 2.0–2.5 | needs rotation + per-channel K + sink/window protection |

## 3. Proposed ideas for Strata (ranked)

**Tier 1 — cheap, big quality-per-bit wins on the existing kernels**

1. **Sliding FP16 window + sink tokens** (SKVQ/KIVI/KVSink). Keep last ~128 tokens and first ~8 tokens in f16; quantize the bulk. At 2-bit this is the single largest quality lever in the literature and it's ~10 lines of bookkeeping in `kv_stream.cu`'s append path. Cost: ~1% of cache.
2. **Asymmetric K/V precision** (KIVI + "Key-Driven Quantization" theorem): K at q4_0/iq4_nl, V at q4_0 is fine, but the reverse is worse; K8V4 hybrid already exists — add **K4V2** and **K4V4-with-sinks** variants, and give K the rotation, V the per-token scaling (OScaR recipe: K = Hadamard + token-L2 + per-channel quant; V = offline Hadamard on weights + per-token quant).
3. **IQ4_NL KV** (4.5 bits, non-uniform grid) as a drop-in upgrade over q4_0 — Strata already has `iq_kernels.cu`; llama.cpp reports IQ4_NL beats q4_0 for KV at equal size.
4. **Per-channel K scales** (KIVI/KVQuant): Strata's KV_Q8_GROUP groups run along head_dim (per-token). Keys have persistent channel outliers → add a per-channel scale layout for K (scales amortized across tokens: 16 bits per channel per 128-token block ≈ 0.125 b/elem). Combine with the existing FWHT.

**Tier 2 — Blackwell-native formats (RTX 5080 / sm_120)**

5. **NVFP4 KV cache type**: store K/V as E2M1 + E4M3 scale per 16 (4.5 b/elem, same size as q4_0) and run **QKᵀ/PV in FP4 tensor cores with no dequant** (ScaleSearchAttention pattern). Same bytes as today's q4_0, strictly faster attention, NVIDIA-validated <1% loss. SM120 needs `mma.sync`-style kernels (no SM100/TMEM code) + CUDA 13.0. Nobody ships this on SM120 — differentiator.
6. **FP8 E4M3 KV** as the "safe 2×" default (1 B/elem): relative-precision step is better than q8_0 on outlier-heavy layers; Blackwell FP8 tensor cores for attention; also the **RDNA4 native format** (gfx1201 WMMA) — one format, both vendors.
7. **FP4 grid alignment via rotation**: E2M1's levels {0,±0.5,1,1.5,2,3,4,6}·s are near-optimal for Gaussian data; the existing fwht256 rotation makes rotated KV Gaussian → NVFP4 stops being "lossy 4-bit" and becomes near-optimal 4-bit. Rotation + NVFP4 is the synergy paper results point at.
8. **E1M2 KV experiment** (ReQAT finding): if the tensor-core path accepts it, E1M2 for KV measured lower-loss than E2M1.

**Tier 3 — TurboQuant / codebook track**

9. **Wire TQ1_0/TQ2_0 (already declared in the ggml fork) into the CUDA KV path**: Hadamard rotation + precomputed Lloyd-Max codebook (256-entry, computed once at model load) + optional 1-bit QJL sign residual. Target: **3.5-bit quality-neutral, 2.5-bit marginal** per the paper — a full tier below q4_0. Data-oblivious = no calibration pipeline needed.
10. **Layer-sensitivity bit allocation** (KVQuant nuqX / KV-Tuner): one offline calibration pass (Fisher-diag sensitivity per layer) → 2-bit on tolerant layers, 4-bit on sensitive ones; average ~3 bits at near-4-bit quality. Fits Strata's existing per-layer quant philosophy (`arch_defaults.cpp`).
11. **Pre-RoPE / pre-bias K quantization**: quantize keys before RoPE pairs mix channels and before the K-projection bias inflates ranges (KVQuant: −0.8 PPL at 3-bit; pre-bias: +7.8 pts at 3-bit). Needs a hook in `layer.cpp` where K is produced.

**Anti-goals (measured, don't chase)**
- FP8 attention on Hopper-class long-context accumulation bug — N/A on Blackwell, keep the two-level accumulation note if a ROCm/RDNA4 path ever runs >100K contractions.
- 2-bit without rotation/sinks: every 2024–2026 paper agrees plain 2-bit per-token collapses; 2-bit is only viable as rotation+window+asymmetric.
- MXFP4 for K: power-of-two scale dial is measurably worse than NVFP4's E4M3 scale on KV (~5% gap per NVIDIA).

## 4. Implementation notes for this box

- Build new CUDA kernels with **CUDA 13.0** (release toolchain). 13.2.0/13.2.1 miscompile IQ2_S/IQ3_S kernels on sm_120 (known landmine); 13.0/13.1/13.2.2 are clean. Keep `engine/BUILD.json` at `source: release` — do not rebuild `engine/strata.exe` from source here.
- Validation harness: extend `kv_q8_parity.cpp`-style bitwise/quality parity tests per new format (codes bitwise-equal to host reference; dequant error ≤ half-step + fp16 rounding), then a PPL + needle-in-haystack sweep at ctx 32k/128k on swift-iq3_xxs, A/B against the current q4_0 baseline (`strata-sweep-*.json` pattern).
- Quality gate: adopt the community bar — q8_0/FP8 delta within noise; q4_0/IQ4_NL/NVFP4 within ~1% on long-context retrieval; 2.5–3.5-bit TurboQuant tier must pass needle@128k ≥ 0.95.

## 5. Key references

- KIVI 2402.02750 · KVQuant 2401.18079 · TurboQuant 2504.19874 (ICLR'26) · QuaRot 2404.00456 · RotateKV 2501.16383 · OScaR 2605.19660 · OptR 2608.02691 · SKVQ 2405.06219 · KVSink 2508.04257 · Quantize What Counts 2502.15075 · ScaleSearchAttention 2605.12464 · ReQAT 2606.15682 · Quantization Hurts Reasoning? 2504.04823 · XQuant (EMNLP'25) · PM-KVQ 2505.18610 · Dual-QK 2610.09827 · surveys: 2607.08057, 2607.02574, 2503.24000
- NVIDIA: research.nvidia.com "Pushing Intelligence to 4-bit" (NVFP4 KV numbers) · Blackwell tensor-core docs · OCP MX spec
- AMD: HotChips 2025 RDNA4 deck (F8/INT8/INT4 tensor types, sparsity) · RX 9070 product page (FP8 289 TFLOPS dense) · vllm-rocm-rdna4 fork
- llama.cpp: ik_llama.cpp parameters.md (hadamard KV flags) · GGML_CUDA_MPREC docs · vLLM FP8 KV blog 2026-04-22
