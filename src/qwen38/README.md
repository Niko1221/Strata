# Strata-owned dense Qwen35 / Qwen3.8 CUDA path

This standalone, opt-in engine owns GGUF admission, original compressed weight
storage, the 64-layer dense forward loop, resident request state and sampling.
There is no llama model/context/graph/scheduler/sampler API, linked ggml/llama
runtime or external inference process. Existing MoE defaults are unchanged.

Build this directory with CMake, MSVC, CUDA 13 and SM120. The target is
`strata-q38`. Request at most five compiler jobs. Windows runtime resource
floors remain available RAM 3 GiB, commit 4 GiB, CUDA free reserve 768 MiB,
H: free 10 GiB and C: free 5 GiB. No diagnostic process memory cap is implied.

`--model FILE --inspect` reads CPU metadata only. Admission uses architecture,
tensor geometry, operation-specific types and file extents, not author or model
names. It accepts dense qwen35 with the 5120/17408/248320 geometry, 64 trunk
layers in a 3:1 GDN/attention pattern, and zero or one declared MTP layer.
An optional block 64 is validated for safe extents but ignored, never loaded.
Split files, experts, unknown trunk roles/types and overlapping extents fail.

`--model FILE --ctx-size 4096 --tokens IDS --n 16` generates token IDs. The
server supplies tokenization and the embedded chat template. No native text
tokenizer is claimed. `--force IDS` overrides `--n` with forced-token count
plus one; `--dump-full-logits FILE` writes raw little-endian float32 rows of
248320 values. Row 0 is after the final prompt token; row i+1 follows forced
token i. No EOS stopping occurs on the forced-token path. Without `--force`,
the dump contains one row per generated token, before sampling that token.

`--serve` provides INFO/READY/GEN/PP/T/DONE/STOP/QUIT. Sampling options are
temperature, top_k, top_p, min_p, seed, penalty_repeat, penalty_last_n,
penalty_freq and penalty_present. top_k=0 considers the whole vocabulary.
An active generation owns output through its terminal flush. STOP cancels
between layers; state is reset after completion, cancellation or errors.
Prefix snapshots, shift, vision, speculation and batched sessions are absent
and rejected. INFO uses fixed ASCII values and numeric byte counts.

Weights remain original compressed rows on CUDA. Embeddings remain a read-only
host mapping and upload one compressed row per input token, without pinning
the mapping. There is PCIe traffic. F32/F16/BF16 auxiliary tensors convert in
bounded tensor scratch; compressed alpha/beta remain compressed matvecs.
Attention uses F16 K/V. GDN state and activations are FP32. Sequential token
prefill is the only supported batch behavior; there is no CUDA graph.

PQ2_0 is type 142, 128 elements per 34 bytes, independently of the existing
MoE type 42 (64/18). The owned CUDA dot accumulates FP32 activations against
decoded original packed values. It does not first quantize activations to Q8.
Supported matrices are F32, F16, BF16, PQ2_0, IQ4_XS, Q4_K, Q5_K, Q6_K and Q8_0.

The Hikari checkpoint also requires its explicit Hadamard metadata. Version 1,
normalized Sylvester H1024, explicit sign vectors and the input-last-dimension
axis are admitted. Folded matvec input is optionally grouped for ssm_out from
[128,16,3] to [128,3,16], multiplied by its signs, then transformed by H1024.
Latent embedding rows instead receive H1024 followed by signs. A CUDA butterfly
implements these transforms without expanding weights. Missing or malformed
transform metadata fails. The shared GGUF reader has an opt-in set of retained
array keys capped at 65536 items; default callers retain the historical 64-item
sample behavior. GDN itself still maps value head to Q/K head with modulo 16.

Numerical fixtures execute the same production CUDA functions. Decoders compare
exactly against independent CPU block loops. Matvec tolerance is absolute
2e-4 plus relative 2e-5 (FP32 accumulation against FP64 reference). Hadamard uses
absolute 3e-6 plus relative 2e-5 against a direct FP64 parity matrix, including
all three widths, inverse order and grouped GDN. GDN output tolerance is
5e-5 + 5e-4 relative; state tolerance is 5e-6 + 5e-4 relative over six steps.
GQA uses 4e-4 + 3e-4 relative, accounting for FP16 KV rounding boundaries;
Q/RoPE uses 2e-5 + 2e-5 relative. These tolerances were specified before running
the first fixture. Tests include finite failures, capacity, reset, protocol
terminal races and the real Python DONE consumer. They do not qualify a real
checkpoint, model quality, full context, concurrency or throughput.

Provenance: PQ CPU byte semantics and Hadamard ordering were inspected at
PrismML 0.4.1 commit `1965755ceee6738b9a04ceaf5d30e6305d6ccd00`, specifically
ggml/src/ggml-quants.c dequantize_row_pq2_0 and src/llama-graph.cpp
build_lora_mm/build_embd_rows, src/llama-model.cpp transform metadata.
Quantized layouts and codebook agree with existing Strata
src/kernels/cuda/iq_kernels.cu and native_mmvq.cu. The existing Strata
fused_gdn.cu is compiled unchanged (SiLU variant, modulo head mapping).
The attention mathematics follow q36_engine.cu k_attn_prep and causal GQA,
with new owned F16 cache kernels. Runtime-independent protocol helpers were
copied from Strata commit e1f528ec04e5de0a7ee5411e08ddaf62fedbda6c and extended
so QUIT requests cancellation. Format attribution is in LICENSE-formats.txt.

Actual full-model acceptance and comparison with an independently built Prism
oracle are a separate controller operation after this source/build handoff.
This implementation does not claim release qualification.
