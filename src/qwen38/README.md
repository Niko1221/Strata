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
Attention uses the selected native packed KV mode. GDN state and activations are FP32. Sequential token
prefill is the only supported batch behavior; there is no CUDA graph.

PQ2_0 is type 142, 128 elements per 34 bytes, independently of the existing
MoE type 42 (64/18). The owned CUDA dot accumulates FP32 activations against
decoded original packed values. It does not first quantize activations to Q8.
Supported matrices are F32, F16, BF16, PQ2_0, IQ2_S, IQ3_S, IQ4_XS, Q4_K, Q5_K, Q6_K and Q8_0.
IQ2_S and IQ3_S retain their original GGUF bytes and use 256-value blocks of
82 and 110 bytes respectively. Their codebooks and decoding follow ggml commit
`643956891dfdf51aecdbd05ca508dd21e21e8195`; the corresponding tables and
decoder bodies also match the model producer's commit
`030ebb558a5820b444a8f836ed5cdd46c9b4bd7a`. See `LICENSE-formats.txt`.
The runner validates and skips optional MTP tensors; it does not execute MTP
drafting, and reports `drafting=off`. Presence of those tensors does not enable
MTP acceleration. GPU and real-model acceptance must be recorded separately.

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

## Native KV storage and automatic admission

The new runner defaults to `--kv auto`. Before bulk CUDA allocations it budgets
original GPU weights, unchanged GDN/conv state, the entire explicitly requested
context, activation/score/logit buffers, embedding row, sign and rotation data,
bounded auxiliary conversion scratch and allocator allowances. The CUDA reserve
remains 768 MiB. Auto tries `f16`, then `f16_turbo4`, then `spin8_k8v4`. This is
an allocation policy, not a measured model-quality ranking. An explicit `--kv`
mode is never changed. If no candidate fits, admission fails with the required,
available and reserve bytes. Concurrent later VRAM pressure still fails through
the normal guard; a live session never changes its format. Context 98304 and
131072 remain valid requested capacities, without an inference-quality claim.

Available modes are `f16`, `k8v4`, `spin8_k8v4`, `f16_turbo2`, `f16_turbo3` and
`f16_turbo4`. K and V have separate formats in INFO. Ordinary `k8v4` stores
Q8_0 K and Q4_0 V. Spin8 uses those same physical layouts after its selected
rotation. Turbo modes retain ordinary post-RoPE F16 K and ordinary Q; only V
uses Turbo2, Turbo3 or Turbo4. Turbo K and composed Spin8+Turbo modes are absent.

For Spin8, `--spin8-view vector|plus|minus` selects one representation, default
vector. `--spin8-profile FILE` optionally replaces the reference profile.
Both switches require explicit `spin8_k8v4`. The default embeds the exact bytes
of `data/spin8/reference.profile` at build time and works without source files or
a working-directory convention. Actual `spin8_plan_create`, plan audit and
`spin8_plan_matrix` produce the chosen matrix. Rounded F32 coefficients receive
another orthogonality/determinant audit before allocation. INFO binds the
effective mode, view, profile SHA-256 and deployed matrix SHA-256.

The block-diagonal 8x8 matrix rotates Q and K after normalization and partial
RoPE, and rotates V once. Packed attention reads Q8_0/Q4_0 history directly.
The transpose rotates the aggregate back before the nonuniform sigmoid gate.
Turbo V uses the pinned TheTom 64395689 F=D2 H128 D1/sqrt128 transform, two
independent blocks per head, nonuniform centroids and corrected FP16 norm.
The inverse is D1 H128 D2/sqrt128, applied once to the value aggregate before
the gate. Production reuses `tqcodec/src/dialect_cuda_device.cuh`, including
its scalar decoder; attribution remains in `tqcodec/DIALECT_LICENSE.txt`.
Q8_0/Q4_0 layouts and rounding follow the pinned ggml reference encoders.

Two session-owned CUDA allocations contain all layers' K and V, with non-owning
per-layer views. There is no full-history floating shadow. Apart from causal
scores (24 * context FP32 elements), transform/aggregation scratch is bounded
to one position. Reset clears packed history, lengths, GDN, convolution and
error state through the same production helper used by cancellation and errors.
Invalid positions/rewinds are rejected. Numerical errors poison the cache until
whole-session reset. Device status prevents attention from reading failed
encoded storage. The forward loop checks sticky status at bounded group fences
and before any token can be published.

At 16 attention layers, 4 KV heads, head dimension 256 and capacity 4096,
the payload is 268435456 bytes for F16/F16, 109051904 for Q8_0/Q4_0, 152043520
for F16/Turbo2, 160432128 for F16/Turbo3 and 168820736 for F16/Turbo4.
Q8_0/Q4_0 use 34/18 bytes per 32 values; Turbo2/3/4 use 34/50/66 bytes per 128.
Fixed cache scratch is 4 bytes for F16, 24580 for plain K8/V4 or Turbo V, 57348 for
Spin8; Spin8 coefficients add 256 bytes. Alignment/backing allocation is separate:
CUDA 13/SM120 measurements showed 2 MiB granularity, including small-buffer pools.
Admission reserves per-allocation rounded slack for KV, observed large-buffer
slack for other allocations and the existing 16 MiB general safety allowance.
INFO reports requested/effective modes, selection reason, estimated total with
reserve, available bytes at admission, packed payload, scratch and coefficients.

GPU weight upload now uses a native 16 MiB pageable staging reader, exact admitted
offsets and short-read checks, and resource checks per chunk. It copies original
packed bytes without full-weight mmap access, host weight mirrors or pinning.
The read-only metadata/embedding mapping is retained for individual embedding
rows. This addresses a demonstrated RAM-pressure stop in the prior upload path;
the actual IQ4_XS replay remains a separate acceptance test.

The focused CUDA suite compares all stored bytes against independent CPU layouts
and the pinned CPU Turbo dialect, over 257 positions for every exposed mode/view.
It checks GQA 24/4, causal lengths 1/31/32/33/127/128/129/255/256/257, nonuniform
gates, nonsymmetric Spin8 inverse, F16 legacy equivalence and whole reset/error
reuse. Attention tolerance is absolute 3e-5 plus relative 4e-5 against FP64 CPU
attention over decoded stored bytes; pure F32 Spin8 inverse uses 5e-7 + 5e-7 relative.
These tolerances concern arithmetic correctness, not quantization quality.
Actual allocation snapshots at 513 and 4096 cells bound allocator overhead and
establish fixed scratch. The CPU suite checks production auto-selection exact-fit,
one-byte-below, no-fit, overflow and explicit-mode boundaries, profile admission
and exact reader bytes/bounds/short reads. The existing CUDA dialect golden-file
suite also covers the extracted scalar decoder. These tests do not qualify
27B model answers, 98k/131k full-window inference, throughput or application delivery.

## Weight arena and bounded execution

One session-owned allocation contains GPU weights as 256-byte-aligned,
non-owning spans. The original compressed payloads and host-only embedding stay
unchanged. Auxiliary conversion writes into its exact F32 span and retains at
most one temporary compressed auxiliary tensor. The 16 MiB pageable upload
reader and resource checks remain. Checked layout arithmetic includes padding
in admission; stderr reports payload, arena bytes, padding, allocation count,
and measured free VRAM before/after allocation. Failed construction releases
the owner once. Individual views cannot free storage.

Before admission, explicit kernel attribute queries and a tiny production KV
and GDN warmup observe initial code/launch residency. The small-pool allowance
is retained and another 128 MiB covers launch-size-dependent residency beyond
that warmup. This allowance is conservative, not a proven universal maximum.
The 768 MiB GPU reserve and RAM/commit/disk floors are unchanged. Any later
reserve failure remains fatal with stage, tensor, free, needed and reserve
bytes. Preload snapshots do not establish the cause of any historical failure.

Normal execution checks STOP before each layer and synchronizes at most eight
layers at once. At the fence it checks sticky KV and activation errors, STOP,
and resource floors. KV consumers check device status before reading encoded
storage. Numerical validation remains enabled; cancellation/error still resets
the whole session. `--conservative-sync` is a diagnostic reference path with
per-layer and intermediate KV status fences, not a product UI setting.

The hardware fixture checks bit-exact conservative/grouped outputs for all six
KV modes, all three Spin8 views and changing inputs at capacities 9 and 33;
NaN/encoding overflow, poison/reset/reuse; exact/below admission; allocation
alignment, overflow, conversion boundaries and failed-construction cleanup.
Its CUDA-event and wall timings include bounded host scheduling gaps. It reports
two warmup runs and seven measurements per mode, plus explicit host-fence counts.
These are small synthetic KV measurements, not full-model throughput. Packed
matvec specialization, batched prefill, MTP and graphs are not part of this pass.
