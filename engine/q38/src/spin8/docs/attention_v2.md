# CPU attention v2

`spin8/attention_v2.h` is an opt-in, standalone FP64 self-attention API. It adds
explicit head mapping, buffer capacities and attention ordering to a separate
C ABI. It does not change the v1 attention implementation or enable an engine
path. It is not evidence of GPU, quantized, learned-adapter or model support.

## Plan and source contract

Zero-initialize `spin8_attention_v2_config`, set `abi_version` to
`SPIN8_ATTENTION_V2_ABI_VERSION` (2), and set `struct_bytes` to `sizeof(config)`.
Pass that exact size as `config_bytes`. All four reserved words must be zero.
The API checks both sizes. It does not accept larger future descriptors silently.

Set `semantics=SPIN8_ATTENTION_V2_SELF`, `dtype=SPIN8_F64` and
`codec=SPIN8_ATTENTION_V2_CODEC_NONE`. Choose positive `n_q_heads`, `n_kv_heads`,
`d_qk` and `d_v`. Q and K share `d_qk`; V and output use the independent `d_v`.
Provide `n_q_heads` uint32 entries in `q_to_kv`, with byte capacity in
`q_to_kv_bytes`. Every entry must be less than `n_kv_heads`. Divisibility and
contiguous groups are not required. For example, five Q heads can map to three
KV heads as `{2, 0, 2, 1, 0}`. The plan copies the map.

`source_id` must be nonzero. It is the caller's identity for one self-attention
source, including its layer and sequence as needed by the caller. Each request's
`q_source_id` and `kv_source_id` must equal this plan identity. These are explicit
caller assertions, not tensor addresses, content hashes or authenticated IDs.
The library cannot detect a caller falsely labelling cross-attention as self-
attention. Cross-attention semantics and mismatched source IDs are rejected.

Q and K must already have their position handling applied. Each request declares
`position_stage=SPIN8_ATTENTION_V2_POST_POSITION`. The library does not apply
RoPE, normalization or a learned query adapter. A query subset and a longer KV
prefix are permitted within the same source. The caller supplies their exact
visibility mask; there is no implicit causal or sliding-window mask.

`spin8_attention_v2_plan_create` publishes an opaque immutable plan only after
validation and audits succeed. On failure, the caller's plan pointer is unchanged.
The output pointer slot must not overlap the config, mapping or plane storage.
The plan owns all retained data; the caller may release the config, mapping and
planes after creation. Free it with `spin8_attention_v2_plan_free`; null is valid.
Calls may share a live plan with distinct output buffers. Destruction must be
serialized against all users. There is no mutable session or cache state.

## Basis modes

`SPIN8_ATTENTION_V2_IDENTITY` is the original-space baseline. Its `view`,
`hadamard_block`, `planes`, `plane_count` and `planes_bytes` must all be zero.

`SPIN8_ATTENTION_V2_FIXED` uses one fixed view (`SPIN8_VECTOR`,
`SPIN8_SPIN_PLUS` or `SPIN8_SPIN_MINUS`) for the whole plan. Provide plane factors
and their count/capacity, or use null/zero for identity factors. The constructor
calls the existing audited `spin8_plan_create`; execution calls the existing
`spin8_transform`. The selected basis is common to Q and K for every head. Thus
all Q heads sharing one KV head necessarily share its basis. This API has no
per-head independent basis or multi-view reduction.

Set `hadamard_block=128` to apply normalized, all-positive Sylvester H128 before
Spin8, or zero to omit it. Other block sizes are unsupported here. H128 uses
`spin8_hadamard_block`; its FP64 coefficient audit is checked at plan creation.
Each complete 128-coordinate block transforms, and its incomplete tail is
unchanged. Spin8 then transforms each complete group of eight, with its own
incomplete tail unchanged. These rules apply separately to `d_qk` and `d_v`.
For width 192, H128 leaves the last 64 coordinates unchanged, but the following
Spin8 stage still transforms those eight complete blocks of eight.

Forward Q, K and V use H128 then Spin8. After attention reduces transformed V,
output uses inverse Spin8 then inverse H128. An optional elementwise gate acts
only after this inverse. The implementation performs the actual transforms;
it does not substitute original-space arithmetic based on orthogonality.

## Requests, strides and memory

Zero-initialize `spin8_attention_v2_request` and set its ABI version, exact
struct size and source/position fields. Set positive `n_query` and `n_key`.
Tensor shapes are:

| Field | Shape | Element |
| --- | --- | --- |
| q | `[n_query,n_q_heads,d_qk]` | double |
| k | `[n_key,n_kv_heads,d_qk]` | double |
| v | `[n_key,n_kv_heads,d_v]` | double |
| output, gate | `[n_query,n_q_heads,d_v]` | double |
| mask | `[n_query,n_q_heads,n_key]` | uint8 |
| bias | `[n_query,n_q_heads,n_key]` | double |

Each descriptor has its own positive token, head and channel **byte** strides.
For mask and bias, the channel axis indexes keys. K and V strides are independent.
Inputs support positive striding and read-only sharing, including Q/K aliases.
Every capacity must cover the last accessed byte, including intervening padding:

```text
(tokens-1)*token_stride + (heads-1)*head_stride
  + (channels-1)*channel_stride + element_bytes
```

Products, sums, internal element counts and pointer-range arithmetic are checked
before data loads. Declared spans must fit `PTRDIFF_MAX` and the address space.
Tensor data, maps and plane arrays may be unaligned; copies use byte operations.
The outer descriptors and plan output pointer slot require native alignment.
The caller must provide accessible memory for the declared spans, keep input
data stable throughout a call and pass only a live plan returned by this API.
Numeric checks cannot establish pointer provenance or actual allocation size.

Output is caller-owned and may be padded or axis-permuted. To prove its elements
are disjoint, the API sorts axes by stride. Each non-singleton axis must have a
stride at least the byte span of the preceding axes, starting with eight bytes.
Exotic interleaved layouts that fail this conservative test are rejected even
if their individual elements could be disjoint. All strides remain positive,
including singleton axes. Output's entire declared capacity must be disjoint
from input capacities, the request descriptor and plan storage. Exact in-place
and partial input/output overlaps are rejected; padding overlap is also rejected.

Mask is required. Zero means excluded; any nonzero byte means allowed. Bias and
gate are optional: their entire descriptor must be zero when absent. Present
bias/gate have the shapes above, with no implicit broadcasting.

## Logits and failure behavior

`scale` is explicit and finite. There is no hidden `1/sqrt(d_qk)` factor. Zero
and negative scales are accepted. `softcap` is finite and nonnegative; zero
disables it. For positive `cap`, the two supported orders are:

```text
SCALE_BIAS_SOFTCAP: cap*tanh((scale*dot(Q,K) + bias)/cap)
SCALE_SOFTCAP_BIAS: cap*tanh((scale*dot(Q,K))/cap) + bias
```

Exactly one stable softmax runs per query/head over allowed keys. Excluded keys
are never sent through bias or softcap and receive zero weight. Every row must
have at least one allowed key. All Q/K/V, bias and gate elements must be finite,
including data at masked keys. Padding is not inspected. In particular, an
infinite bias is not a substitute for the explicit mask.

Arithmetic overflow in transforms, dot products, scaling, bias, value reduction
or gating returns `SPIN8_NONFINITE`. Softmax's negative exponential tail and
softcap's saturating ratio may underflow/saturate normally. All results are
staged in private memory. Any failure leaves every caller output byte unchanged;
success writes only output elements and preserves padding.

| Status | Meaning |
| --- | --- |
| `SPIN8_INVALID_ARGUMENT` | Bad size, reserved value, geometry, span, stride, alias, mapping, negative cap or all-masked row |
| `SPIN8_UNSUPPORTED` | ABI, dtype, codec, semantics, source, position stage, basis/view, H block or logit order not supported |
| `SPIN8_NONFINITE` | Nonfinite input or numerical overflow |
| `SPIN8_AUDIT_FAILED` | Existing Spin8/H128 audit rejected the plan |
| `SPIN8_ALLOCATION_FAILED` | Scratch or plan allocation could not be satisfied |

No C++ exception crosses the C ABI. Allocation and length errors become allocation
failure; unexpected exceptions become invalid argument. Invalid raw pointers and
use-after-free are outside the caller contract, not recoverable C++ exceptions.

## Verification and limits

The Windows VS 2022 x64 Release tests compare against an independent scalar FP64
original-space oracle with absolute tolerance `1e-11`. Deterministic cases cover
MHA, noncontiguous GQA and MQA at Q/K widths 64, 80, 96, 128, 130, 192 and 256,
with different V widths, both logit orders, cap on/off, independent strides and
sentinels. Fixed basis tests cover all three views, with and without H128.

Wrong Q/K bases and gate-before-inverse controls must diverge on the same fixture
that the real path passes. Finite-input transform-overflow witnesses distinguish
real Q/K/V and H128 execution from an identity shortcut. Boundary tests exercise
bad ABI, reserved fields, dtypes/codecs, masks, NaN/Inf, overflow, copied inputs
and exact/partial aliases. A separate `.c` executable compiles and calls the API.
Fresh command results are in
[`_docs/2026-11-06_spin8-attention-v2_OpenCode.md`](../../_docs/2026-11-06_spin8-attention-v2_OpenCode.md).

This implementation allocates FP64 scratch and is intended for bounded CPU
workloads. It does not provide quantization, learned transforms, multi-view
attention, sessions, GPU execution, batching or native engine integration.
The numerical bound is measured on these deterministic cases, not a universal
floating-point error bound for arbitrary magnitudes or sequence lengths. No
model quality, throughput, memory-saving, GUI, installer or release claim follows.
