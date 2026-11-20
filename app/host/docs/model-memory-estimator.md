# Host model-memory estimates

The Host can calculate theoretical attention KV bytes from validated GGUF
geometry. This is a calculation, not a GPU allocation or performance result.
For Hikari's qwen35 geometry, 131,072 tokens and one sequence with q8_0 for
both K and V require **4.25 GiB of theoretical attention KV only**. This does
not establish that the model fits a 16 GiB GPU. Backend weights placement,
recurrent state, allocator overhead and compute workspace still need measurement.

`server/model-memory.mjs` exposes the public APIs, backed by the pure
`server/model-memory-geometry.mjs` arithmetic. `modelMemoryFromMetadata(metadata,
{diskBytes})` validates metadata and returns a small serializable geometry record.
`estimateModelMemory({modelMemory, resolvedKv, contextTokens, sequences,
backendProfile})` consumes the K/V pair from `resolveKvSettings`. Callers may
instead supply `expert` to use that resolver inside the estimator.

The estimator's context is per sequence. It pads each sequence's capacity to
256 tokens and multiplies by `sequences` once. The result exposes
`contextSemantics`, `allocationTokensPerSequence`, `allocationTokenCapacity`
and `contextPaddingTokens`. A caller dealing with a runtime's total `-c`
capacity must map that value before calling. These functions do not alter
launch arguments or resolve the app's existing parallel/context behavior.
The 256-token padding follows the pinned reference's non-unified sequence
layout. Unified and other backend allocation strategies require separate
qualification. `backendProfile` currently records its `id` only; it cannot
assert weights allocation, change row sizes or enable TurboQuant arithmetic.

For qwen35/qwen35moe, a valid explicit `attention.recurrent_layers` scalar
or integer/boolean array takes precedence. An array must have `block_count`
entries, each boolean or 0/1. An invalid explicit map produces unknown geometry;
it never falls back to the interval. Without an explicit map, a declared
`full_attention_interval` must be a positive integer no larger than the trunk.
Only an absent interval uses the reference default of 4.
`nextn_predict_layers` is subtracted from the total block count to obtain the
trunk. MTP blocks are recorded separately and are not recurrent trunk layers.
The attention estimate covers the trunk; additional MTP allocations remain
part of the unmeasured allocation requirement.

Hikari's 64 trunk layers, interval 4, four KV heads and separate 256-element
K/V head widths give 16 full-attention and 48 recurrent layers. Its embedding
length of 5,120 divided by 24 attention heads is not the K/V head width.
SSM dimensions are retained but never multiplied by context to fabricate KV.
Recurrent state depends on backend state/rollback layout and remains unknown.

Each K/V side uses its own GGML row layout. q8_0 uses a 32-element block of
34 bytes; q4_0 uses 32 elements and 18 bytes; f16/bf16 use two bytes per
element. The scale bytes are already included. Quantized head widths must
contain whole blocks; incompatible shapes are unknown rather than rounded
to a different precision. At 131,072 tokens, one sequence, Hikari yields
4.25 GiB for q8_0/q8_0, 2.25 GiB for q4_0/q4_0, 8 GiB for f16/f16, and
6.125 GiB for q8_0/f16. TurboQuant remains explicitly unqualified here.
No total GPU allocation or benchmark throughput is implied by these numbers.

The result's `components` separate `disk`, `attentionK`, `attentionV`,
`attentionKv`, `backendWeights`, `recurrentState` and `workspace`. Each
component has `bytes`, `source` and `reason`. Unknown bytes are null.
`knownSubtotalBytes` counts attention KV once and any known recurrent state;
it excludes disk and the duplicate K/V detail rows. `totalBytes` and `fits`
remain null because required backend allocations are unmeasured. The
confidence is `theoretical-attention-only` or `unknown`.

Simple dense GQA families with validated geometry retain theoretical attention
estimates. Unknown architectures, MLA, sliding-window-specific layouts and
other unqualified hybrids do not inherit a confident dense formula. This
restriction does not change which models the existing importer can load.
Built-in catalog footprint behavior is unchanged.

The existing `estimateModelMemory({metadata, context, kType, vType, ...})`
plan API remains available, as do `extractHybridGeometry`, `ggmlRowBytes`,
`estimateAttentionKvBytes`, `classifyHybridLayers` and `hybridKvKbPerToken`.
The legacy plan still returns `attentionKvGiB`, `nFullAttention`, `nRecurrent`
and high/low confidence. Its calculations now use the same padding, MTP split,
strict block alignment and unknown handling as the geometry API. Unknown
architectures have null attention bytes rather than an all-layer candidate.
The legacy classification helper accepts explicit layer indices; it is not a
GGUF parser. Raw GGUF maps are validated as bitmaps/scalars by the geometry API.
Imported variants retain `memoryPlan` and `local.nFullAttention` alongside the
additive `modelMemory`, with naming and rename-on-reinspection behavior intact.

New imported variants and their local records contain additive `modelMemory`
metadata. The retained GGUF metadata participates in `metadataSignature`,
which is a metadata hash, not a full-file content hash or provenance signature.
No read-time migration writes existing local records; old records remain
readable and may have no modelMemory until the user reimports them.

The legacy `kvKbPerToken` display field means KiB/token at one byte for each
K and V element, before codec scales or padding. Hikari's baseline is 32,
counting only 16 full-attention trunk layers, instead of 128 for all 64 layers.
It is not an exact q8_0 estimate. Consumers of the new API must use exact row
bytes directly and must not multiply by the legacy codec factor again.
Non-hybrid legacy imports keep their prior display baseline behavior.

The source reference is Prism commit
`73bfe6cc485554822f80c473a3c0c4f25e80f41b`: `src/models/qwen35.cpp`,
`src/llama-arch.cpp`, `ggml/src/ggml-common.h`, `src/llama-kv-cache.cpp`,
`src/llama-context.cpp`, `src/llama-memory-hybrid.cpp` and
`src/llama-memory-recurrent.cpp`. The installed runtime is a different build
(`c9c34bc9c`); this source inspection does not validate its allocations.

Focused tests use tiny synthetic GGUF files with the existing scratch HOME
and filesystem cleanup guard. They exercise codec sizes, mixed/asymmetric
K/V, explicit maps and scalar forms, MTP/trunk counts, malformed/truncated
arrays, context bounds, sequences, overflow, old-record reads and metadata
identity changes. They never open the user's real GGUF or launch GPU inference.
