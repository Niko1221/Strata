# Query-v3 adapter sidecar, version 1

This is an optional, engine-independent CPU container. It does not install a
model hook, run a model, authenticate a publisher or establish that an adapter
was trained. It is not a standard LoRA file. No pickle, executable object,
filesystem path or external payload is present.

All integers are unsigned little-endian. All parameters are finite IEEE 754
binary32, little-endian, row-major, with their original bits retained, including
negative zero and subnormals. A file is at most 67,108,864 bytes including its
footer. There is exactly one header, one ordered model digest list, one sorted
layer directory, contiguous layer payloads, and one footer. Gaps, overlaps and
trailing bytes are invalid. There is no compression or alignment padding.

## Header (160 bytes)

| Byte offset | Size | Required value or meaning |
| --- | --- | --- |
| 0 | 8 | ASCII `SP8QV3` followed by CR LF |
| 8 | 4 | Wire version 1 |
| 12 | 4 | Header byte count 160 |
| 16 | 4 | Flags 0; every bit is reserved |
| 20 | 4 | Model file count, 1 through 256 |
| 24 | 4 | Selected layer count, 1 through 4096 |
| 28 | 4 | Directory record bytes 32 |
| 32 | 8 | Exact complete file byte count, including footer |
| 40 | 4 | Mathematical controller version 3 |
| 44 | 4 | Rank 8 |
| 48 | 4 | Chart 1: rational chart below |
| 52 | 4 | Basis 1: existing positive-chirality Gamma/C/J convention |
| 56 | 4 | Placement 1: after model Q normalization, before RoPE |
| 60 | 4 | Angle marker `0x3ec90fdb`; nearest F32 pi/8 |
| 64 | 32 | Intended codec/profile identity, SHA-256 |
| 96 | 32 | Intended position-processing identity, SHA-256 |
| 128 | 32 | Fixed convention contract SHA-256, defined below |

There are no optional fields or other reserved bytes. Unknown flags, versions,
tags or angle markers fail. The angle marker names the mathematical pi/8
bound; it is not substituted for pi/8 in controller evaluation.

The convention digest hashes exactly these concatenated ASCII bytes, with no
newline or NUL terminator:

```text
spin8-query-v3;rank=8;chart=rational-tau-tan-pi/8;basis=positive-chirality-Gamma-C-J-v1;map=C(c)C(e0)^T;placement=post-Qnorm-pre-RoPE;mlp=Wout*tanh(Win*q+bin)+bout;no-extra-RMS;tail=identity
```

This digest identifies a documented contract, not executable semantics. The
existing `spin8_query_v3_basis_audit` verifies the actual core basis separately.
The fixed convention uses Gamma0 through Gamma7 in that order, positive
chirality first, `C(c)[i,a] = sum_b Gamma_i[a,8+b] c[b]`, and
`J_a = C(e_(a+1)) C(e0)^T`. Its map is `A(c) = C(c) C(e0)^T`.
For already normalized model Q, the controller computes
`a = Wout tanh(Win q + bin) + bout`, then
`tau = tan(pi/8)`, `u = tau a / sqrt(tau^2 + ||a||^2)`, and
`c = (1,u) / sqrt(1 + ||u||^2)`. Complete blocks of eight use A(c); a final
one through seven coordinates pass through exactly. It does not add another
RMS normalization, exponentiate an old proxy, compose alternate factors or
silently normalize supplied data. See `include/spin8/query_v3.h` and the
independent existing v3 tests for the signed-permutation basis convention.

## Model identities and directory

Immediately after the header are `model_count` 32-byte SHA-256 digests, in the
actual model file/shard order. A single Bonsai file and the two Strata shards
use the same format. Every model, profile and position digest must be nonzero.
The format stores no local paths. Integrity is not publisher authentication.

Each following 32-byte directory record has four u32 fields: model layer
index, Q head count, KV head count, head width. They are followed by the
absolute payload byte offset (u64) and exact payload byte length (u64).
Layer indices must be strictly increasing, at most 1,048,575. Q and KV head
counts must be 1 through 4096, with Q divisible by KV. Width must be 1 through
16,384, including widths with a partial final block. The file size limit also
applies to the aggregate geometry. These bounds precede size arithmetic.

Each layer payload has four consecutive arrays in this order:
`Win[Q,8,width]`, `bin[Q,8]`, `Wout[Q,7,8]`, `bout[Q,7]`.
It is exactly `4 * Q * (8 * width + 71)` bytes. The first payload starts
immediately after the last directory record; each next payload starts exactly
where the preceding one ends. The final 32 bytes are SHA-256 of every
preceding file byte. The metadata also exposes SHA-256 of the complete file,
including the footer, as its content identity.

## C ABI and lifetime

`include/spin8/query_adapter.h` is usable from C and C++. No GGML, GPU,
filesystem or network type appears. ABI version 1 descriptors carry exact
`struct_bytes` and explicit byte capacities. Expected descriptors and layer
arrays need native alignment; source file bytes may be unaligned. The caller
provides real accessible spans, and keeps all input memory stable for the
duration of a call. Numerical pointer and span checks cannot establish that
arbitrary addresses reference accessible memory.

The caller supplies an independently established ordered model digest list,
profile digest, position-processing digest and exact selected layer specs.
It must derive them from the actual selected model files and runtime settings.
Copying expectations out of the untrusted sidecar defeats the binding. This
memory-only loader does not read model files or establish runtime settings.
The application owns the canonical meaning of profile and position identities;
their full definitions must cover all relevant runtime options. The loader
compares the complete digest bytes and geometry; it does not infer them.

`spin8_query_adapter_load` verifies the checksum, canonical structure, bindings,
geometry and finite scalars before publishing an object. Payload is copied
into owned `float`-aligned storage. No C++ exception crosses the ABI. A valid,
disjoint result pointer becomes null on validation failure after every declared
input span has passed numerical range and output-disjointness checks. An invalid
or input-aliased result pointer, or any nonnull input span whose endpoint cannot
be represented, is rejected without output writes. This conservative early
failure preserves the input even when a claimed capacity is SIZE_MAX. Nonnull
zero-capacity inputs protect their base byte for alias checks. The same rule
applies to top-level file/expectation spans and nested model/layer capacities;
invalid counts cannot disable it. An invalid expectation descriptor is rejected
before any nested field is read. No wire or ABI version changes are introduced.

`get_info` and `get_layer` take output byte capacities and leave output unchanged
on failure. Directory ordinals start at zero and are different from model layer
indices. F32 view capacities count bytes for each entire per-layer array.
Returned pointers are immutable borrowed views. They expire on adapter free.
Concurrent immutable readers are allowed; the caller must serialize free
against readers and must not mutate any borrowed storage. Getter output may
not overlap the adapter or its owned storage. `free(NULL)` is allowed.

| Status | Failure class |
| --- | --- |
| `SPIN8_INVALID_ARGUMENT` | Invalid spans/alignment/capacities/descriptor, bounds, geometry or canonical layout |
| `SPIN8_UNSUPPORTED` | Unknown ABI/wire version, flags or mathematical tags |
| `SPIN8_AUDIT_FAILED` | Footer or expected model/profile/position/convention/selected-layer mismatch |
| `SPIN8_NONFINITE` | A parameter's F32 exponent encodes NaN or infinity |
| `SPIN8_ALLOCATION_FAILED` | Owned adapter allocation fails |

Failure precedence is not an interface guarantee when several problems coexist.
Allocation is bounded by the 64 MiB input and bounded metadata, with one owned
F32 payload copy. The input itself remains caller-owned.

## Python writer and inspection

`tools/spin8_query_sidecar.py` uses the standard library only. `encode` accepts
explicit bindings and `Layer` objects with F32 byte arrays. `decode` independently
checks the format and requires external bindings and exact selected specs.
`identity_layer(spec, win_scale=...)` permits nonzero Win while Wout and bout
are zero, giving identity output. Initialization is deterministic and does not
claim learned weights. Files are created exclusively unless `--overwrite` is
explicit. The CLI requires all bindings and geometry even for inspection:

```text
python tools/spin8_query_sidecar.py create adapter.sp8q --model-sha256 <actual-file-hash> --profile-sha256 <profile-hash> --position-sha256 <position-hash> --layer 2:24:4:256 --layer 7:2:1:11 --win-scale 0.125
python tools/spin8_query_sidecar.py inspect adapter.sp8q --model-sha256 <actual-file-hash> --profile-sha256 <profile-hash> --position-sha256 <position-hash> --layer 2:24:4:256 --layer 7:2:1:11
```

Repeat `--model-sha256` in the actual shard order. The CLI returns JSON with
full content and integrity hashes, counts and byte length. The C++ library
does no file reading. Adding this container leaves the default model path
unchanged; actual model integration is a separate reviewed stage.
