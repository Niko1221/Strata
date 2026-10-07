# Visual embedding transport

This is the host contract for the video implementation in progress. It does not enable video in the server or
engine. Integration and shipment still require the unrun decoder, projector, inference and backend gates.

Implementations: `serve/media.py` and `include/strata/program/media_embeddings.hpp`, with the C++ definitions in
`src/program/media_embeddings.cpp`. Neither implementation requires a model, decoder, server or GPU.

## Compatibility

Existing image requests still use their existing `ENC` / `SVE1` / `GENI` path. Nothing in that path has been
replaced. The new legacy adapter exists for future mixed image/video requests, not to change image-only requests.

SVE1 is a concatenation of image records. Each has five little-endian signed 32-bit fields:
`{0x31455653, rows, nx, ny, width}`, followed by `rows * width` IEEE float32 values. All dimensions are positive,
`rows == nx * ny`, and floats are finite. The adapter requires one record per complete image-pad run, matching
widths and a following text token, as the existing engine does. It creates row-major relative `(t,h,w)` positions
`(0, row / nx, row % nx)` with position advance `max(nx, ny)`.

## SVE2 wire layout

All integers and floats are little-endian. Signed fields use two's complement. There is no alignment padding,
serialized C++ struct, compression, JSON embedding array or optional extension area. Unknown versions, flags or
reserved values are errors. Streams must be blocking binary file streams with exact EOF after the bundle.

The four contiguous sections following the header are tokens, span descriptors, positions and embeddings.

### Header: 64 bytes

| Offset | Type | Meaning |
|---:|---|---|
| 0 | 4 bytes | ASCII `SVE2` |
| 4 | uint16 | Version, exactly 2 |
| 6 | uint16 | Header length, exactly 64 |
| 8 | uint32 | Flags, exactly 0 |
| 12 | uint32 | Shared embedding width |
| 16 | uint64 | Token count |
| 24 | uint64 | Visual span count |
| 32 | uint64 | Total visual rows |
| 40 | uint64 | Position bytes, exactly `rows * 12` |
| 48 | uint64 | Embedding bytes, exactly `rows * width * 4` |
| 56 | uint64 | Total bytes, including header and all sections |

Tokens are signed int32 IDs. Text, timestamps, delimiters and visual pad IDs all appear in this sequence.
A zero-span bundle is allowed, including an empty token sequence; it still has a positive embedding width.

### Span descriptor: 64 bytes each

| Offset | Type | Meaning |
|---:|---|---|
| 0 | uint64 | Start cell in the token sequence |
| 8 | uint64 | Row count, positive |
| 16 | uint32 | Kind: image 1, video 2 |
| 20 | int32 | Pad token ID |
| 24 | uint64 | Position advance after the span |
| 32 | uint32 | Image grid width `nx`; zero for video |
| 36 | uint32 | Image grid height `ny`; zero for video |
| 40 | uint64 | Byte offset into the position section |
| 48 | uint64 | Byte offset into the embedding section |
| 56 | uint64 | Reserved, exactly 0 |

Spans are ordered and nonoverlapping. Adjacent spans are allowed. Every covered token must equal that span's
pad ID. Section offsets are canonical cumulative offsets: the first is zero, each next offset follows the
previous span's payload, and all section bytes are used exactly once. There are no holes, aliases or unused rows.

Each row has three signed int32 relative coordinates `(t,h,w)`, followed in the separate embedding section by
`width` float32 values. Coordinates are nonnegative. The producer supplies verified coordinates and advances;
this contract does not invent a relationship between frame number, FPS and rotary time.

Images retain the rectangular SVE1 positions and advance. Video positions are explicit, with zero grid fields.
For either kind, the advance must exceed every relative coordinate. Absolute coordinates and following text
must also fit the caller's position limit. The generic fixture deliberately includes a nonzero relative time;
it is a transport test, not a claim about Qwen's native frame positions.

## Validation and ownership

The default codec ceilings are 1,048,576 tokens, 128 spans, 16,384 visual rows, width 16,384, and 256 MiB of wire
bytes. These are host-codec ceilings, not approved serving defaults or model limits. The caller must supply
`expected_width`, vocabulary size, verified allowed pad IDs, effective context/rotary limits and smaller request
budgets as appropriate. An allowed-pad list is not a model profile: callers must also bind each kind to its
verified pad ID. Source bytes, decoded frames/pixels, retained RGB, disk use and deadlines need separate
limits in the decoder/server; this codec cannot enforce them.

Readers check header counts, checked byte arithmetic and canonical offsets before payload allocation. They
validate token/span/grid/position structure before reading embeddings, then require exact embedding lengths,
finite floats and EOF. C++ float payload I/O is chunked and does not reserve the entire claimed payload before
reading it. Python payload storage is immutable bytes. A truncated file or invalid bundle never returns a
partially usable result. Writers validate the complete bundle before emitting its header; I/O failure may still
leave a partial output, so future encoder/server callers must publish owned files atomically.

The position planner builds `(t,h,w)` for each prompt cell and requested generation-tail cell. Text advances all
three axes by one. A span adds its relative coordinates to the current base, then advances that base by its
explicit advance, not its row count. Token capacity and rotary range are validated separately.

Row bindings are indices `(span,row)`, not pointers into movable payload storage. C++ uses `MediaRow::text` and
Python uses `None` for text/generation cells. A later engine adapter may bind pointers only after final storage
is stable, and must compare bundle token IDs to the actual generation request before touching live state.

## Cache fingerprints

Image fingerprints preserve the existing FNV seed `1469598103934665603` and byte order: three little-endian
signed 64-bit values `{rows,nx,ny}`, then embedding bytes, including the sign bit of zero. Video fingerprints
add `SVE2`, width, kind, pad ID, start cell, row count, advance and all relative positions before embeddings.
These are non-cryptographic cache aids, not authenticity checks. Cache callers still need exact token/prefix
binding, model/preprocessing identity and span metadata comparisons. Decoder cache identity is a separate task.

## Host tests

The shared hand-built hex fixture is `tools/vision/fixtures/media-v2.hex`; its decoded SHA256 is
`35becf7b88dfdb0dc6d1c76a0afe2c8e073052f36303e89730f53982a5b8c24a`. It contains text, one image and one video span
with synthetic width-2 embeddings. It includes negative zero and is not an encoder/model correctness fixture.

Build in a separate directory, with no GPU or model:

```sh
cmake -S . -B /tmp/strata-media-host \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_MEDIA_TESTS=ON -DSTRATA_WERROR=ON
cmake --build /tmp/strata-media-host --target media_embeddings_test -j4
ctest --test-dir /tmp/strata-media-host -R '^media_embeddings_test$' --output-on-failure
MEDIA_TEST_EXE=/tmp/strata-media-host/media_embeddings_test \
  python -m unittest serve.test_media -v
```

Python/C++ tests share bytes, legacy adaptation, fingerprints, positions, truncations and seeded mutations.
Without `MEDIA_TEST_EXE`, the cross-language tests are unrun, not passed. Run the standalone C++ test under
ASan/UBSan as well. None of these results establishes decoder, projector, inference or backend parity.
