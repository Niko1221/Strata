# Visual embedding transport

This is the host contract for the video implementation in progress. Encoder/server/engine adapters now use
it only with explicit video opt-in and matching capabilities; native inference remains unvalidated.

Implementations: `serve/media.py` and `include/strata/program/media_embeddings.hpp`, with the C++ definitions in
`src/program/media_embeddings.cpp`. Neither implementation requires a model, decoder, server or GPU.

## Frame sampling

Frames are chosen on a fixed time grid: 0, 1/fps, 2/fps ... seconds from the first frame, and for each point the
source frame whose presentation timestamp is nearest is decoded (the earlier frame wins an exact tie, and a point
that would repeat the previous frame is dropped). Nothing in the choice uses a reported frame rate, so a
variable-frame-rate clip - a phone recording, a screen capture - is sampled by the time its frames actually show.
No transcode to constant frame rate is done or required.

Each emitted frame keeps the timestamp of the frame that was chosen, and a temporal group is labelled with the mean
of its two frames' times, as the pinned processor does. Short constant-rate test clips reproduce the pinned
processor's frame count; long clips can exceed that processor's configurable 768-frame default. Which frame the
grid lands on can differ by one where time-based and index-based rules disagree. `sample_indices()` keeps the
pinned index-linspace rule for the trace fixtures and is no longer the serving path.

`max_frames` counts unique frames emitted after repeated PTS selections are removed, not grid points. A clip that
selects too many frames is rejected without reducing its cadence. The grid itself is bounded to 86,401 points by
the supported 3,600 s / 24 FPS ceilings. Duration has its own limit: a sparse VFR clip may use more grid points
than emitted frames. At the default 2 FPS, 1,024 densely sampled frames cover about 512 s.

## Compatibility

Existing image requests still use their existing `ENC` / `SVE1` / `GENI` path. Nothing in that path has been
replaced. The legacy adapter is used for mixed image/video requests, not to change image-only requests.

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
bytes. These generic defaults are not video serving defaults or model limits. The video caller explicitly raises
span/row/wire allowances to its configured budgets; the engine bounds SVE2 requests to 65,536 rows and 768 MiB
of wire bytes. The encoder advertises its ceilings in `CAPS` and accepts matching budgets through `VSET`.
The caller must supply `expected_width`, vocabulary size, verified allowed pad IDs, effective context/rotary limits
and smaller request budgets as appropriate. An allowed-pad list is not a model profile: callers must also bind
each kind to its verified pad ID. Source bytes, decoded frames/pixels, retained RGB, disk use and deadlines need separate
limits in the decoder/server; this codec cannot enforce them.

The opt-in serving defaults are 1,024 selected frames, 600 s clip duration, 65,536 visual rows, 256 MiB retained
RGB, 512 MiB wire, 4 GiB streamed decoder output, 256 MiB source, 2 GiB shared disk and a 600 s request deadline.
The configured frame ceiling is 4,096; it does not relax the row, RGB, wire, decoder-output, disk, duration or
request deadline budgets. `ClipInfo` calculates rows, RGB, decoder output and a conservative wire reservation from
the actual selected indices and resized dimensions before decoding. The disk quota covers source, RGB spool,
encoder output, request artifact and completed cache files, including concurrent requests. At their default
individual maxima, source + RGB + two wire artifacts + cache total at most 1,792 MiB against a 2 GiB disk quota;
requests can also be rejected when the shared quota is busy. HTTP request-body limits independently restrict large
base64 data URLs. These are safety budgets, not model context or video-quality guarantees.

Readers check header counts, checked byte arithmetic and canonical offsets before payload allocation. They
validate token/span/grid/position structure before reading embeddings, then require exact embedding lengths,
finite floats and EOF. C++ float payload I/O is chunked and does not reserve the entire claimed payload before
reading it. Python payload storage is immutable bytes. A truncated file or invalid bundle never returns a
partially usable result. Writers validate the complete bundle before emitting its header; I/O failure may still
leave a partial output; encoder/server adapters publish owned files atomically.

The position planner builds `(t,h,w)` for each prompt cell and requested generation-tail cell. Text advances all
three axes by one. A span adds its relative coordinates to the current base, then advances that base by its
explicit advance, not its row count. Token capacity and rotary range are validated separately.

Row bindings are indices `(span,row)`, not pointers into movable payload storage. C++ uses `MediaRow::text` and
Python uses `None` for text/generation cells. The engine adapter binds pointers only after final storage
is stable. C++ `read_media(..., true, &request_tokens)` compares token count/content immediately after token
reads, before descriptors, positions or embeddings. It then applies the Qwen profile before embedding reads.
Python `read_bundle(..., qwen4=True)` applies the same profile before reading embeddings.

### Qwen4 profile, separate from generic transport

The candidate `qwen4_exp_16x2x2_2560_v1` profile requires width 2560, image pad 248056 and video pad 248057.
Every visual pad must belong to a span. Every span has immediate vision-start/end tokens 248053/248054, relative
time zero, a rectangular row-major spatial grid, and advance equal to its largest spatial dimension. Generic
SVE2 remains broader: the width-2/nonzero-time fixture below is deliberately not a Qwen profile fixture.
Ordered whole-slot splicing replaces complete image/video wrappers and rebases spans; it rejects missing,
extra, mismatched or unbound slots. These checks do not prove that embeddings came from the correct projector.

## Cache fingerprints

Image fingerprints preserve the existing FNV seed `1469598103934665603` and byte order: three little-endian
signed 64-bit values `{rows,nx,ny}`, then embedding bytes, including the sign bit of zero. Video fingerprints
add `SVE2`, width, kind, pad ID, start cell, row count, advance and all relative positions before embeddings.
These are non-cryptographic cache aids, not authenticity checks. Cache callers still need exact token/prefix
binding, model/preprocessing identity and span metadata comparisons. `media_fingerprints()` validates the
complete C++ bundle once rather than rescanning its payload per span. The server's decoder/encoder cache has
separate content identity, byte limits and read ownership; it is not secured by FNV.

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

What the video path has been measured doing, on which build and against which reference, is written up in
[VIDEO_VALIDATION.md](VIDEO_VALIDATION.md): frames chosen by PTS and the pixels they decode to, encoder rows
against the checkpoint's visual tower, repeat encodes, positions and prompt tokens, and requests to a running server.
