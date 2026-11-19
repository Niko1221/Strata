# Pinned dialect fixtures

`dialects-v1.bin` is an 86,204-byte deterministic selection of the independent
C oracle outputs. SHA256 is
`326ec8f6e7c17b626a1b7c13845e5b547703f7ec2568a0b4f69b64964ae1d755`.
The regular checked-ABI test reads this file without external dependencies.
`source-manifest.json` identifies the three clean pinned sources and generated
oracle translation units. `full-fixture-manifest.json` records all 4,240 cases,
input/output hashes, and the measured Windows oracle executable hashes.

Each little-endian record holds `u32 dialect, u32 mode, u32 columns, f32 norm`,
then `f32 input[columns]`, `u32 payload_bytes`, payload bytes and reconstructed
`f32 output[columns]`. Mode 0 calls the original encode wrapper, mode 1 probes
the original scalar packing with a harness-injected rotated vector, and mode 2
decodes arbitrary payloads. Encoding bytes and decoded F32 bits must match
exactly, including signed zero. A source-produced nonfinite FP16 scale is
outside the checked API contract: encoding must return RANGE without writes,
and decoding must return NONFINITE. This exception was declared before the
first oracle comparison. No equality tolerance was relaxed.

To regenerate, use existing local checkouts at the manifest's exact revisions.
Set `DEPS` and `EVIDENCE` to suitable local directories. No downloads occur.

```powershell
uv run --offline python tools/verify_codec_dialects.py --prepare --output "$EVIDENCE/oracles" --thetom "$DEPS/thetom-64395689" --prism "$DEPS/prism-73bfe6cc" --native "$DEPS/ggml-3cf03257"
cmake -S "$EVIDENCE/oracles" -B "$EVIDENCE/oracles/build" -G "Visual Studio 17 2022" -A x64
cmake --build "$EVIDENCE/oracles/build" --config Release --parallel 1
uv run --offline python tools/verify_codec_dialects.py --fixtures --output "$EVIDENCE/oracles" --thetom "$DEPS/thetom-64395689" --prism "$DEPS/prism-73bfe6cc" --native "$DEPS/ggml-3cf03257"
uv run --offline python tools/verify_codec_dialects.py --golden tqcodec/tests/fixtures/dialects-v1.bin --output "$EVIDENCE/oracles" --thetom "$DEPS/thetom-64395689" --prism "$DEPS/prism-73bfe6cc" --native "$DEPS/ggml-3cf03257"
cmake -S tqcodec -B "$EVIDENCE/codec" -DTQCODEC_DIALECTS=ON -DTQCODEC_SPIN8_TESTS=ON -DTQCODEC_ORACLE_FIXTURES="$EVIDENCE/oracles"
cmake --build "$EVIDENCE/codec" --config Release --parallel 1
ctest --test-dir "$EVIDENCE/codec" -C Release --output-on-failure
```

The source generator uses original GGML headers and original C quantization
functions. It does not include or call the new implementation. TheTom's
transform-entry hook is enabled only for scalar mode. Encoder packing and norm
correction remain the original source; ordinary wrapper fixtures leave that
hook disabled. Fixed seed 1234 covers zeros, signs, lanes, multiple blocks,
random vectors, half boundaries and centroid threshold ties. The C++ runner
also repeats wrapper cases in padded non-contiguous rows.

`verify_codec_dialect_faults.py` generates six isolated wrong implementations.
Their CTest cases use `WILL_FAIL` because a normal nonzero test result is the
required result; crashes do not count as passing negative controls. These
sources are test evidence, not runtime implementations.
