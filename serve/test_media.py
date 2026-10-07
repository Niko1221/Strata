"""The media wire contract, no model/GPU/decoder. MEDIA_TEST_EXE adds C++ parity."""
import dataclasses
import hashlib
import io
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

from serve.media import (DEFAULT_LIMITS, HEADER, SPAN, LegacyImage, MediaBundle, MediaError, MediaKind,
                         MediaLimits, VisualSpan, adapt_legacy_images, build_positions, decode, encode,
                         read_bundle, read_legacy_images, span_fingerprint, validate, write_bundle,
                         write_legacy_images)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tools/vision/fixtures/media-v2.hex"
IMAGE_PAD, VIDEO_PAD = 248056, 248057


def floats(*values):
    return struct.pack(f"<{len(values)}f", *values)


def sample():
    return MediaBundle(2, (11, 20, IMAGE_PAD, IMAGE_PAD, 21, VIDEO_PAD, VIDEO_PAD, 22), (
        VisualSpan(2, IMAGE_PAD, MediaKind.IMAGE, 2, ((0, 0, 0), (0, 0, 1)), floats(.25, -0., .5, -1.), 2, 1),
        VisualSpan(5, VIDEO_PAD, MediaKind.VIDEO, 4, ((0, 0, 0), (3, 1, 2)), floats(2., 3., 4., 5.)),
    ))


def replace_span(bundle, index=0, **kwargs):
    spans = list(bundle.spans)
    spans[index] = dataclasses.replace(spans[index], **kwargs)
    return dataclasses.replace(bundle, spans=tuple(spans))


def legacy_bytes(images):
    out = io.BytesIO()
    write_legacy_images(out, images)
    return out.getvalue()


class MediaContract(unittest.TestCase):
    def test_golden_and_float_bits(self):
        raw = bytes.fromhex(GOLDEN.read_text())
        self.assertEqual(encode(sample()), raw)
        self.assertEqual(decode(raw), sample())
        self.assertEqual(HEADER.size, 64)
        self.assertEqual(SPAN.size, 64)
        self.assertEqual(encode(decode(raw)), raw)
        self.assertEqual(sample().spans[0].embeddings[4:8], b"\0\0\0\x80")
        # This locks the shared fixture as well as the code against accidental format changes.
        self.assertEqual(hashlib.sha256(raw).hexdigest(), GOLDEN_SHA256)

    def test_positions_and_generation_tail(self):
        plan = build_positions(sample(), 10)
        self.assertEqual(plan.positions, ((0, 0, 0), (1, 1, 1), (2, 2, 2), (2, 2, 3),
                                         (4, 4, 4), (5, 5, 5), (8, 6, 7), (9, 9, 9), (10, 10, 10), (11, 11, 11)))
        self.assertEqual(plan.rows, (None, None, (0, 0), (0, 1), None, (1, 0), (1, 1), None, None, None))
        self.assertEqual(build_positions(sample(), 8).positions, plan.positions[:8])
        self.assertEqual(build_positions(MediaBundle(1, (), ()), 2).positions, ((0, 0, 0), (1, 1, 1)))
        with self.assertRaises(MediaError):
            build_positions(sample(), 7)
        with self.assertRaises(MediaError):
            build_positions(sample(), DEFAULT_LIMITS.max_tokens + 1)
        with self.assertRaises(MediaError):
            build_positions(sample(), 10, dataclasses.replace(DEFAULT_LIMITS, max_position=10))

    def test_all_truncations_and_trailing_bytes(self):
        raw = encode(sample())
        for cut in range(len(raw)):
            with self.subTest(cut=cut), self.assertRaises(MediaError):
                decode(raw[:cut])
        for extra in (b"\0", raw, b"SVE1"):
            with self.assertRaises(MediaError):
                decode(raw + extra)

    def test_header_fields_and_canonical_descriptors(self):
        raw = encode(sample())
        bad_fields = [(0, "4s", b"SVE1"), (4, "H", 1), (6, "H", 65), (8, "I", 1),
                      (12, "I", 0), (16, "Q", 1 << 63), (24, "Q", 0), (32, "Q", 0),
                      (40, "Q", 47), (48, "Q", 31), (56, "Q", len(raw) - 1)]
        first = HEADER.size + len(sample().tokens) * 4
        bad_fields += [(first, "Q", 9), (first + 8, "Q", 0), (first + 16, "I", 3),
                       (first + 20, "i", -1), (first + 24, "Q", 0), (first + 32, "I", 0),
                       (first + 40, "Q", 12), (first + 48, "Q", 4), (first + 56, "Q", 1),
                       (first + SPAN.size, "Q", 3)]
        for offset, fmt, value in bad_fields:
            changed = bytearray(raw)
            struct.pack_into("<" + fmt, changed, offset, value)
            with self.subTest(offset=offset, value=value), self.assertRaises(MediaError):
                decode(bytes(changed))

    def test_semantic_mutations(self):
        b = sample()
        bad = [dataclasses.replace(b, width=0), dataclasses.replace(b, width=3),
               dataclasses.replace(b, tokens=(True,) + b.tokens[1:]),
               dataclasses.replace(b, tokens=(-1,) + b.tokens[1:]),
               dataclasses.replace(b, tokens=b.tokens[:2] + (21,) + b.tokens[3:]),
               replace_span(b, positions=()), replace_span(b, start=8), replace_span(b, start=True),
               replace_span(b, kind=3), replace_span(b, kind=1), replace_span(b, nx=1),
               replace_span(b, positions=((0, 0, 0), (0, 1, 0))),
               replace_span(b, embeddings=b"x"), replace_span(b, embeddings=floats(.25, 0, float("nan"), 0)),
               replace_span(b, embeddings=floats(float("inf"), 0, 0, 0)),
               replace_span(b, 1, positions=((0, 0, 0), (-1, 1, 2))),
               replace_span(b, 1, positions=((0, 0, 0), (1 << 31, 1, 2))),
               replace_span(b, 1, advance=3), replace_span(b, 1, nx=2, ny=1),
               replace_span(b, 1, start=3), replace_span(b, 1, pad_id=IMAGE_PAD)]
        for bundle in bad:
            with self.subTest(bundle=bundle), self.assertRaises(MediaError):
                encode(bundle)

    def test_limits_and_binding(self):
        b, raw = sample(), encode(sample())
        for limits in (MediaLimits(max_tokens=7), MediaLimits(max_spans=1), MediaLimits(max_rows=3),
                       MediaLimits(max_width=1), MediaLimits(max_bytes=len(raw) - 1), MediaLimits(max_position=7),
                       MediaLimits(vocab_size=VIDEO_PAD), MediaLimits(expected_width=3),
                       MediaLimits(allowed_pad_ids=(IMAGE_PAD,))):
            for operation in (lambda: validate(b, limits), lambda: decode(raw, limits)):
                with self.subTest(limits=limits), self.assertRaises(MediaError):
                    operation()
        for changes in ({"max_rows": 0}, {"max_bytes": -1}, {"max_tokens": True}, {"max_position": -1},
                        {"vocab_size": 0}, {"vocab_size": (1 << 31) + 1}, {"allowed_pad_ids": (-1,)}):
            with self.subTest(changes=changes), self.assertRaises(MediaError):
                decode(raw, dataclasses.replace(DEFAULT_LIMITS, **changes))
        self.assertEqual(decode(raw, MediaLimits(expected_width=2, vocab_size=248320,
                                               allowed_pad_ids=(IMAGE_PAD, VIDEO_PAD))), b)
        # Invalid count headers are refused without trying to read their claimed payload.
        class HeaderOnly(io.BytesIO):
            def read(self, n=-1):
                self.assertion = n
                if self.tell() >= HEADER.size:
                    raise AssertionError("payload read before header budgets were checked")
                return super().read(n)
        changed = bytearray(raw[:64])
        struct.pack_into("<Q", changed, 16, (1 << 64) - 1)
        with self.assertRaises(MediaError):
            read_bundle(HeaderOnly(changed))

    def test_structure_is_checked_before_embedding_reads(self):
        raw = encode(sample())
        payload_start = HEADER.size + len(sample().tokens) * 4 + 2 * SPAN.size + 4 * 12
        class NoEmbeddings(io.BytesIO):
            def read(self, n=-1):
                if self.tell() >= payload_start:
                    raise AssertionError("embedding read before structure was checked")
                return super().read(n)
        first = HEADER.size + len(sample().tokens) * 4
        for offset, value in ((payload_start - 48, -1), (first + 20, VIDEO_PAD), (first + 24, 3)):
            changed = bytearray(raw[:payload_start])
            struct.pack_into("<i", changed, offset, value)
            with self.subTest(offset=offset), self.assertRaises(MediaError):
                read_bundle(NoEmbeddings(changed))

    def test_empty_and_text_only_bundle(self):
        for bundle in (MediaBundle(1, (), ()), MediaBundle(1, (0, 1, 2), ())):
            self.assertEqual(decode(encode(bundle)), bundle)
        self.assertEqual(build_positions(MediaBundle(1, (0, 1, 2), ()), 5).positions,
                         tuple((i, i, i) for i in range(5)))

    def test_writer_validation_and_short_writes(self):
        out = io.BytesIO()
        with self.assertRaises(MediaError):
            write_bundle(out, replace_span(sample(), advance=0))
        self.assertEqual(out.getvalue(), b"")
        class Short(io.BytesIO):
            def write(self, data):
                return super().write(data[:3])
        short = Short()
        write_bundle(short, sample())
        self.assertEqual(short.getvalue(), encode(sample()))
        class Broken(io.BytesIO):
            def write(self, data):
                return 0
        with self.assertRaises(OSError):
            write_bundle(Broken(), sample())

    def test_fingerprint_covers_video_inputs_and_keeps_legacy_hash(self):
        b = sample()
        self.assertNotEqual(span_fingerprint(b, 1), span_fingerprint(replace_span(b, 1, advance=5), 1))
        self.assertNotEqual(span_fingerprint(b, 1), span_fingerprint(replace_span(b, 1, positions=((0, 0, 0), (3, 2, 2))), 1))
        self.assertNotEqual(span_fingerprint(b, 1), span_fingerprint(replace_span(b, 1, embeddings=floats(2, 3, 4, 6)), 1))
        video = replace_span(b, 1, pad_id=IMAGE_PAD)
        video = dataclasses.replace(video, tokens=tuple(IMAGE_PAD if t == VIDEO_PAD else t for t in video.tokens))
        self.assertNotEqual(span_fingerprint(b, 1), span_fingerprint(video, 1))
        shifted = dataclasses.replace(replace_span(b, 1, start=6), tokens=b.tokens[:5] + (23,) + b.tokens[5:])
        self.assertNotEqual(span_fingerprint(b, 1), span_fingerprint(shifted, 1))
        self.assertEqual(span_fingerprint(b, 0), IMAGE_HASH)

    def test_seeded_mutations_are_rejected_or_canonical(self):
        raw, rng = encode(sample()), random.Random(2031)
        for _ in range(500):
            data = bytearray(raw)
            for _ in range(rng.randrange(1, 5)):
                index = rng.randrange(len(data))
                data[index] ^= 1 << rng.randrange(8)
            try:
                result = decode(bytes(data))
            except MediaError:
                continue
            self.assertEqual(encode(result), bytes(data))


class LegacyImages(unittest.TestCase):
    def test_roundtrip_and_position_equivalence(self):
        images = (LegacyImage(2, 2, 1, floats(.25, -0., .5, -1.)),
                  LegacyImage(2, 1, 2, floats(2, 3, 4, 5)))
        raw = legacy_bytes(images)
        expected = b"SVE1" + struct.pack("<iiii", 2, 2, 1, 2) + images[0].embeddings
        expected += b"SVE1" + struct.pack("<iiii", 2, 1, 2, 2) + images[1].embeddings
        self.assertEqual(raw, expected)
        self.assertEqual(read_legacy_images(io.BytesIO(raw)), images)
        tokens = (11, IMAGE_PAD, IMAGE_PAD, 21, IMAGE_PAD, IMAGE_PAD, 22)
        b = adapt_legacy_images(images, tokens, IMAGE_PAD)
        self.assertEqual(build_positions(b, 9).positions,
                         ((0, 0, 0), (1, 1, 1), (1, 1, 2), (3, 3, 3), (4, 4, 4),
                          (4, 5, 4), (6, 6, 6), (7, 7, 7), (8, 8, 8)))
        self.assertEqual(span_fingerprint(b, 0), IMAGE_HASH)
        self.assertEqual(decode(encode(b)), b)
        self.assertEqual(read_legacy_images(io.BytesIO()), ())

    def test_partial_records_and_invalid_headers(self):
        raw = legacy_bytes((LegacyImage(1, 2, 1, floats(1, 2)),))
        for cut in range(1, len(raw)):
            with self.subTest(cut=cut), self.assertRaises(MediaError):
                read_legacy_images(io.BytesIO(raw[:cut]))
        for data in (raw + b"x", raw.replace(b"SVE1", b"SVE2"),
                     b"SVE1" + struct.pack("<iiii", 2, 2, 2, 1),
                     b"SVE1" + struct.pack("<iiii", -1, -1, 1, 1),
                     b"SVE1" + struct.pack("<iiii", 1, 1, 1, 1) + floats(float("nan"))):
            with self.assertRaises(MediaError):
                read_legacy_images(io.BytesIO(data))
        with self.assertRaises(MediaError):
            read_legacy_images(io.BytesIO(raw * 2), MediaLimits(max_rows=3))
        with self.assertRaises(MediaError):
            read_legacy_images(io.BytesIO(raw), MediaLimits(max_bytes=len(raw) - 1))

    def test_binding_refuses_missing_extra_and_partial_runs(self):
        image = LegacyImage(1, 2, 1, floats(1, 2))
        for images, tokens in (((image,), (1, 2)), ((image,), (1, IMAGE_PAD, 2)),
                               ((image,), (1, IMAGE_PAD, IMAGE_PAD)),
                               ((image, image), (1, IMAGE_PAD, IMAGE_PAD, 2)),
                               ((image,), (1, IMAGE_PAD, IMAGE_PAD, 2, IMAGE_PAD, 2))):
            with self.subTest(images=images, tokens=tokens), self.assertRaises(MediaError):
                adapt_legacy_images(images, tokens, IMAGE_PAD)


@unittest.skipUnless(os.environ.get("MEDIA_TEST_EXE"), "C++ parity job needs MEDIA_TEST_EXE")
class CrossLanguage(unittest.TestCase):
    def invoke(self, mode, data, *args):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            path.write_bytes(data)
            return subprocess.run([os.environ["MEDIA_TEST_EXE"], mode, str(path), *map(str, args)],
                                  capture_output=True, timeout=10)

    def test_roundtrip_positions_and_hashes(self):
        raw = encode(sample())
        result = self.invoke("--roundtrip", raw)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, raw)
        plan = build_positions(sample(), 10)
        result = self.invoke("--positions", raw, 10)
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = []
        for pos, row in zip(plan.positions, plan.rows):
            span, index = row if row is not None else (-1, 0)
            expected.append(f"{pos[0]} {pos[1]} {pos[2]} {span} {index}\n")
        self.assertEqual(result.stdout.decode(), "".join(expected))
        result = self.invoke("--hashes", raw)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(list(map(int, result.stdout.split())), [span_fingerprint(sample(), i) for i in range(2)])

    def test_multiple_sizes_and_zero_spans(self):
        rng = random.Random(82)
        for width in (1, 3, 16):
            count = 7
            positions = tuple((i, i % 3, i % 2) for i in range(count))
            payload = floats(*(rng.uniform(-10, 10) for _ in range(width * count)))
            bundle = MediaBundle(width, (20,) + (VIDEO_PAD,) * count + (21,),
                                 (VisualSpan(1, VIDEO_PAD, MediaKind.VIDEO, 7, positions, payload),))
            result = self.invoke("--roundtrip", encode(bundle))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, encode(bundle))
        for bundle in (MediaBundle(1, (), ()), MediaBundle(1, (1, 2, 3), ())):
            result = self.invoke("--roundtrip", encode(bundle))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, encode(bundle))

    def test_seeded_parser_decisions_and_all_truncations(self):
        raw, rng = encode(sample()), random.Random(770)
        cases = [raw[:cut] for cut in range(len(raw))]
        for _ in range(60):
            data = bytearray(raw)
            data[rng.randrange(len(data))] ^= 1 << rng.randrange(8)
            cases.append(bytes(data))
        cases.append(raw + b"x")
        for data in cases:
            try:
                decode(data)
                accepted = True
            except MediaError:
                accepted = False
            result = self.invoke("--roundtrip", data)
            self.assertEqual(result.returncode == 0, accepted, result.stderr)
            if accepted:
                self.assertEqual(result.stdout, data)

    def test_legacy_cpp_binding(self):
        images = (LegacyImage(2, 2, 1, floats(.25, -0., .5, -1.)),)
        tokens = (11, IMAGE_PAD, IMAGE_PAD, 12)
        result = self.invoke("--legacy", legacy_bytes(images), IMAGE_PAD)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, encode(adapt_legacy_images(images, tokens, IMAGE_PAD)))


# Filled once from the checked-in fixture, never computed from the implementation under test.
GOLDEN_SHA256 = "35becf7b88dfdb0dc6d1c76a0afe2c8e073052f36303e89730f53982a5b8c24a"
IMAGE_HASH = 2114594927496901928


if __name__ == "__main__":
    unittest.main()
