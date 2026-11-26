"""The two `.strata` header layouts in the Python readers: the 4 KiB layout of every file up to 0.2.22 and the 16 KiB-aligned layout of
the re-adjusted Inkling SQ-20 and Vision (section table at 0x20, u16 tile_layout_id at 0x10, sections 16 KiB aligned).

Readers covered: tools/inkling_tokenizer.read_manifest, tools/hy4_tokenizer.read_manifest (what preview, adopt and the server's
tokenizer extraction call) and tools/squant/strata_format.parse_header_frame / verify_strata_file (the squant tools' reader).
The golden frames in tests/data/strata_headers are the first 448 bytes of the real files (old SQ-20 and Vision of 0.2.22, new SQ-20 and
Vision); the synthetic files are written here (no model, no download, no GPU).  The C++ twin of this test is
engine/tests/quant/strata_reader_layout_test.cpp.

    python -m unittest tools.test_strata_header_layouts -v
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import hy4_tokenizer as HT  # noqa: E402
import inkling_tokenizer as IT  # noqa: E402

try:  # the squant tools live beside terminal/ and need numpy and zstandard
    sys.path.insert(0, str(ROOT.parent / "tools"))
    from squant import strata_format as SF  # noqa: E402
except Exception:  # pragma: no cover - a Python without those modules skips the squant tests
    SF = None

DATA = ROOT / "tests" / "data" / "strata_headers"
ENTRY = struct.Struct("<QQ32sIIQ")


def _has_zstd() -> bool:
    try:
        from compression import zstd  # noqa: F401
        return True
    except ImportError:
        try:
            import zstandard  # noqa: F401
            return True
        except ImportError:
            return False


def zstd_raw_frame(text: bytes) -> bytes:
    """A zstd frame holding `text` as one raw block (single segment, 4-byte content size): no compressor needed to make it."""
    return struct.pack("<IBI", 0xFD2FB528, 0xA0, len(text)) + struct.pack("<I", 1 | (len(text) << 3))[:3] + text


def align(n: int, page: int) -> int:
    return (n + page - 1) // page * page


def synthetic(k16: bool, manifest: dict, profile: int = 0, tile_id: int = 7) -> bytes:
    """A small .strata file: header frame, Section 1 (100 B), Section 3 (24 entries, padded in the 16 KiB layout), Section 0 last."""
    page = 16384 if k16 else 4096
    man = zstd_raw_frame(json.dumps(manifest).encode())
    s1 = (page, 100)
    s3 = (align(s1[0] + s1[1], page), align(24 * 34, page) if k16 else 24 * 34)
    s0 = (align(s3[0] + s3[1], page), len(man))
    size = align(s0[0] + s0[1], page)
    f = bytearray(size)
    f[0:8] = b"STRATA\x00\x01"
    struct.pack_into("<IHH", f, 8, 1, profile, 6)
    table = 0x20 if k16 else 0x10
    if k16:
        struct.pack_into("<H", f, 0x10, tile_id)
    spans = {0: s0, 1: s1, 3: s3}
    for i in range(6):
        off, ln = spans.get(i, (0, 0))
        f[table + 64 * i: table + 64 * (i + 1)] = ENTRY.pack(off, ln, bytes([i]) * 32, i, 1 if i == 3 else 0, 0)
    f[s0[0]: s0[0] + len(man)] = man
    return bytes(f)


MANIFEST = {"format": "strata-v1.1", "arch": "inkling2_loop", "config": {"num_physical_layers": 99}, "chat_template": "x",
            "num_physical_layers": 4, "num_experts_per_layer": 6, "build": "SQ-TEST"}


def golden(name: str) -> bytes:
    return (DATA / name).read_bytes().ljust(4096, b"\0")


@unittest.skipUnless(_has_zstd(), "this Python has no zstd module")
class ReadManifestTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def write(self, name: str, data: bytes) -> Path:
        p = self.dir / name
        p.write_bytes(data)
        return p

    def each(self):
        return (("inkling_tokenizer", IT.read_manifest), ("hy4_tokenizer", HT.read_manifest))

    def test_4k_file_reads_as_before(self):
        p = self.write("old.strata", synthetic(False, MANIFEST))
        for name, rm in self.each():
            with self.subTest(name):
                self.assertEqual(rm(p), MANIFEST)

    def test_16k_file_reads(self):
        p = self.write("new.strata", synthetic(True, MANIFEST, profile=1))
        for name, rm in self.each():
            with self.subTest(name):
                self.assertEqual(rm(p), MANIFEST)

    def test_16k_header_with_an_old_looking_u64_at_0x10_is_still_16k(self):
        data = bytearray(synthetic(True, MANIFEST))
        struct.pack_into("<QQ", data, 0x10, 1 << 20, 4096)  # a 4 KiB reader would take this for "Section 0 at 1 MiB"
        p = self.write("ext.strata", bytes(data))
        for name, rm in self.each():
            with self.subTest(name):
                self.assertEqual(rm(p), MANIFEST)

    def test_not_a_strata_file(self):
        p = self.write("x.strata", b"GGUF" + bytes(8192))
        for name, rm in self.each():
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, "not a .strata file"):
                    rm(p)

    def test_headers_that_fit_neither_layout_are_refused_not_misread(self):
        good = synthetic(True, MANIFEST)
        cases = {}
        cut = bytearray(good)
        cut[0x20 + 64 + 48: 0x20 + 64 + 52] = struct.pack("<I", 9)                      # wrong section_id in a 16 KiB table
        cases["bad section_id"] = bytes(cut)
        cases["cut file"] = good[: len(good) - 20000]                                   # Section 0 (last) is gone
        z = bytearray(good)
        z[0x10: 0x1C0] = bytes(0x1B0)                                                    # no section table at all
        cases["empty table"] = bytes(z)
        for label, data in cases.items():
            p = self.write("bad.strata", data)
            for name, rm in self.each():
                with self.subTest(label, reader=name):
                    with self.assertRaises(ValueError):
                        rm(p)

    def test_a_table_that_reads_as_both_layouts_is_refused(self):
        f = bytearray(32768)
        f[0:8] = b"STRATA\x00\x01"
        struct.pack_into("<IHH", f, 8, 1, 0, 6)
        for i in range(6):
            struct.pack_into("<I", f, 0x10 + i * 64 + 48, i)
            struct.pack_into("<I", f, 0x20 + i * 64 + 48, i)
        for i in range(1, 6):
            struct.pack_into("<I", f, 0x10 + i * 64, i - 1)   # A_i offset low word = B_(i-1) section_id, length 0
        struct.pack_into("<QQ", f, 0x10, 65536 // 2, 100)     # A_0: 32 KiB, 100 B (4 KiB aligned, inside the 32 KiB test file? it is not: widen)
        f = bytearray(f.ljust(1 << 20))
        struct.pack_into("<QQ", f, 0x20, 16384, 100)           # B_0
        p = self.write("both.strata", bytes(f))
        for name, rm in self.each():
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, "ambiguous"):
                    rm(p)

    def test_the_two_tokenizer_readers_agree_with_each_other(self):
        for k16 in (False, True):
            p = self.write("a.strata", synthetic(k16, MANIFEST))
            self.assertEqual(IT.read_manifest(p), HT.read_manifest(p))


class SectionZeroTests(unittest.TestCase):
    """_section0 on synthetic files: needs no zstd module, so it runs everywhere (ReadManifestTests above need one to decode the manifest)."""

    def locate(self, data: bytes):
        r = []
        for mod in (IT, HT):
            r.append(mod._section0(data[:4096], len(data)))
        self.assertEqual(r[0], r[1])
        return r[0]

    def test_4k_file(self):
        data = synthetic(False, MANIFEST)
        off, ln = self.locate(data)
        self.assertEqual(off % 4096, 0)
        self.assertEqual(data[off: off + 4], struct.pack("<I", 0xFD2FB528))

    def test_16k_file(self):
        data = synthetic(True, MANIFEST, profile=1)
        off, ln = self.locate(data)
        self.assertEqual(off % 16384, 0)
        self.assertEqual(data[off: off + 4], struct.pack("<I", 0xFD2FB528))
        self.assertEqual(ln, len(zstd_raw_frame(json.dumps(MANIFEST).encode())))

    def test_16k_header_with_an_old_looking_u64_at_0x10_is_still_16k(self):
        data = bytearray(synthetic(True, MANIFEST))
        struct.pack_into("<QQ", data, 0x10, 1 << 20, 4096)
        off, ln = self.locate(bytes(data))
        self.assertEqual(data[off: off + 4], struct.pack("<I", 0xFD2FB528))

    def test_refusals(self):
        good = synthetic(True, MANIFEST)
        bad_id = bytearray(good)
        bad_id[0x20 + 64 + 48: 0x20 + 64 + 52] = struct.pack("<I", 9)
        z = bytearray(good)
        z[0x10: 0x1C0] = bytes(0x1B0)
        for label, data in (("bad section_id", bytes(bad_id)), ("cut file", good[: len(good) - 20000]), ("empty table", bytes(z))):
            for mod in (IT, HT):
                with self.subTest(label, reader=mod.__name__):
                    with self.assertRaises(ValueError):
                        mod._section0(data[:4096], len(data))

    def test_ambiguous(self):
        f = bytearray(1 << 20)
        f[0:8] = b"STRATA\x00\x01"
        for i in range(6):
            struct.pack_into("<I", f, 0x10 + i * 64 + 48, i)
            struct.pack_into("<I", f, 0x20 + i * 64 + 48, i)
        for i in range(1, 6):
            struct.pack_into("<I", f, 0x10 + i * 64, i - 1)
        struct.pack_into("<QQ", f, 0x10, 65536, 100)
        struct.pack_into("<QQ", f, 0x20, 16384, 100)
        for mod in (IT, HT):
            with self.subTest(mod.__name__):
                with self.assertRaisesRegex(ValueError, "ambiguous"):
                    mod._section0(bytes(f[:4096]), len(f))


class GoldenHeaderTests(unittest.TestCase):
    """The real headers (448 bytes each), padded to 4 KiB; the file sizes are the real ones."""

    CASES = {
        # name: (file size, profile, layout table at, section 0 (offset, length), section 1 (offset, length), section 3 length)
        "sq20-old.hdr": (19346685952, 0, 0x10, (19346673664, 10375), (4096, 3670344256), 104448),
        "vision-old.hdr": (636526592, 3, 0x10, (636518400, 4097), (4096, 636512256), 0),
        "sq20-new.hdr": (19374768128, 1, 0x20, (19374751744, 10699), (16384, 3671130112), 114688),
        "vision-new.hdr": (636551168, 2, 0x20, (636534784, 4102), (16384, 636512256), 0),
    }

    def test_section0_of_every_real_header(self):
        for name, (size, _prof, _table, s0, _s1, _s3) in self.CASES.items():
            with self.subTest(name):
                self.assertEqual(IT._section0(golden(name), size), s0)
                self.assertEqual(HT._section0(golden(name), size), s0)
                self.assertEqual(IT._section0(golden(name), 0), s0)   # file size unknown

    @unittest.skipIf(SF is None, "tools/squant needs numpy and zstandard")
    def test_squant_parse_header_frame_on_every_real_header(self):
        for name, (size, prof, table, s0, s1, s3) in self.CASES.items():
            with self.subTest(name):
                flags, profile, secs, engram = SF.parse_header_frame(golden(name), size)
                self.assertEqual((flags, profile), (1, prof))
                self.assertEqual((secs[0].file_offset, secs[0].byte_length), s0)
                self.assertEqual((secs[1].file_offset, secs[1].byte_length), s1)
                self.assertEqual(secs[3].byte_length, s3)
                self.assertEqual([s.section_id for s in secs], list(range(6)))
                self.assertEqual(engram, bytes(32))
                self.assertEqual(SF.detect_layout(golden(name), size), table)

    def test_a_truncated_head_of_a_real_file_is_not_taken_for_the_other_layout(self):
        # the first 4 MiB of the new SQ-20 (Section 0 is at the end): with the real size it is the 16 KiB layout, and without it too
        for size in (0, 19374768128):
            self.assertEqual(IT._section0(golden("sq20-new.hdr"), size), (19374751744, 10699))


@unittest.skipIf(SF is None, "tools/squant needs numpy and zstandard")
class SquantTests(unittest.TestCase):
    def sections(self, page: int, k16: bool):
        return [SF.StrataSectionEntry(0 if i in (2, 4, 5) else page * (i + 1), 0 if i in (2, 4, 5) else 100, hashlib.sha256(bytes([i])).digest(), i, 1 if i == 3 else 0)
                for i in range(6)]

    def test_writer_and_reader_round_trip_in_both_layouts(self):
        for k16 in (False, True):
            with self.subTest(k16=k16):
                secs = self.sections(16384 if k16 else 4096, k16)
                frame = SF.build_header_frame(1, secs, b"\x07" * 32, flags=1, layout_16k=k16, tile_layout_id=3)
                self.assertEqual(SF.detect_layout(frame), 0x20 if k16 else 0x10)
                flags, profile, back, engram = SF.parse_header_frame(frame)
                self.assertEqual((flags, profile, engram), (1, 1, b"\x07" * 32))
                self.assertEqual([(s.file_offset, s.byte_length, s.section_id, s.flags) for s in back], [(s.file_offset, s.byte_length, s.section_id, s.flags) for s in secs])
                if k16:
                    self.assertEqual(struct.unpack_from("<H", frame, 0x10)[0], 3)

    def test_the_4k_writer_output_is_what_it_was(self):
        frame = SF.build_header_frame(0, self.sections(4096, False))
        self.assertEqual(frame[0x10:0x50], SF.StrataSectionEntry(4096, 100, hashlib.sha256(bytes([0])).digest(), 0).pack())
        self.assertEqual(len(frame), 4096)

    def test_an_all_zero_table_is_the_4k_layout_as_before(self):
        frame = bytearray(4096)
        frame[0:8] = b"STRATA\x00\x01"
        struct.pack_into("<IHH", frame, 8, 0, 0, 6)
        flags, profile, secs, engram = SF.parse_header_frame(bytes(frame))
        self.assertEqual([s.byte_length for s in secs], [0] * 6)

    def test_verify_strata_file_checks_the_sha256_of_a_16k_file(self):
        with tempfile.TemporaryDirectory() as d:
            data = bytearray(synthetic(True, MANIFEST, profile=1))
            for i in range(6):  # the real section sha256s, as the writer records them
                off, ln, _sha, _sid, _fl, _res = ENTRY.unpack_from(data, 0x20 + 64 * i)
                ENTRY.pack_into(data, 0x20 + 64 * i, off, ln, hashlib.sha256(bytes(data[off: off + ln])).digest(), i, 1 if i == 3 else 0, 0)
            p = Path(d) / "m.strata"
            p.write_bytes(bytes(data))
            res = SF.verify_strata_file(str(p))
            self.assertEqual(res["manifest"], MANIFEST)
            self.assertEqual([s["offset"] for s in res["sections"]][1], 16384)
            data[16384 + 5] ^= 1  # a flipped bit in Section 1
            p.write_bytes(bytes(data))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch in section 1"):
                SF.verify_strata_file(str(p))


if __name__ == "__main__":
    unittest.main()
