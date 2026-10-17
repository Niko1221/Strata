"""A malformed GGUF header is refused with a ValueError that names the file, never a KeyError, OverflowError,
MemoryError or struct.error, and a shard whose last tensor has no known byte size is not reported whole (#1919).
Over minimal GGUF v3 files written here (no download, no model, no GPU).

    python -m unittest tools.test_gguf_reader_malformed
"""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import setup as S  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402


class Stop(Exception):
    """setup.fail(), caught instead of exiting."""


def gstr(s: bytes) -> bytes:
    return struct.pack("<Q", len(s)) + s


def header(n_tensors: int, kv: bytes = b"", n_kv: int = 0) -> bytes:
    return struct.pack("<IIQQ", 0x46554747, 3, n_tensors, n_kv) + kv


def tensor(name: bytes, n: int, type_id: int, offset: int) -> bytes:
    return gstr(name) + struct.pack("<IQIQ", 1, n, type_id, offset)


def pad(b: bytes, data: int) -> bytes:
    return b + bytes(-len(b) % 32 + data)


class MalformedHeader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "m-00001-of-00002.gguf"

    def tearDown(self):
        self.tmp.cleanup()

    def refused(self, raw: bytes) -> str:
        self.path.write_bytes(raw)
        with self.assertRaises(ValueError) as cm:
            GGUFFile(self.path)
        self.assertIn(self.path.name, str(cm.exception))
        return str(cm.exception)

    def test_unknown_metadata_value_type(self):
        self.refused(pad(header(0, gstr(b"k") + struct.pack("<I", 99), 1), 0))

    def test_unknown_array_element_type(self):
        self.refused(pad(header(0, gstr(b"k") + struct.pack("<IIQ", 9, 99, 1), 1), 0))

    def test_string_length_past_the_end(self):
        self.refused(pad(header(0, struct.pack("<Q", 2 ** 63), 1), 0))

    def test_array_count_past_the_end(self):
        self.refused(pad(header(0, gstr(b"k") + struct.pack("<IIQ", 9, 4, 2 ** 40), 1), 0))

    def test_string_array_count_past_the_end(self):
        self.refused(pad(header(0, gstr(b"k") + struct.pack("<IIQ", 9, 8, 2 ** 62), 1), 0))

    def test_tensor_count_past_the_end(self):
        self.refused(header(2 ** 40))

    def test_header_cut_mid_tensor_directory(self):
        self.refused(header(1) + tensor(b"blk.0.w", 8, 0, 0)[:-5])

    def test_header_cut_inside_the_magic(self):
        self.refused(b"GG")


class LastTensorSize(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        patcher = mock.patch.object(S, "fail", side_effect=lambda msg, hint=None: (_ for _ in ()).throw(
            Stop(msg + " " + (hint or ""))))
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def test_metadata_only_shards_need_no_tensor_alignment_padding(self):
        # A complete metadata directory ends at byte 84; its hypothetical data start is 96.
        kv = gstr(b"general.name") + struct.pack("<I", 8) + gstr(b"m" * 28)
        for raw in (header(0), header(0, kv, 1)):
            for padded in (False, True):
                with self.subTest(header_bytes=len(raw), padded=padded):
                    s = self.dir / "metadata.gguf"
                    s.write_bytes(pad(raw, 0) if padded else raw)
                    self.assertEqual(GGUFFile(s).data_end(), len(raw))
                    self.assertTrue(S.whole_shard(s))
                    S.check_shards([s])
            # Dropping metadata bytes is still a truncated header, not an empty payload.
            s.write_bytes(raw[:-1])
            self.assertFalse(S.whole_shard(s))
            with self.assertRaises(Stop):
                S.check_shards([s])

    def test_unknown_type_last_tensor_truncated_is_not_whole(self):
        # type 4 (Q4_2) has no block geometry: the shard's needed length cannot be computed from its header
        s = self.dir / "m-00001-of-00001.gguf"
        s.write_bytes(pad(header(2) + tensor(b"a", 8, 0, 0) + tensor(b"b", 64, 4, 32), 32))
        self.assertFalse(S.whole_shard(s))
        with self.assertRaises(Stop) as cm:
            S.check_shards([s])
        self.assertIn(s.name, str(cm.exception))

    def test_unknown_type_in_the_middle_is_not_proven_by_next_offset(self):
        s = self.dir / "m-00001-of-00001.gguf"
        s.write_bytes(pad(header(2) + tensor(b"a", 64, 4, 0) + tensor(b"b", 8, 0, 64), 64 + 32))
        self.assertFalse(S.whole_shard(s))
        with self.assertRaises(Stop) as cm:
            S.check_shards([s])
        self.assertIn("cannot verify", str(cm.exception))
        self.assertNotIn("delete", str(cm.exception))

    def test_complete_unknown_last_tensor_is_preserved(self):
        s = self.dir / "complete.gguf"
        raw = pad(header(1) + tensor(b"future", 64, 999, 0), 256)
        s.write_bytes(raw)
        self.assertFalse(S.whole_shard(s))
        with self.assertRaises(Stop) as cm:
            S.check_shards([s])
        self.assertIn("cannot verify", str(cm.exception))
        self.assertNotIn("delete", str(cm.exception))
        self.assertEqual(s.read_bytes(), raw)
        self.assertFalse(s.with_name(s.name + ".done").exists())

    def test_all_known_extents_checked_even_when_offsets_are_unsorted(self):
        s = self.dir / "overrun.gguf"
        s.write_bytes(pad(header(2) + tensor(b"large", 100, 0, 0) + tensor(b"last", 8, 0, 64), 96))
        self.assertEqual(GGUFFile(s).data_end(), GGUFFile(s).data_start + 400)
        self.assertFalse(S.whole_shard(s))
        with self.assertRaises(Stop):
            S.check_shards([s])

    def test_known_overrun_not_hidden_by_unknown_type(self):
        s = self.dir / "mixed.gguf"
        s.write_bytes(pad(header(2) + tensor(b"large", 100, 0, 0) + tensor(b"unknown", 8, 999, 64), 96))
        self.assertFalse(S.whole_shard(s))
        with self.assertRaises(Stop) as cm:
            S.check_shards([s])
        self.assertIn("requires at least", str(cm.exception))

    def test_integer_and_ternary_geometries(self):
        for type_id, count, size in ((24, 8, 8), (25, 8, 16), (26, 8, 32), (27, 8, 64),
                                     (34, 256, 54), (35, 256, 66)):
            with self.subTest(type_id=type_id):
                s = self.dir / "valid.gguf"
                s.write_bytes(pad(header(1) + tensor(b"weight", count, type_id, 0), size))
                self.assertTrue(S.whole_shard(s))
                S.check_shards([s])
                self.assertEqual(GGUFFile(s).data_end(), s.stat().st_size)
                s.write_bytes(s.read_bytes()[:-1])
                self.assertFalse(S.whole_shard(s))

    def test_repacked_types_are_unverifiable_not_corrupt(self):
        for type_id in (31, 32, 33, 36, 37, 38, 41):
            with self.subTest(type_id=type_id):
                s = self.dir / "repacked.gguf"
                s.write_bytes(pad(header(1) + tensor(b"weight", 256, type_id, 0), 256))
                with self.assertRaises(Stop) as cm:
                    S.check_shards([s])
                self.assertNotIn("delete", str(cm.exception))
                self.assertIn("cannot verify", str(cm.exception))

    def test_invalid_block_count_is_malformed(self):
        s = self.dir / "bad-count.gguf"
        s.write_bytes(pad(header(1) + tensor(b"weight", 7, 2, 0), 18))
        self.assertFalse(S.whole_shard(s))
        with self.assertRaises(Stop) as cm:
            S.check_shards([s])
        self.assertIn("invalid element count", str(cm.exception))


class Alignment(unittest.TestCase):
    setUp = MalformedHeader.setUp
    tearDown = MalformedHeader.tearDown
    refused = MalformedHeader.refused
    def test_invalid_alignment(self):
        for alignment in (0, -8, 3):
            with self.subTest(alignment=alignment):
                kv = gstr(b"general.alignment") + struct.pack("<Ii", 5, alignment)
                self.assertIn("alignment", self.refused(pad(header(0, kv, 1), 0)))


if __name__ == "__main__":
    unittest.main()
