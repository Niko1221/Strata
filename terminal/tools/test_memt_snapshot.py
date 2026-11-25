"""tools/test_memt_snapshot.py - `memt-memory-v1` round-trip and validation.

    python tools/test_memt_snapshot.py [-v]
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import memt_audit as A  # noqa: E402
import memt_snapshot as S  # noqa: E402
import test_memt_audit as T  # noqa: E402

IDENTITY = "a" * 64


def states(layers=(1, 3), heads=T.H, dim=T.D):
    out = {}
    for index, layer in enumerate(layers):
        values = np.arange(heads * dim * dim, dtype=np.float32).reshape(1, heads, dim, dim)
        out[layer] = values + index
    return out


def tokenizer():
    return {"chat_template": "{{ messages }}", "vocab": {"a": 0, "b": 1, "c": 2}}


class Identity(unittest.TestCase):
    def test_matches_the_documented_hash(self):
        config = T.config()
        want = hashlib.sha256(
            json.dumps(
                {
                    "backbone": config["backbone_configs"],
                    "memt": config["memt"],
                    "codec": "qwen-text-raw-v1",
                    "template": tokenizer()["chat_template"],
                    "vocabulary": tokenizer()["vocab"],
                },
                sort_keys=True,
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(S.snapshot_identity(config, tokenizer()), want)

    def test_rejects_another_family(self):
        config = T.config()
        config["model_type"] = "memt_deepseek_v4"
        with self.assertRaises(S.SnapshotError):
            S.snapshot_identity(config, tokenizer())


class RoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "state.safetensors"

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip(self):
        source = states()
        result = S.save(self.path, IDENTITY, source)
        self.assertTrue(result["has_memory"])
        got = S.load(self.path, IDENTITY, {1: (1, T.H, T.D, T.D), 3: (1, T.H, T.D, T.D)})
        self.assertEqual(set(got), {1, 3})
        for layer, value in source.items():
            np.testing.assert_array_equal(got[layer], value)

    def test_header_is_safetensors(self):
        S.save(self.path, IDENTITY, states(layers=(1,)))
        header = A.read_safetensors_header(self.path)
        self.assertEqual(set(header), {"memory.1"})
        self.assertEqual(header["memory.1"][0], "F32")
        self.assertEqual(header["memory.1"][1], (1, T.H, T.D, T.D))

    def test_identity_mismatch(self):
        S.save(self.path, IDENTITY, states())
        with self.assertRaises(S.SnapshotError):
            S.load(self.path, "b" * 64)

    def test_layer_set_mismatch(self):
        S.save(self.path, IDENTITY, states(layers=(1,)))
        with self.assertRaises(S.SnapshotError):
            S.load(self.path, IDENTITY, {1: (1, T.H, T.D, T.D), 3: (1, T.H, T.D, T.D)})

    def test_wrong_shape(self):
        with self.assertRaises(S.SnapshotError):
            S.encode({1: np.zeros((1, T.H, T.D, T.D + 1), dtype=np.float32)}, IDENTITY)

    def test_non_finite(self):
        bad = states(layers=(1,))
        bad[1][0, 0, 0, 0] = np.nan
        with self.assertRaises(S.SnapshotError):
            S.encode(bad, IDENTITY)

    def test_wrong_dtype(self):
        with self.assertRaises(S.SnapshotError):
            S.encode({1: np.zeros((1, T.H, T.D, T.D), dtype=np.float16)}, IDENTITY)

    def test_refuses_overwrite(self):
        S.save(self.path, IDENTITY, states())
        with self.assertRaises(FileExistsError):
            S.save(self.path, IDENTITY, states())
        S.save(self.path, IDENTITY, states(), overwrite=True)

    def test_empty_snapshot(self):
        result = S.save(self.path, IDENTITY, {})
        self.assertFalse(result["has_memory"])
        self.assertEqual(S.load(self.path, IDENTITY, {}), {})

    def test_corrupt_header(self):
        self.path.write_bytes(b"\x10" + b"\x00" * 7 + b"not json")
        with self.assertRaises(S.SnapshotError):
            S.load(self.path, IDENTITY)

    def test_truncated_data(self):
        blob = S.encode(states(layers=(1,)), IDENTITY)
        with self.assertRaises(S.SnapshotError):
            S.decode(blob[:-4], IDENTITY)

    def test_invalid_tensor_ranges(self):
        blob = S.encode(states(heads=2, dim=2), IDENTITY)
        overlap = blob.replace(b"[32, 64]", b"[16, 48]", 1)
        with self.assertRaisesRegex(S.SnapshotError, "overlap"):
            S.decode(overlap, IDENTITY)
        with self.assertRaisesRegex(S.SnapshotError, "unexpected data"):
            S.decode(blob + b"\0", IDENTITY)

    def test_duplicate_header_key(self):
        blob = S.encode(states(heads=2, dim=2), IDENTITY)
        duplicate = blob.replace(b'"shape"', b'"dtype"', 1)
        with self.assertRaisesRegex(S.SnapshotError, "duplicate snapshot header key"):
            S.decode(duplicate, IDENTITY)


class SafetensorsInterop(unittest.TestCase):
    """The library memt_sglang writes and loads snapshots with must agree with this codec.

    The reference's `runtime/snapshots.py` uses `safetensors` (`save_file` /
    `safe_open`); our encoder writes the format by hand and our decoder reads it
    by hand.  This class pins both directions against the library itself, so the
    HF side of P6 is known to accept an engine-written file.  It is skipped where
    `safetensors` is not installed (it is, on any host that runs memt_sglang).
    """

    @classmethod
    def setUpClass(cls):
        try:
            import safetensors.numpy  # noqa: F401
        except ImportError:   # pragma: no cover - the local CPU box has no safetensors
            raise unittest.SkipTest("safetensors is not installed here (it is on the memt_sglang host)")

    def test_our_file_is_read_by_safetensors(self):
        from safetensors import safe_open
        layers = states(layers=(3, 7))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "engine.safetensors"
            S.save(path, IDENTITY, layers)
            with safe_open(str(path), framework="numpy") as handle:
                self.assertEqual(handle.metadata(), {"format": S.FORMAT, "identity": IDENTITY})
                self.assertEqual(set(handle.keys()), {"memory.3", "memory.7"})
                for layer, value in layers.items():
                    tensor = handle.get_tensor("memory.%d" % layer)
                    self.assertEqual(tensor.dtype, np.float32)
                    self.assertEqual(tuple(tensor.shape), tuple(value.shape))
                    np.testing.assert_array_equal(tensor, value)

    def test_a_safetensors_file_is_read_by_us(self):
        from safetensors.numpy import save_file
        layers = states(layers=(5,))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hf.safetensors"
            save_file({"memory.5": layers[5]}, str(path),
                      metadata={"format": S.FORMAT, "identity": IDENTITY})
            decoded = S.load(path, IDENTITY)
            self.assertEqual(sorted(decoded), [5])
            np.testing.assert_array_equal(decoded[5], layers[5])

    def test_an_empty_memory_round_trips(self):
        from safetensors.numpy import save_file
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.safetensors"
            save_file({}, str(path), metadata={"format": S.FORMAT, "identity": IDENTITY})
            self.assertEqual(S.load(path, IDENTITY), {})

    def test_a_safetensors_file_with_another_identity_is_refused(self):
        from safetensors.numpy import save_file
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "other.safetensors"
            save_file({"memory.5": states(layers=(5,))[5]}, str(path),
                      metadata={"format": S.FORMAT, "identity": "b" * 64})
            with self.assertRaises(S.SnapshotError):
                S.load(path, IDENTITY)


if __name__ == "__main__":
    unittest.main()
