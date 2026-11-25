"""tools/test_memt_compare.py - the snapshot comparison used by P6.

    python terminal/tools/test_memt_compare.py [-v]

Builds small snapshots with the real codec and checks the per-layer metrics, the
layer-set rules, and the CLI's exit codes.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import memt_compare as C  # noqa: E402
import memt_snapshot as S  # noqa: E402

H, D = 2, 2
IDENTITY = "ab" * 32


def state(seed: int, scale: float = 1.0) -> np.ndarray:
    values = np.arange(H * D * D, dtype=np.float32).reshape(1, H, D, D)
    return (values + np.float32(seed)) * np.float32(scale)


class Metrics(unittest.TestCase):
    def test_identical(self):
        a = state(3)
        item = C.compare_layer(a, a.copy())
        self.assertTrue(item["same_shape"] and item["bit_identical"])
        self.assertEqual(item["max_abs"], 0.0)
        self.assertAlmostEqual(item["cosine"], 1.0, places=12)

    def test_small_difference(self):
        a = state(3)
        b = a + np.float32(1e-5)
        item = C.compare_layer(a, b)
        self.assertFalse(item["bit_identical"])
        self.assertAlmostEqual(item["max_abs"], 1e-5, places=6)
        self.assertGreater(item["cosine"], 0.999999)
        self.assertEqual(item["nonfinite"], 0)

    def test_shape_mismatch(self):
        item = C.compare_layer(state(3), np.zeros((1, H, D, D + 1), dtype=np.float32))
        self.assertFalse(item["same_shape"])

    def test_nonfinite_is_reported(self):
        b = state(3)
        b[0, 0, 0, 0] = np.nan
        item = C.compare_layer(state(3), b)
        self.assertEqual(item["nonfinite"], 1)

    def test_layer_sets(self):
        a = {3: state(1), 7: state(2)}
        ok = C.compare(a, {3: state(1), 7: state(2)})
        self.assertTrue(ok["ok"])
        missing = C.compare(a, {3: state(1)})
        self.assertFalse(missing["ok"])
        self.assertEqual(missing["missing_in_engine"], [7])
        extra = C.compare(a, {3: state(1), 7: state(2), 11: state(3)})
        self.assertFalse(extra["ok"])
        self.assertEqual(extra["extra_in_engine"], [11])

    def test_a_scaled_state_fails_despite_cosine_one(self):
        a = {3: state(3)}
        b = {3: state(3) * np.float32(2.0)}
        report = C.compare(a, b)
        self.assertAlmostEqual(report["layers"]["3"]["cosine"], 1.0, places=6)
        self.assertAlmostEqual(report["layers"]["3"]["rel_l2"], 1.0, places=6)
        self.assertFalse(report["ok"])
        self.assertTrue(C.compare(a, b, max_rel_l2=1.5)["ok"])

    def test_the_table_shows_rel_l2_and_the_bound(self):
        text = C.format_table(C.compare({3: state(1)}, {3: state(1)}))
        self.assertIn("rel_l2", text)
        self.assertIn("max rel_l2", text)

    def test_cosine_threshold(self):
        a = {3: state(3)}
        b = {3: state(3) * np.float32(1.01)}
        # a uniform scale leaves the cosine at 1; a per-element tilt does not
        b[3][0, 0, 0, 1] = -b[3][0, 0, 0, 1]
        report = C.compare(a, b, min_cosine=0.999999)
        self.assertFalse(report["ok"])
        self.assertLess(report["layers"]["3"]["cosine"], 1.0)


class Files(unittest.TestCase):
    def write(self, root: Path, name: str, layers: dict, identity=IDENTITY) -> Path:
        path = root / name
        S.save(path, identity, layers)
        return path

    def test_cli_pass_and_fail_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = {3: state(1), 7: state(2)}
            self.write(root, "a.safetensors", a)
            self.write(root, "b.safetensors", a)
            self.assertEqual(C.main(["--a", str(root / "a.safetensors"), "--identity-a", IDENTITY,
                                     "--b", str(root / "b.safetensors"), "--identity-b", IDENTITY]), 0)
            b = {3: -state(1), 7: state(2)}   # anti-correlated: cosine < 0
            self.write(root, "c.safetensors", b)
            self.assertEqual(C.main(["--a", str(root / "a.safetensors"), "--identity-a", IDENTITY,
                                     "--b", str(root / "c.safetensors"), "--identity-b", IDENTITY]), 1)
            self.assertEqual(C.main(["--a", str(root / "a.safetensors"), "--identity-a", "cd" * 32,
                                     "--b", str(root / "b.safetensors"), "--identity-b", IDENTITY]), 2)

    def test_cli_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = {3: state(1)}
            self.write(root, "a.safetensors", a)
            self.write(root, "b.safetensors", a)
            import contextlib
            import io
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                C.main(["--a", str(root / "a.safetensors"), "--identity-a", IDENTITY,
                        "--b", str(root / "b.safetensors"), "--identity-b", IDENTITY, "--json"])
            report = json.loads(out.getvalue())
            self.assertTrue(report["ok"])
            self.assertIn("3", report["layers"])

    def test_shape_mismatch_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write(root, "a.safetensors", {3: state(1)})
            self.write(root, "b.safetensors", {3: np.zeros((1, H, D + 1, D + 1), dtype=np.float32)})
            self.assertEqual(C.main(["--a", str(root / "a.safetensors"), "--identity-a", IDENTITY,
                                     "--b", str(root / "b.safetensors"), "--identity-b", IDENTITY]), 2)


if __name__ == "__main__":
    unittest.main()
