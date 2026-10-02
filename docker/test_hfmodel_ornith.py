"""Single-file (Ornith) resolution in docker/hfmodel.py.

Run with: python -m unittest docker/test_hfmodel_ornith.py

The two-shard Qwen3.8 layout is the historical one; Ornith is one stock GGUF plus a separate MTP draft.
This builds both cache layouts by hand and checks the resolver reports them correctly, so a change to the
shared code cannot quietly break the family it was not aimed at.
"""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_hfmodel():
    spec = importlib.util.spec_from_file_location("hfmodel_under_test", HERE / "hfmodel.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def put_snapshot(cache: Path, repo: str, rev: str, relpath: str) -> Path:
    """A cache entry: hub/models--org--repo/snapshots/<rev>/<relpath> + refs/main -> rev."""
    rd = cache / ("models--" + repo.replace("/", "--"))
    snap = rd / "snapshots" / rev
    snap.mkdir(parents=True, exist_ok=True)
    (rd / "refs").mkdir(parents=True, exist_ok=True)
    (rd / "refs" / "main").write_text(rev)
    f = snap / relpath
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"GGUF")
    return f


class OrnithResolver(unittest.TestCase):
    def setUp(self):
        self.hf = load_hfmodel()
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_single_file_family_resolves(self):
        want = put_snapshot(self.cache, "AtomicChat/Ornith-1.5-35B-A3B-GGUF", "abc123",
                            "Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf")
        got = self.hf.shard_path(self.cache, "ornith", 1)
        self.assertEqual(got, want)
        self.assertIsNone(self.hf.shard_path(self.cache, "ornith", 2),
                          "a one-file family has no second file")
        self.assertEqual(self.hf.present(self.cache, "ornith"), (True, False))

    def test_external_mtp_resolves(self):
        put_snapshot(self.cache, "AtomicChat/Ornith-1.5-35B-A3B-GGUF", "abc123",
                     "Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf")
        want = put_snapshot(self.cache, "EryriLabs/Ornith-1.5-35B-A3B-BigBang-MTP-GGUF", "def456",
                            "mtpdraft-Q8_0.gguf")
        self.assertEqual(self.hf.mtp_path(self.cache, "ornith"), want)

    def test_mtp_absent_is_none(self):
        put_snapshot(self.cache, "AtomicChat/Ornith-1.5-35B-A3B-GGUF", "abc123",
                     "Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf")
        self.assertIsNone(self.hf.mtp_path(self.cache, "ornith"))

    def test_two_shard_family_unchanged(self):
        put_snapshot(self.cache, "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "rev",
                     "IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")
        put_snapshot(self.cache, "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "rev",
                     "IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf")
        self.assertEqual(self.hf.present(self.cache, "IQ3_XXS"), (True, True))
        self.assertIsNone(self.hf.mtp_path(self.cache, "IQ3_XXS"),
                          "the Qwen3.8 families have no external MTP draft in this table")

    def test_unknown_single_file_falls_back_to_title_glob(self):
        # A rename upstream: the table's exact name is absent, but the family title still finds the file.
        want = put_snapshot(self.cache, "AtomicChat/Ornith-1.5-35B-A3B-GGUF", "rev",
                            "Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS-renamed.gguf")
        self.assertEqual(self.hf.shard_path(self.cache, "ornith", 1), want)

    def test_shell_output_names_mtp_and_file_count(self):
        import subprocess
        import sys
        put_snapshot(self.cache, "AtomicChat/Ornith-1.5-35B-A3B-GGUF", "rev",
                     "Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf")
        put_snapshot(self.cache, "EryriLabs/Ornith-1.5-35B-A3B-BigBang-MTP-GGUF", "rev",
                     "mtpdraft-Q8_0.gguf")
        out = subprocess.run([sys.executable, str(HERE / "hfmodel.py"), "--cache", str(self.cache),
                              "--model", "ornith", "--print", "shell"],
                             capture_output=True, text=True, check=True).stdout
        self.assertIn("STRATA_FILES=1", out)
        self.assertIn("STRATA_MTP_GGUF_CACHED=1", out)
        self.assertIn("mtpdraft-Q8_0.gguf", out)
        self.assertIn("Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf", out)
        # entrypoint-hip.sh evals this output and keeps its own STRATA_MTP (the packed runtime
        # directory); a bare STRATA_MTP key here would clobber it and disable --spec.  Regression guard.
        self.assertNotIn("STRATA_MTP=", out)
        self.assertNotIn("STRATA_MTP_CACHED=", out)


if __name__ == "__main__":
    unittest.main()
