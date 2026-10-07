"""tools/set_context.py (make run CONTEXT=N): python -m unittest tools.test_setup_context"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import set_context as S  # noqa: E402


class SetContext(unittest.TestCase):
    def test_only_the_context_changes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-q2_0.json"
            cfg = {"args": ["--gguf", "m.gguf", "--max-context", "32768", "--kv", "f16"], "parallel": 2}
            p.write_text(json.dumps(cfg, indent=1))
            self.assertEqual(S.set_context(131072, Path(d)), [p])
            out = json.loads(p.read_text())
            self.assertEqual(out["args"], ["--gguf", "m.gguf", "--max-context", "131072", "--kv", "f16"])
            self.assertEqual(out["parallel"], 2)                     # a hand-set key survives

    def test_out_of_range_and_no_install_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                S.set_context(S.TRAINED_CTX * 2, Path(d))
            with self.assertRaises(ValueError):
                S.set_context(131072, Path(d))                       # nothing installed


if __name__ == "__main__":
    unittest.main()
