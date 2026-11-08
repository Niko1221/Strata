"""0.2.10: for an Intel Arc card Strata builds the pack itself (setup.py --prepare-pack)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

class IntelPack(unittest.TestCase):
    def test_prepare_pack_fetches_gguf_py_and_runs_the_pack_tool(self):
        import setup
        calls = {}
        with tempfile.TemporaryDirectory() as d:
            gdir = Path(d) / "models"
            gdir.mkdir()
            fam = setup.FAMILIES["qwen"]
            shards = [gdir / setup.model_file(fam, "IQ3_XXS", i) for i in range(1, setup.model_shards(fam, "IQ3_XXS") + 1)]
            for s in shards:
                s.write_bytes(b"x")
            pack = Path(d) / "packs" / "p"

            def fake_run(cmd, cwd=None, env=None, check=True, quiet=False):
                calls["cmd"], calls["env"] = cmd, env
                (pack / "tokenizer").mkdir(parents=True)
                (pack / "tokenizer" / "vocab.json").write_text("{}")

            args = mock.Mock(gguf_dir=str(gdir), family="qwen", model="IQ3_XXS", prepare_pack=str(pack))
            with mock.patch.object(setup, "get_llama_cpp", lambda: Path(d) / "llama"), mock.patch.object(setup, "run", fake_run), \
                    mock.patch.object(setup, "gguf_dir_problem", lambda *a: None):
                self.assertEqual(setup.prepare_pack_only(args), 0)
        self.assertIn("iq_pack.py", " ".join(map(str, calls["cmd"])))
        self.assertEqual(calls["env"]["STRATA_GGUF_PY"], str(Path(d) / "llama" / "gguf-py"))


if __name__ == "__main__":
    unittest.main()
