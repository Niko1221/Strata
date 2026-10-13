"""SC117 installer integration: mocked downloads and GPU; never fetch model weights."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup
import strata_mcp
import strata_inspect
from test_setup_golden import PROFILES, install


class Abliterated(unittest.TestCase):
    def test_install_all_sizes_keeps_packs_and_configs_separate(self):
        ram, gpu = PROFILES["64GB-1x32GB"]
        family = setup.FAMILIES["abliterated"]
        for model in ("Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S"):
            with self.subTest(model=model):
                downloads, runs, verified = [], [], []

                def download(url, dst, what=None):
                    downloads.append(url)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(b"")
                    setup.mark(dst)

                extra = [mock.patch.object(setup, "download", download),
                         mock.patch.object(setup, "run", lambda cmd, **kw: runs.append(list(map(str, cmd)))),
                         mock.patch.object(setup, "verify_sha256",
                                           lambda p, size, sha: verified.append((p.name, size, sha)))]
                code, out, cfg, asked = install(ram, gpu, ["--family", "abliterated", "--model", model,
                                                          "--context", "8192", "--no-start"], extra=extra)
                self.assertEqual(code, 0, out)
                self.assertEqual(asked, [])
                names = [setup.model_file(family, model, i) for i in (1, 2)]
                self.assertEqual(downloads, [family["hf"].format(q=model) + name for name in names])
                self.assertEqual(verified, [(name, *family["sha256"][name]) for name in names])
                args = cfg["args"]
                self.assertTrue(args[args.index("--pack") + 1].endswith("abliterated-" + model.lower()))
                self.assertTrue(args[args.index("--native") + 1].endswith(names[0]))
                self.assertTrue(args[args.index("--ple-gguf") + 1].endswith(names[1]))
                self.assertIn("/mtp/rt", args[args.index("--mtp") + 1].replace("\\", "/"))
                self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-abliterated-" + model.lower())
                self.assertNotIn("--control-vector-scaled", args)
                pack = next(cmd for cmd in runs if cmd[1].endswith("iq_pack.py"))
                self.assertTrue(pack[pack.index("--gguf") + 1].endswith(names[0]))
                self.assertNotIn("--compat-bf16", pack)

    def test_config_recognition_and_mcp_sizes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "strata-abliterated-iq2_xs.json"
            path.write_text(json.dumps({"args": ["--max-context", "8192"]}))
            choices = setup.choices_from_config(path)
            self.assertEqual((choices["family"], choices["model"]), ("abliterated", "IQ2_XS"))
        for models, families in ((setup.MODELS, setup.FAMILIES),
                                 (strata_mcp.FALLBACK_MODELS, strata_mcp.FALLBACK_FAMILIES)):
            self.assertEqual(strata_mcp.Strata.sizes_of(models, "abliterated"),
                             ["Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S"])
            self.assertEqual(families["abliterated"]["download_gb"], setup.ABLITERATED_DOWNLOAD_GB)
        self.assertEqual(next(iter(setup.FAMILIES)), "qwen")

    def test_original_download_estimates_and_shard_identity_stay(self):
        for model in ("Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S"):
            family = setup.FAMILIES["abliterated"]
            name = setup.model_file(family, model, 1)
            self.assertEqual(setup.gguf_choice(name), ("abliterated", model))
            self.assertEqual(setup.model_download_gb(setup.FAMILIES["qwen"], model),
                             setup.MODELS[model]["download_gb"])
            self.assertGreater(setup.model_download_gb(family, model), setup.MODELS[model]["download_gb"])

    def test_header_fingerprints_identify_all_four_sizes(self):
        table = strata_inspect.load_fingerprints()
        for model in ("Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S"):
            fp = table["abliterated/" + model]
            kind, text = strata_inspect.verdict(fp, table)
            self.assertEqual(kind, "known")
            self.assertIn("--family abliterated --model " + model, text)
            self.assertEqual(fp["expert_bytes"], table["qwen/" + model]["expert_bytes"])
            self.assertEqual(fp["revision"], setup.HF_REVISIONS[fp["repo"]])

    def test_modelscope_does_not_download_from_an_unconfigured_mirror(self):
        ram, gpu = PROFILES["64GB-1x32GB"]
        with mock.patch.dict(os.environ, {"STRATA_SOURCE": "modelscope"}):
            code, out, _, _ = install(ram, gpu, ["--family", "abliterated", "--no-start"])
        self.assertEqual(code, 1)
        self.assertIn("use --source huggingface", out)


if __name__ == "__main__":
    unittest.main()
