"""Tests for setup.py's experimental Unsloth UD-Q4_K_XL choice (docs/UNSLOTH_Q4.md): the four pinned shards with
their sizes and SHA-256, the RAM budget from the PC's RAM, the pack with --compat-bf16 (never experts.bin), the
engine version it needs, one GPU, no images.  Mocked - no GPU, no downloads, nothing written outside a temp folder.

    python -m unittest tools.test_setup_unsloth
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

M = "UD-Q4_K_XL"
REV = "38bb39ee97821de2c9009abb7e93950eec396e66"


class Pins(unittest.TestCase):
    def test_the_four_shards(self):
        fam = setup.FAMILIES["unsloth"]
        names = [fam["file"].format(q=M, i=i) for i in range(1, fam["shards"] + 1)]
        self.assertEqual(names, list(setup.UNSLOTH_SHARDS))
        self.assertEqual(sum(b for b, _ in setup.UNSLOTH_SHARDS.values()), 111334654784)
        for _, sha in setup.UNSLOTH_SHARDS.values():
            self.assertRegex(sha, r"^[0-9a-f]{64}$")
        self.assertEqual(fam["hf"].format(q=M),
                         f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REV}/UD-Q4_K_XL/")
        self.assertEqual(setup.MODELS[M]["families"], ("unsloth",))   # no other family lists it
        self.assertTrue(fam["experimental"])
        self.assertEqual(list(setup.FAMILIES)[0], "qwen")             # not the default choice

    def test_not_in_the_other_families(self):
        for f in ("qwen", "swift", "coder"):
            self.assertNotIn(M, [m for m in setup.MODELS if f in setup.MODELS[m].get("families", ("qwen", "swift"))])


class Budget(unittest.TestCase):
    def test_ram_less_24(self):
        self.assertEqual(setup.resident_budget_gib(M, 63.7), 40)       # this PC (64 GB): the measured setting
        self.assertEqual(setup.resident_budget_gib(M, 47.8), 24)
        self.assertEqual(setup.resident_budget_gib(M, 96), 71)         # at most all of the experts (77 GB = 71.7 GiB)
        self.assertEqual(setup.resident_budget_gib(M, 256), 71)
        self.assertEqual(setup.resident_budget_gib(M, 20), 8)          # the floor
        self.assertEqual(setup.resident_budget_gib(M, 63.7, 1.7), 38)  # a KV cache streamed to RAM comes out of it


def quiet(fn, *args):
    """-> (exit code or None, printed text)."""
    out = io.StringIO()
    code = None
    with contextlib.redirect_stdout(out):
        try:
            code = fn(*args)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


class Sha256(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.f = Path(self.tmp.name) / "x-00002-of-00004.gguf"
        self.f.write_bytes(b"strata" * 1000)
        self.sha = hashlib.sha256(self.f.read_bytes()).hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def test_good_file_marked_once(self):
        code, out = quiet(setup.verify_sha256, self.f, 6000, self.sha)
        self.assertIsNone(code, out)
        self.assertIn(f"sha256 {self.sha}", self.f.with_name(self.f.name + ".done").read_text())
        with mock.patch("hashlib.sha256", side_effect=AssertionError("hashed again")):
            code, out = quiet(setup.verify_sha256, self.f, 6000, self.sha)    # the mark says it was checked
        self.assertIsNone(code, out)

    def test_a_download_mark_alone_is_not_enough(self):
        setup.mark(self.f)                                             # the download's finish mark (a date)
        with mock.patch("hashlib.sha256", wraps=hashlib.sha256) as h:
            code, out = quiet(setup.verify_sha256, self.f, 6000, self.sha)
        self.assertIsNone(code, out)
        self.assertTrue(h.called)

    def test_wrong_hash_deletes_it(self):
        setup.mark(self.f)
        code, out = quiet(setup.verify_sha256, self.f, 6000, "0" * 64)
        self.assertEqual(code, 1)
        self.assertIn("wrong SHA-256", out)
        self.assertFalse(self.f.exists())
        self.assertFalse(self.f.with_name(self.f.name + ".done").exists())   # downloaded again next time

    def test_wrong_size(self):
        code, out = quiet(setup.verify_sha256, self.f, 6001, self.sha)
        self.assertEqual(code, 1)
        self.assertIn("6,000 bytes, not 6,001", out)
        self.assertTrue(self.f.exists())


class Config(unittest.TestCase):
    def test_choices_from_config(self):
        with tempfile.TemporaryDirectory() as t:
            for name, family, model in (("strata-unsloth-ud-q4_k_xl.json", "unsloth", M),
                                        ("strata-swift-iq3_xxs.json", "swift", "IQ3_XXS"),
                                        ("strata-iq3_s.json", "qwen", "IQ3_S"),
                                        ("strata-coder-iq1_m.json", "coder", "IQ1_M")):
                p = Path(t) / name
                p.write_text(json.dumps({"args": ["--max-context", "8192"]}))
                ch = setup.choices_from_config(p)
                self.assertEqual((ch["family"], ch["model"]), (family, model), name)


class FakeGGUF:
    """gguf_reader.GGUFFile: shard 2 holds the PLE table."""
    def __init__(self, path):
        names = ["per_layer_token_embd.weight"] if "00002-of" in str(path) else ["blk.0.ffn_up_exps.weight"]
        self.tensors = [types.SimpleNamespace(name=n) for n in names]


class Main(unittest.TestCase):
    """setup.main() for the Unsloth choice, every outside effect mocked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        (self.t / "data" / "mtp" / "rt").mkdir(parents=True)
        (self.t / "data" / "mtp" / "rt" / "experts.bin").write_bytes(b"")
        self.downloads, self.runs, self.verified = [], [], []

    def tearDown(self):
        self.tmp.cleanup()

    def main(self, argv, ram=63.7, version="0.1.32", n_gpus=1, model=True):
        eng = self.t / "engine"
        eng.mkdir(exist_ok=True)
        (eng / "BUILD.json").write_text(json.dumps({"version": version, "source": "local"}))
        found = [{"index": i, "name": "NVIDIA GeForce RTX 5070", "vram_gb": 11.9, "arch": "120", "driver": "580.97"}
                 for i in range(n_gpus)]

        def fake_download(url, dst, what=None):
            self.downloads.append(url)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)

        def fake_run(cmd, *a, **k):
            self.runs.append([str(x) for x in cmd])

        patches = [
            mock.patch.object(setup, "ROOT", self.t),
            mock.patch.object(setup, "GPU_PICK", None),
            mock.patch.object(setup, "data_folder", lambda d: (self.t / "data", [])),
            mock.patch.object(setup, "installed_configs", lambda: []),
            mock.patch.object(setup, "gpus", lambda: found),
            mock.patch.object(setup, "amd_gpus", lambda: []),
            mock.patch.object(setup, "ram_gb", lambda: ram),
            mock.patch.object(setup, "cpu_info", lambda: ("Test CPU", True, True)),
            mock.patch.object(setup, "page_file_gb", lambda: 16.0),
            mock.patch.object(setup, "free_gb", lambda p: 500.0),
            mock.patch.object(setup, "pip_install", lambda *a, **k: None),
            mock.patch.object(setup, "get_llama_cpp", lambda: self.t / "llama.cpp"),
            mock.patch.object(setup, "get_prebuilt", lambda *a, **k: eng),
            mock.patch.object(setup, "download", fake_download),
            mock.patch.object(setup, "check_shards", lambda shards: None),
            mock.patch.object(setup, "verify_sha256", lambda s, size, sha: self.verified.append((s.name, size, sha))),
            mock.patch.object(setup, "run", fake_run),
            mock.patch.object(setup, "refresh_draft_vocab", lambda *a, **k: None),
            mock.patch.object(setup, "write_run_script", lambda tag, cfg, port: self.t / f"start-{tag}.bat"),
            mock.patch.object(setup, "saved_calibration", lambda cfg: None),
            mock.patch.object(setup, "start", mock.Mock(side_effect=AssertionError("started"))),
            mock.patch.dict(sys.modules, {"gguf_reader": types.SimpleNamespace(GGUFFile=FakeGGUF)}),
            mock.patch.object(sys, "argv", ["setup.py", "--family", "unsloth", *(["--model", M] if model else []),
                                            "--yes", "--no-start",
                                            "--models-dir", str(self.t / "models"), *argv]),
            mock.patch("builtins.input", mock.Mock(side_effect=AssertionError("asked"))),
        ]
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            code, out = quiet(setup.main)
        cfg = self.t / "strata-unsloth-ud-q4_k_xl.json"
        return code, out, (json.loads(cfg.read_text()) if cfg.exists() else None)

    def test_install(self):
        code, out, cfg = self.main(["--context", "8192"])
        self.assertEqual(code, 0, out)
        self.assertIn("EXPERIMENTAL", out)
        base = f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REV}/UD-Q4_K_XL/"
        self.assertEqual(self.downloads, [base + n for n in setup.UNSLOTH_SHARDS])   # no image encoder
        self.assertEqual([v[0] for v in self.verified], list(setup.UNSLOTH_SHARDS))
        self.assertEqual([v[1:] for v in self.verified], list(setup.UNSLOTH_SHARDS.values()))
        packs = [r for r in self.runs if r[1].endswith("iq_pack.py")]
        self.assertEqual(len(packs), 1, self.runs)
        self.assertIn("--compat-bf16", packs[0])
        self.assertNotIn("--experts-bin", packs[0])
        self.assertTrue(packs[0][packs[0].index("--gguf") + 1].endswith("UD-Q4_K_XL-00001-of-00004.gguf"))
        args = cfg["args"]
        self.assertEqual(args[args.index("--resident-budget-gib") + 1], "40")
        self.assertTrue(args[args.index("--native") + 1].endswith("-00001-of-00004.gguf"))
        for flag in ("--ple-gguf", "--mmap-experts", "--resident-experts", "--vision", "--control-vector-scaled",
                     "--kv-resident"):
            self.assertNotIn(flag, args)
        self.assertEqual(args[args.index("--max-context") + 1], "8192")
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-unsloth-ud-q4_k_xl")
        self.assertNotIn("vision", cfg)

    def test_recommended_context_on_12_gb(self):
        code, out, cfg = self.main([])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "8192")

    def test_more_ram_bigger_budget(self):
        code, out, cfg = self.main(["--context", "8192"], ram=95.8)
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["args"][cfg["args"].index("--resident-budget-gib") + 1], "71")

    def test_kv_streaming_comes_out_of_the_budget(self):
        code, out, cfg = self.main(["--context", "131072"])
        self.assertEqual(code, 0, out)
        args = cfg["args"]
        self.assertIn("--kv-resident", args)
        kv_gb = 131072 * 13 * 1056 / 1e9
        self.assertEqual(args[args.index("--resident-budget-gib") + 1], str(40 - int(-(-kv_gb // 1))))

    def test_old_engine_stops_before_the_download(self):
        code, out, cfg = self.main(["--context", "8192"], version="0.1.31")
        self.assertEqual(code, 1)
        self.assertIn("needs engine 0.1.32 or newer; this one is 0.1.31", out)
        self.assertEqual(self.downloads, [])
        self.assertIsNone(cfg)

    def test_too_little_ram(self):
        code, out, cfg = self.main(["--context", "8192"], ram=31.9, model=False)   # --yes alone: still a stop
        self.assertEqual(code, 1)
        self.assertIn("needs 48 GB of RAM or more", out)
        self.assertIn("--model UD-Q4_K_XL --yes", out)                             # the way to insist
        self.assertEqual(self.downloads, [])

    def test_too_little_ram_with_an_explicit_model_installs(self):
        """S6 (the owner's rule): --model with --yes is the consent; the risk is said."""
        code, out, cfg = self.main(["--context", "8192"], ram=31.9)
        self.assertEqual(code, 0, out)
        self.assertIn("needs 48 GB of RAM or more; this PC has 32 GB", out)
        self.assertIn("installing UD-Q4_K_XL with 32 GB of RAM, as you chose", out)
        self.assertEqual(cfg["args"][cfg["args"].index("--resident-budget-gib") + 1], "8")

    def test_budget_flag(self):
        """S4: --resident-budget-gib N is kept; more than the recommendation says what it risks."""
        code, out, cfg = self.main(["--context", "8192", "--resident-budget-gib", "48"])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["args"][cfg["args"].index("--resident-budget-gib") + 1], "48")
        self.assertIn("more than setup recommends for this PC (40 GiB", out)
        self.assertIn("Kept as you chose", out)
        code, out, cfg = self.main(["--context", "8192", "--resident-budget-gib", "32.5"])
        self.assertEqual(cfg["args"][cfg["args"].index("--resident-budget-gib") + 1], "32.5")
        self.assertNotIn("more than setup recommends", out)

    def test_budget_flag_with_kv_streaming_is_kept(self):
        code, out, cfg = self.main(["--context", "131072", "--resident-budget-gib", "40"])
        self.assertEqual(code, 0, out)
        args = cfg["args"]
        self.assertIn("--kv-resident", args)
        self.assertEqual(args[args.index("--resident-budget-gib") + 1], "40")      # not 38: the user's number
        self.assertIn("come on top of your 40 GiB RAM budget", out)

    def test_budget_flag_needs_a_positive_number(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code, out, cfg = self.main(["--context", "8192", "--resident-budget-gib", "0"])
        self.assertEqual(code, 2)                                                   # argparse: N > 0
        self.assertIsNone(cfg)

    def test_one_gpu_and_no_images(self):
        code, out, cfg = self.main(["--context", "8192", "--gpus", "0,1", "--vision", "yes", "--low-ram", "on"],
                                   n_gpus=2)
        self.assertEqual(code, 0, out)
        self.assertIn("runs on one GPU", out)
        self.assertIn("images are not available with UD-Q4_K_XL", out)
        self.assertIn("--low-ram on does not apply", out)
        self.assertNotIn("layer_split", cfg)
        self.assertNotIn("--vision", cfg["args"])
        self.assertNotIn("--mmap-experts", cfg["args"])
        self.assertFalse(any("--experts-bin" in r for r in self.runs))


if __name__ == "__main__":
    unittest.main()
