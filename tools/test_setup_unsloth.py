"""Tests for setup.py's Unsloth choices (docs/UNSLOTH_Q4.md): UD-IQ4_XS (regular from 0.1.39) and the experimental
UD-Q4_K_XL: the four pinned shards with
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
        self.assertTrue(setup.MODELS[M]["experimental"])               # 0.1.39: per model, not the family
        self.assertFalse(fam.get("experimental"))
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


class Base(unittest.TestCase):
    """setup.main() for the Unsloth choice, every outside effect mocked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        (self.t / "data" / "mtp" / "rt").mkdir(parents=True)
        (self.t / "data" / "mtp" / "rt" / "experts.bin").write_bytes(b"")
        self.downloads, self.runs, self.verified = [], [], []

    def tearDown(self):
        self.tmp.cleanup()

    def main(self, argv, ram=63.7, version="0.1.32", n_gpus=1, model=True, amd=(), free=500.0, m=M, answers=None,
             family=True):
        """answers: None = --yes; else {words of a question: its answer} (Enter for the others), and every question
        asked is kept in self.asked.  family=False: no --family (the first menu is shown)."""
        self.asked = []

        def fake_input(prompt=""):
            if answers is None:
                raise AssertionError(f"asked {prompt!r}")
            self.asked.append(prompt)
            return next((v for k, v in answers.items() if k in prompt), "")

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
            mock.patch.object(setup, "amd_gpus", lambda *a: list(amd)),
            mock.patch.object(setup, "ram_gb", lambda: ram),
            mock.patch.object(setup, "cpu_info", lambda: ("Test CPU", True, True)),
            mock.patch.object(setup, "cpu_cores", lambda: None),   # #642: not a hybrid CPU
            mock.patch.object(setup, "page_file_gb", lambda: 16.0),
            mock.patch.object(setup, "free_gb", lambda p: free),
            mock.patch.object(setup, "rotational_disk", lambda p: None),   # #605: not the test PC's disk
            mock.patch.object(setup, "is_wsl", lambda: False),            # #974: the tests are not run inside WSL
            mock.patch.object(setup, "pip_install", lambda *a, **k: None),
            mock.patch.object(setup, "get_llama_cpp", lambda: self.t / "llama.cpp"),
            mock.patch.object(setup, "get_prebuilt", lambda *a, **k: eng),
            mock.patch.object(setup, "download", fake_download),
            mock.patch.object(setup, "check_shards", lambda shards: None),
            mock.patch.object(setup, "verify_sha256", lambda s, size, sha: self.verified.append((s.name, size, sha))),
            mock.patch.object(setup, "run", fake_run),
            mock.patch.object(setup, "refresh_draft_vocab", lambda *a, **k: None),
            mock.patch.object(setup, "write_run_script", lambda tag, cfg, port, *_: self.t / f"start-{tag}.bat"),
            mock.patch.object(setup, "saved_calibration", lambda cfg: None),
            mock.patch.object(setup, "start", mock.Mock(side_effect=AssertionError("started"))),
            mock.patch.dict(sys.modules, {"gguf_reader": types.SimpleNamespace(GGUFFile=FakeGGUF)}),
            mock.patch.object(sys, "argv", ["setup.py", *(["--family", "unsloth"] if family else []),
                                            *(["--model", m] if model else []),
                                            *(["--yes"] if answers is None else []), "--no-start",
                                            "--models-dir", str(self.t / "models"), *argv]),
            mock.patch("builtins.input", fake_input),
        ]
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            code, out = quiet(setup.main)
        cfg = self.t / f"strata-unsloth-{m.lower()}.json"
        return code, out, (json.loads(cfg.read_text()) if cfg.exists() else None)

class Main(Base):
    """UD-Q4_K_XL."""

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
        self.assertIn("--model UD-IQ4_XS --yes", out)       # the way to insist (0.1.39: the family's default size)
        self.assertEqual(self.downloads, [])

    def test_amd_is_asked_before_the_download(self):
        """#429: UD-Q4_K_XL has not run on AMD (CUDA-only prompt kernels): --yes alone stops before the 111 GB
        download, saying why; it is not refused outright (the owner's rule: --model with --yes goes on)."""
        r9700 = [{"index": 0, "name": "AMD Radeon AI PRO R9700", "vram_gb": 31.9, "arch": "gfx1201",
                  "driver": "amdgpu"}]
        # chosen in the menu (2 = UD-Q4_K_XL, after the regular UD-IQ4_XS), not by --model: the answer is no
        code, out, cfg = self.main(["--context", "8192", "--backend", "hip", "--vision", "no"], model=False,
                                   amd=r9700, answers={"Which size": "2", "Try it anyway": "n"})
        self.assertEqual(code, 1, out)
        self.assertIn("has not been run on AMD cards yet", out)
        self.assertIn("--model UD-Q4_K_XL --yes", out)
        self.assertEqual(self.downloads, [])

    def test_resumed_download_needs_room_for_the_rest_only(self):
        """#425: the disk check counts the finished shards and .part files already there (sizes scaled down: a
        2 MB "model" with 1.5 MB on the disk, 8 GB of other room needed, 8.001 GB free)."""
        d = self.t / "models" / "unsloth-UD-Q4_K_XL"
        d.mkdir(parents=True)
        (d / (list(setup.UNSLOTH_SHARDS)[0] + ".part")).write_bytes(b"x" * 1_500_000)
        with mock.patch.dict(setup.MODELS[M], {"download_gb": 0.002}):
            code, out, cfg = self.main(["--context", "8192"], free=8.001)
        self.assertNotIn("not enough free disk space", out)
        self.assertEqual(code, 0, out)
        for f in d.iterdir():                                                         # the first run "downloaded" them
            f.unlink()
        import shutil
        shutil.rmtree(self.t / "data" / "mtp", ignore_errors=True)       # #897: a draft layer already there needs less room
        (d / (list(setup.UNSLOTH_SHARDS)[0] + ".part")).write_bytes(b"x" * 1_500_000)
        with mock.patch.dict(setup.MODELS[M], {"download_gb": 0.003}):              # 1.5 MB still missing: no room
            code, out, cfg = self.main(["--context", "8192"], free=8.001)
        self.assertEqual(code, 1)
        self.assertIn("not enough free disk space", out)

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

    def test_one_gpu_and_images_with_a_warning(self):
        """#967: images are allowed with UD-Q4_K_XL, said to be untested."""
        code, out, cfg = self.main(["--context", "8192", "--vision", "yes", "--low-ram", "on"], n_gpus=2)
        self.assertEqual(code, 0, out)
        self.assertIn("(NVIDIA GeForce RTX 5070, 12 GB) only (--gpus 0,1 shares it across", out)
        self.assertNotIn("images are not available", out)
        self.assertIn("images: on", out)
        self.assertIn("images with UD-Q4_K_XL are untested", out)
        self.assertIn("--low-ram on does not apply", out)
        self.assertNotIn("layer_split", cfg)
        self.assertIn("--vision", cfg["args"])
        self.assertTrue(cfg["vision"]["model"].endswith("UD-Q4_K_XL-00001-of-00004.gguf"))
        self.assertNotIn("--mmap-experts", cfg["args"])
        self.assertFalse(any("--experts-bin" in r for r in self.runs))

    @mock.patch.object(setup, "MIN_ENGINE", (0, 1, 39))   # before RESIDENT_SPLIT_ENGINE: no budget on a split
    def test_split_when_the_ram_holds_it(self):
        # #498: 165 GiB and --gpus 0,1: the split without the RAM budget (the engine refuses the pair)
        code, out, cfg = self.main(["--context", "8192", "--gpus", "0,1"], n_gpus=2, ram=165.0)
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertEqual(cfg["layer_split"], "auto")
        for flag in ("--resident-budget-gib", "--mmap-experts", "--resident-experts"):
            self.assertNotIn(flag, cfg["args"])
        self.assertIn("as you chose (--gpus): no RAM budget", out)
        self.assertIn("OS file cache", out)
        self.assertNotIn("RAM budget: ", out)

    @mock.patch.object(setup, "MIN_ENGINE", (0, 1, 39))   # before RESIDENT_SPLIT_ENGINE: no budget on a split
    def test_split_without_the_ram_is_a_warning_not_a_wall(self):
        # #737: below the worst-case estimate (a 128 GB PC ran it) the split is still the user's call
        need = setup.unsloth_split_need_gb()
        self.assertAlmostEqual(need, setup.MODELS[M]["download_gb"] + setup.UNSLOTH_RAM_LEFT_GB)
        code, out, cfg = self.main(["--context", "8192"], n_gpus=2, ram=127.8)       # no --gpus: stays on one GPU
        self.assertEqual(code, 0, out)
        self.assertIn("uses them together anyway", out)
        self.assertNotIn("layer_split", cfg)
        self.assertEqual(cfg["args"][cfg["args"].index("--resident-budget-gib") + 1], "71")   # all of them
        code, out, cfg = self.main(["--context", "8192", "--gpus", "0,1"], n_gpus=2, ram=127.8)   # asked for by name
        self.assertEqual(code, 0, out)
        self.assertIn("worst case", out)
        self.assertIn("as you chose (--gpus), with 128 GB of RAM", out)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertNotIn("--resident-budget-gib", cfg["args"])

    @mock.patch.object(setup, "MIN_ENGINE", (0, 1, 39))   # before RESIDENT_SPLIT_ENGINE: no budget on a split
    def test_an_explicit_budget_keeps_one_gpu(self):
        code, out, cfg = self.main(["--context", "8192", "--gpus", "0,1", "--resident-budget-gib", "60"], n_gpus=2,
                                   ram=165.0)
        self.assertEqual(code, 0, out)
        self.assertIn("--resident-budget-gib has no layer split", out)
        self.assertNotIn("layer_split", cfg)
        self.assertEqual(cfg["args"][cfg["args"].index("--resident-budget-gib") + 1], "60")

    def test_split_keeps_the_budget(self):
        """#642: with a resident-split engine the cards asked for keep the RAM budget, at any RAM."""
        with mock.patch.object(setup, "MIN_ENGINE", setup.RESIDENT_SPLIT_ENGINE):
            code, out, cfg = self.main(["--context", "8192", "--gpus", "0,1"], n_gpus=2)      # 64 GB: no 135 GB asked
            self.assertEqual(code, 0, out)
            self.assertEqual(cfg["gpu"], [0, 1])
            self.assertEqual(cfg["layer_split"], "auto")
            self.assertEqual(cfg["args"][cfg["args"].index("--resident-budget-gib") + 1], "40")
            for flag in ("--mmap-experts", "--resident-experts"):
                self.assertNotIn(flag, cfg["args"])
            self.assertIn("as you chose (--gpus), with its RAM budget", out)
            self.assertIn("RAM budget: 40 GiB", out)
            self.assertNotIn("worst case", out)
            code, out, cfg = self.main(["--context", "8192", "--gpus", "0,1", "--resident-budget-gib", "30"], n_gpus=2)
            self.assertEqual(code, 0, out)                                                  # an explicit budget too
            self.assertEqual(cfg["gpu"], [0, 1])
            self.assertEqual(cfg["args"][cfg["args"].index("--resident-budget-gib") + 1], "30")
            self.assertNotIn("has no layer split", out)

    def test_asked_with_the_budget_one_gpu_by_default(self):
        """#642: asked as before (one GPU recommended); "2" takes both cards with the budget."""
        with mock.patch.object(setup, "MIN_ENGINE", setup.RESIDENT_SPLIT_ENGINE):
            code, out, cfg = self.main(["--context", "8192"], n_gpus=2, answers={})          # Enter: one GPU
            self.assertEqual(code, 0, out)
            self.assertTrue(any(f"{M}: which GPUs?" in q and "[1]" in q for q in self.asked), self.asked)
            self.assertNotIn("layer_split", cfg)
            self.assertIn("--resident-budget-gib", cfg["args"])
            code, out, cfg = self.main(["--context", "8192"], n_gpus=2, answers={"which GPUs?": "2"})
            self.assertEqual(code, 0, out)
            self.assertEqual(cfg["gpu"], [0, 1])
            self.assertIn("--resident-budget-gib", cfg["args"])

    def test_yes_alone_keeps_one_gpu(self):
        code, out, cfg = self.main(["--context", "8192"], n_gpus=2, ram=165.0)
        self.assertEqual(code, 0, out)
        self.assertIn("runs on one GPU: using", out)
        self.assertIn("--gpus 0,1 shares it", out)
        self.assertNotIn("layer_split", cfg)
        self.assertIn("--resident-budget-gib", cfg["args"])


X = "UD-IQ4_XS"


class IQ4XS(Base):
    """#621: Unsloth's UD-IQ4_XS, a regular (not experimental) choice from 0.1.39 - three pinned shards, the same
    RAM-budget setup as UD-Q4_K_XL (the pack with --compat-bf16, never experts.bin, one GPU), engine 0.1.38 or newer,
    no AMD question; images are asked like the 2-3-bit models' (off by default)."""

    def test_pins(self):
        fam = setup.FAMILIES["unsloth"]
        names = [setup.model_file(fam, X, i) for i in range(1, setup.model_shards(fam, X) + 1)]
        self.assertEqual(names, list(setup.UNSLOTH_IQ4_XS_SHARDS))
        self.assertEqual(sum(b for b, _ in setup.UNSLOTH_IQ4_XS_SHARDS.values()), 93682584224)
        for n, (size, sha) in setup.UNSLOTH_IQ4_XS_SHARDS.items():
            self.assertRegex(sha, r"^[0-9a-f]{64}$")
            self.assertEqual(fam["sha256"][n], (size, sha))
        for n in setup.UNSLOTH_SHARDS:                                         # UD-Q4_K_XL's are still there
            self.assertEqual(fam["sha256"][n], setup.UNSLOTH_SHARDS[n])
        self.assertAlmostEqual(setup.MODELS[X]["download_gb"], 93682584224 / 1e9, places=1)
        self.assertEqual(setup.MODELS[X]["families"], ("unsloth",))
        self.assertEqual([m for m in setup.MODELS if "unsloth" in setup.MODELS[m].get("families", ())], [M, X])
        self.assertIsNone(setup.gguf_unsupported(names[0]))                    # #444: a file setup runs now
        self.assertEqual(setup.gguf_choice(names[0]), ("unsloth", X))
        self.assertEqual(setup.budget_model({"args": ["--native", "/m/" + names[0]]}), X)
        self.assertEqual(setup.budget_model({"args": ["--native", "/m/" + list(setup.UNSLOTH_SHARDS)[0]]}), M)
        self.assertEqual(setup.budget_model({"args": []}), M)

    def test_budget(self):
        self.assertEqual(setup.resident_budget_gib(X, 63.7), 40)
        self.assertEqual(setup.resident_budget_gib(X, 95.8), 55)              # at most all of them (59.5 GB)
        self.assertAlmostEqual(setup.unsloth_split_need_gb(X), 93.7 + setup.UNSLOTH_RAM_LEFT_GB)

    def test_install(self):
        code, out, cfg = self.main(["--context", "8192"], version="0.1.38", m=X)
        self.assertEqual(code, 0, out)
        self.assertNotIn("is EXPERIMENTAL", out)                               # (the menu still lists UD-Q4_K_XL)
        base = f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REV}/UD-IQ4_XS/"
        self.assertEqual(self.downloads, [base + n for n in setup.UNSLOTH_IQ4_XS_SHARDS])
        self.assertEqual(self.verified, [(n, *v) for n, v in setup.UNSLOTH_IQ4_XS_SHARDS.items()])
        packs = [r for r in self.runs if r[1].endswith("iq_pack.py")]
        self.assertEqual(len(packs), 1, self.runs)
        self.assertIn("--compat-bf16", packs[0])
        self.assertNotIn("--experts-bin", packs[0])
        self.assertTrue(packs[0][packs[0].index("--gguf") + 1].endswith("UD-IQ4_XS-00001-of-00003.gguf"))
        args = cfg["args"]
        self.assertEqual(args[args.index("--resident-budget-gib") + 1], "40")
        self.assertTrue(args[args.index("--native") + 1].endswith("UD-IQ4_XS-00001-of-00003.gguf"))
        for flag in ("--ple-gguf", "--mmap-experts", "--resident-experts", "--vision"):
            self.assertNotIn(flag, args)
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-unsloth-ud-iq4_xs")
        pack = self.t / "data" / "packs" / "unsloth-ud-iq4_xs"                 # its own pack and tokenizer
        self.assertEqual(packs[0][packs[0].index("--out") + 1], str(pack))
        self.assertEqual(args[args.index("--pack") + 1], str(pack))
        self.assertEqual(cfg["tokenizer"], str(pack / "tokenizer"))
        self.assertNotIn("vision", cfg)                                        # images: asked, off by default
        with tempfile.TemporaryDirectory() as t:                               # a start reads the choice back
            p = Path(t) / "strata-unsloth-ud-iq4_xs.json"
            p.write_text(json.dumps(cfg))
            ch = setup.choices_from_config(p)
            self.assertEqual((ch["family"], ch["model"]), ("unsloth", X))

    def test_family_default_and_menus(self):
        """The first menu lists the Unsloth family without [experimental]; its size menu puts UD-IQ4_XS first (the
        default, Enter) and UD-Q4_K_XL second, still marked EXPERIMENTAL; --model names are unchanged."""
        code, out, cfg = self.main(["--context", "8192", "--vision", "no"], version="0.1.38", model=False,
                                   family=False, answers={"Which model": "4"}, m=X)
        self.assertEqual(code, 0, out)
        fam_line = next(ln for ln in out.splitlines() if ln.startswith("  4) "))
        self.assertIn("Qwen3.8-Flash-Next (Unsloth)", fam_line)
        self.assertIn("UD-IQ4_XS: a 94 GB download", fam_line)
        self.assertNotIn("[experimental]", out)
        size_lines = [ln for ln in out.splitlines() if ln.startswith(("  1) UD-", "  2) UD-"))]
        self.assertEqual(len(size_lines), 2, out)
        self.assertTrue(size_lines[0].startswith("  1) UD-IQ4_XS ~4-bit"), size_lines[0])
        self.assertNotIn("EXPERIMENTAL", size_lines[0])
        self.assertTrue(size_lines[1].startswith("  2) UD-Q4_K_XL 4-bit"), size_lines[1])
        self.assertIn("EXPERIMENTAL", size_lines[1])
        self.assertTrue(any("Which size" in q and "[1]" in q for q in self.asked), self.asked)
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-unsloth-ud-iq4_xs")
        code, out, cfg = self.main(["--context", "8192"], version="0.1.38", model=False, m=X)   # --yes: same default
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-unsloth-ud-iq4_xs")
        for m, name in ((M, "qwen3.8-flash-next-unsloth-ud-q4_k_xl"), (X, "qwen3.8-flash-next-unsloth-ud-iq4_xs")):
            code, out, cfg = self.main(["--context", "8192"], version="0.1.38", m=m)        # --model by name
            self.assertEqual((code, cfg["model_name"]), (0, name), out)

    def test_images(self):
        """Images on: the original model's image encoder (the same base model), the vision flags in the config."""
        code, out, cfg = self.main(["--context", "8192", "--vision", "yes"], version="0.1.38", m=X)
        self.assertEqual(code, 0, out)
        self.assertNotIn("images are not available", out)
        self.assertIn("images: on", out)
        self.assertEqual(self.downloads[-1], setup.hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF")
                         + "mmproj-Qwen3.8-Flash-Next-BF16.gguf")
        self.assertIn("--vision", cfg["args"])
        self.assertTrue(cfg["vision"]["model"].endswith("UD-IQ4_XS-00001-of-00003.gguf"))
        self.assertIn("--resident-budget-gib", cfg["args"])

    def test_check_verdict(self):
        code, out, _ = self.main(["--check"], m=X)
        self.assertEqual(code, 0, out)
        line = next(ln for ln in out.splitlines() if ln.strip().startswith(X))
        self.assertIn("fits with 40 GiB of its experts in RAM", line)
        self.assertNotIn("EXPERIMENTAL", line)
        self.assertIn("EXPERIMENTAL", next(ln for ln in out.splitlines() if ln.strip().startswith(M)))

    def test_check_below_every_floor(self):
        # #977: 16 GB fits no size; the RAM line is not [ok] and the verdict says so
        code, out, _ = self.main(["--check"], ram=16.0, m=X)
        self.assertEqual(code, 1, out)
        self.assertNotIn("[ok] RAM", out)
        self.assertIn("[!]  RAM: 16 GB (less than", out)
        self.assertIn("This PC cannot run Strata yet", out)
        self.assertNotIn("This PC can run Strata", out)

    def test_engine_0138(self):
        code, out, cfg = self.main(["--context", "8192"], version="0.1.37", m=X)
        self.assertEqual(code, 1)
        self.assertIn("needs engine 0.1.38 or newer; this one is 0.1.37", out)
        self.assertEqual(self.downloads, [])

    def test_amd_is_not_asked(self):
        r9700 = [{"index": 0, "name": "AMD Radeon AI PRO R9700", "vram_gb": 31.9, "arch": "gfx1201",
                  "driver": "amdgpu"}]
        with mock.patch.object(setup, "get_prebuilt_hip", lambda *a, **k: self.t / "engine"), \
                mock.patch.object(setup, "build_engine_hip", lambda *a, **k: self.t / "engine"):  # Linux compiles it
            code, out, cfg = self.main(["--context", "8192", "--backend", "hip"], version="0.1.38", amd=r9700, m=X)
        self.assertNotIn("has not been run on AMD cards yet", out)
        self.assertIn("RAM budget: 40 GiB of UD-IQ4_XS's experts", out)


class StrixHaloCache(Base):
    """#1715: on a unified-memory APU the expert cache and the RAM budget are the same RAM: setup writes a cache size
    that leaves the OS room, not `--expert-cache auto` (it OOM-killed a 128 GB Strix Halo with the desktop open)."""

    HALO = [{"index": 0, "name": "AMD Radeon 8060S", "vram_gb": 2.0, "arch": "gfx1151", "driver": "amdgpu",
             "uma": True, "dedicated_gb": 2.0, "shared_gb": 112.0}]

    def run_halo(self, ram, argv=("--context", "8192"), amd=None):
        with mock.patch.object(setup, "get_prebuilt_hip", lambda *a, **k: self.t / "engine"),                 mock.patch.object(setup, "WIN", False),                 mock.patch.object(setup, "build_engine_hip", lambda *a, **k: self.t / "engine"):
            return self.main([*argv, "--backend", "hip"], version="0.1.41", amd=self.HALO if amd is None else amd,
                             ram=ram, m="UD-IQ4_XS")

    def test_cache_is_a_number_that_leaves_room(self):
        code, out, cfg = self.run_halo(124.0)
        self.assertEqual(code, 0, out)
        args = cfg["args"]
        self.assertEqual(args[args.index("--resident-budget-gib") + 1], "55")
        slots = int(args[args.index("--expert-cache") + 1])
        # 124 - 24 (left) - 55 (budget) = 45 GiB of 1.1 x the average expert (59.5 GB / 24576)
        self.assertEqual(slots, int(45 / (59.5e9 / setup.UMA_EXPERTS * 1.1 / 2**30)))
        self.assertTrue(0 < slots < setup.UMA_EXPERTS)
        self.assertIn("expert cache: %d slots (not auto)" % slots, out)

    def test_never_more_than_all_the_experts(self):
        self.assertEqual(setup.uma_expert_cache("UD-IQ4_XS", 512.0, 55), setup.UMA_EXPERTS)

    def test_no_room_keeps_auto_and_says_so(self):
        self.assertIsNone(setup.uma_expert_cache("UD-IQ4_XS", 64.0, 40))
        code, out, cfg = self.run_halo(64.0, ("--context", "8192", "--resident-budget-gib", "40"))
        self.assertEqual(code, 0, out)
        args = cfg["args"]
        self.assertEqual(args[args.index("--expert-cache") + 1], "auto")
        self.assertIn("leaves almost none for the expert cache", out)

    def test_a_discrete_card_keeps_auto(self):
        r9700 = [{"index": 0, "name": "AMD Radeon AI PRO R9700", "vram_gb": 31.9, "arch": "gfx1201",
                  "driver": "amdgpu"}]
        code, out, cfg = self.run_halo(124.0, amd=r9700)
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["args"][cfg["args"].index("--expert-cache") + 1], "auto")


class LayerSplit(unittest.TestCase):
    """#498: UD-Q4_K_XL across GPUs - asked at setup (one GPU by default), and a start with --gpus no longer keeps
    the RAM budget the engine refused with a split (it exited with code 2).  #642: from RESIDENT_SPLIT_ENGINE the
    budget is kept on the split."""

    def setUp(self):
        # the behaviour before RESIDENT_SPLIT_ENGINE (an engine from before 0.1.40); the tests of the newer engine
        # patch MIN_ENGINE themselves
        p = mock.patch.object(setup, "MIN_ENGINE", (0, 1, 39))
        p.start()
        self.addCleanup(p.stop)
        sys.path.insert(0, str(ROOT / "tools"))
        from test_setup_golden import card
        self.found = [card(i, "NVIDIA GeForce RTX 3090", 24.0, "86") for i in range(2)]

    def test_asked_one_gpu_by_default(self):
        from test_setup_golden import install
        argv = ["--family", "unsloth", "--model", M, "--context", "32768", "--no-start"]
        code, out, cfg, asked = install(165.0, self.found, argv, answers={"which GPUs?": "1", "Which GPUs?": "1"})
        self.assertEqual(code, 0, out)
        self.assertTrue(any(f"{M}: which GPUs?" in q and "[1]" in q for q in asked), asked)
        self.assertEqual(cfg["gpu"], 0)
        self.assertIn("--resident-budget-gib", cfg["args"])
        code, out, cfg, asked = install(165.0, self.found, argv, answers={f"{M}: which GPUs?": "2"})
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertEqual(cfg["layer_split"], "auto")
        self.assertNotIn("--resident-budget-gib", cfg["args"])

    def start(self, ram, gpu=(0, 1), offered=None):
        from test_setup_risk import run
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "strata.exe"
            exe.write_bytes(b"")
            p = Path(d) / "strata-unsloth-ud-q4_k_xl.json"
            before = {"exe": str(exe), "args": ["--pack", "p", "--resident-budget-gib", "40", "--kv", "int8"],
                      "gpu": 0, "gpus_asked": offered is None}
            p.write_text(json.dumps(before))
            call = mock.Mock(return_value=0)
            with mock.patch.object(setup, "gpus", lambda: self.found), \
                    mock.patch.object(setup, "ram_gb", lambda: ram), \
                    mock.patch.object(setup, "engine_runs_on", lambda g: True), \
                    mock.patch.object(setup, "ensure_engine_for", lambda cards, path, cfg, yes: cfg), \
                    mock.patch.object(setup, "is_wsl", lambda: False), \
                    mock.patch.object(setup.subprocess, "call", call):
                code, out, asked = run(setup.start, p, None, list(gpu) if gpu else None, False, offered is None,
                                       stdin=offered)
            return code, out, json.loads(p.read_text()), call.called

    def test_start_with_gpus_drops_the_budget(self):
        code, out, cfg, started = self.start(165.0)
        self.assertIsNone(code, out)
        self.assertTrue(started)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertEqual(cfg["args"], ["--pack", "p", "--kv", "int8"])   # #1447: no helper cache
        self.assertIn("no RAM budget", out)

    def test_start_with_gpus_without_the_ram_is_asked(self):
        # #737: a warning and a question (default no), not a wall; --gpus with --yes is the consent
        code, out, cfg, started = self.start(63.7, offered="n")          # asked: "n" stops, nothing saved
        self.assertEqual(code, 1)
        self.assertFalse(started)
        self.assertIn("cannot share its RAM budget across GPUs", out)
        self.assertIn("--gpu N", out)
        self.assertEqual(cfg["gpu"], 0)
        self.assertIn("--resident-budget-gib", cfg["args"])
        code, out, cfg, started = self.start(63.7)                       # --gpus + --yes: goes on, no budget
        self.assertIsNone(code, out)
        self.assertTrue(started)
        self.assertIn("cannot share its RAM budget across GPUs", out)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertNotIn("--resident-budget-gib", cfg["args"])

    def test_offered_at_a_start(self):
        code, out, cfg, _ = self.start(165.0, gpu=None, offered="")    # Enter: one GPU (the default here)
        self.assertIsNone(code, out)
        self.assertEqual(cfg["gpu"], 0)
        self.assertIn("--resident-budget-gib", cfg["args"])
        self.assertIn("about twice as fast in #498", out)
        code, out, cfg, _ = self.start(165.0, gpu=None, offered="y")
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertNotIn("--resident-budget-gib", cfg["args"])
        code, out, cfg, _ = self.start(63.7, gpu=None, offered="y")    # not offered: the RAM does not hold it
        self.assertEqual(cfg["gpu"], 0)
        self.assertNotIn("Use both", out)

    def test_resident_split_engine_keeps_the_budget(self):
        """#642: a start with --gpus keeps the budget at any RAM; offered both cards (Enter: one, as before)."""
        with mock.patch.object(setup, "MIN_ENGINE", setup.RESIDENT_SPLIT_ENGINE):
            for ram in (165.0, 63.7):
                code, out, cfg, started = self.start(ram)
                self.assertIsNone(code, out)
                self.assertTrue(started)
                self.assertEqual(cfg["gpu"], [0, 1])
                self.assertEqual(cfg["args"], ["--pack", "p", "--resident-budget-gib", "40", "--kv", "int8"])
                self.assertNotIn("no RAM budget", out)
                self.assertNotIn("cannot share its RAM budget", out)
            code, out, cfg, _ = self.start(63.7, gpu=None, offered="")    # offered at 64 GB too; Enter: one GPU
            self.assertIsNone(code, out)
            self.assertIn("On both it keeps the budget", out)
            self.assertIn("staying on one GPU", out)
            self.assertEqual(cfg["gpu"], 0)
            code, out, cfg, _ = self.start(63.7, gpu=None, offered="y")
            self.assertEqual(cfg["gpu"], [0, 1])
            self.assertIn("--resident-budget-gib", cfg["args"])


if __name__ == "__main__":
    unittest.main()
