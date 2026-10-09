"""Tests for setup.py's Qwen3.6-35B-A3B family (--family qwen36): the two pinned single-file sizes with their sizes
and SHA-256, the engine arguments (no --ple-gguf, --mtp = the model file with the shipped draft vocabulary, its own
expert profile), no MTP download, the data folder's names, the RAM-fit suggestion on 16/24/32/64 GB PCs, the AMD
warning and one GPU.  Mocked - no GPU, no downloads, nothing written outside a temp folder.

    python tools/test_setup_qwen36.py
"""
from __future__ import annotations

import contextlib
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

REV = "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d"
BASE = f"https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/{REV}/"
IQ4, IQ3 = "Qwen3.6-UD-IQ4_XS", "Qwen3.6-UD-IQ3_S"
F_IQ4, F_IQ3 = "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf", "Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
# what ram_gb() reports (GiB) on PCs sold as 16, 24, 32 and 64 GB
RAM16, RAM24, RAM32, RAM64 = 15.6, 23.4, 31.3, 62.7


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


def arg(cfg, flag):
    a = cfg["args"]
    return a[a.index(flag) + 1]


class Tables(unittest.TestCase):
    def test_family(self):
        fam = setup.FAMILIES["qwen36"]
        self.assertEqual(list(setup.FAMILIES)[0], "qwen")                 # never the default family
        self.assertEqual(list(setup.FAMILIES)[-2:], ["qwen36", "ornith"])  # the menu's numbers stay as they were
        self.assertEqual(fam["tag"], "qwen36-")
        self.assertEqual(fam["name"], "qwen3.6-35b-a3b")
        self.assertEqual(fam["hf"], BASE)
        self.assertEqual(fam["pack_args"], ["--compat-bf16"])
        self.assertEqual(fam["profile"], "expert-profile-qwen36.bin")
        self.assertIs(fam["vision"], False)
        self.assertTrue(fam["own_mtp"] and fam["one_gpu"] and fam["nvidia_only"])
        self.assertIs(fam["ple"], False)
        self.assertIn("Apache 2.0", fam["license"])

    def test_sizes(self):
        self.assertEqual([m for m in setup.MODELS if "qwen36" in setup.MODELS[m].get("families", ())], [IQ4, IQ3])
        self.assertEqual((setup.size_of(IQ4), setup.size_of(IQ3)), ("UD-IQ4_XS", "UD-IQ3_S"))
        self.assertEqual(setup.MODELS[IQ4]["arena_gb"], 15.2)              # 14.17 GiB, measured
        self.assertAlmostEqual(setup.MODELS[IQ4]["arena_gb"], 14.17 * 2**30 / 1e9, places=1)
        self.assertEqual(setup.MODELS[IQ3]["arena_gb"], 12.9)              # measured (11.99 GiB)
        self.assertAlmostEqual(setup.MODELS[IQ4]["download_gb"], 18209036576 / 1e9, places=1)
        self.assertAlmostEqual(setup.MODELS[IQ3]["download_gb"], 15346432288 / 1e9, places=1)
        for m in (IQ4, IQ3):
            self.assertEqual(setup.MODELS[m]["engine"], setup.QWEN36_ENGINE)
            self.assertFalse(setup.MODELS[m].get("budget"))
        # the engine version required: this source tree's (CMakeLists.txt)
        self.assertEqual(setup.QWEN36_ENGINE, tuple(int(x) for x in setup.source_version().split(".")))
        # Unsloth's Flash-Next UD-IQ4_XS keeps its key and its numbers
        self.assertEqual(setup.MODELS["UD-IQ4_XS"]["families"], ("unsloth",))
        self.assertEqual(setup.MODELS["UD-IQ4_XS"]["arena_gb"], 59.5)

    def test_files_and_pins(self):
        fam = setup.FAMILIES["qwen36"]
        self.assertEqual([setup.model_file(fam, m, 1) for m in (IQ4, IQ3)], [F_IQ4, F_IQ3])
        self.assertEqual([setup.model_shards(fam, m) for m in (IQ4, IQ3)], [1, 1])
        self.assertEqual(fam["sha256"], {
            F_IQ4: (18209036576, "df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf"),
            F_IQ3: (15346432288, "ab639a7f330f96c47d3e6c2dd2d6445182e7b763e17e6048dc850a71bbc9f27f")})
        self.assertEqual(setup.HF_REVISIONS["unsloth/Qwen3.6-35B-A3B-MTP-GGUF"], REV)

    def test_model_key(self):
        self.assertEqual(setup.model_key("qwen36", "UD-IQ4_XS"), IQ4)
        self.assertEqual(setup.model_key("qwen36", "ud-iq3_s"), IQ3)
        self.assertEqual(setup.model_key("qwen36", IQ4), IQ4)
        self.assertEqual(setup.model_key("unsloth", "UD-IQ4_XS"), "UD-IQ4_XS")
        self.assertIsNone(setup.model_key("qwen", "UD-IQ3_S"))
        self.assertIsNone(setup.model_key("qwen36", "IQ3_S"))
        for size in ("UD-IQ4_XS", "UD-IQ3_S", IQ4):
            self.assertIn(size, setup.model_choices())

    def test_gguf_names(self):
        self.assertEqual(setup.gguf_choice(F_IQ4), ("qwen36", "UD-IQ4_XS"))
        self.assertEqual(setup.gguf_choice(F_IQ3), ("qwen36", "UD-IQ3_S"))
        self.assertIsNone(setup.gguf_unsupported(F_IQ3))
        self.assertEqual(setup.gguf_choice("Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"), ("unsloth", "UD-IQ4_XS"))

    def test_config_names(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "strata-qwen36-ud-iq4_xs.json"
            p.write_text(json.dumps({"args": ["--max-context", "32768", "--kv", "int8"]}))
            ch = setup.choices_from_config(p)
            self.assertEqual((ch["family"], ch["model"], ch["context"]), ("qwen36", IQ4, 32768))
            p = Path(t) / "strata-qwen36-ud-iq3_s.json"
            p.write_text(json.dumps({"args": []}))
            self.assertEqual(setup.choices_from_config(p)["model"], IQ3)
            p = Path(t) / "strata-unsloth-ud-iq4_xs.json"                 # Unsloth's Flash-Next file is not taken
            p.write_text(json.dumps({"args": []}))
            self.assertEqual((setup.choices_from_config(p)["family"], setup.choices_from_config(p)["model"]),
                             ("unsloth", "UD-IQ4_XS"))
        self.assertTrue(setup.one_gpu_family(Path("strata-qwen36-ud-iq4_xs.json")))
        self.assertFalse(setup.one_gpu_family(Path("strata-iq2_xs.json")))


class RamFit(unittest.TestCase):
    """Qwen3.6 is suggested where no Flash-Next size reaches its RAM line (16-24 GB); elsewhere nothing changes."""

    def test_suggested_family(self):
        self.assertTrue(setup.qwen36_recommended(RAM16))
        self.assertTrue(setup.qwen36_recommended(RAM24))
        self.assertFalse(setup.qwen36_recommended(RAM32))
        self.assertFalse(setup.qwen36_recommended(RAM64))

    def test_suggested_size(self):
        self.assertEqual(setup.qwen36_size(RAM16), IQ3)                   # in the low-RAM mode
        self.assertEqual(setup.qwen36_size(RAM24), IQ3)
        self.assertEqual(setup.qwen36_size(RAM32), IQ4)
        self.assertEqual(setup.qwen36_size(RAM64), IQ4)

    def test_low_ram_rules(self):
        self.assertFalse(setup.low_ram_needed(IQ4, RAM32))                # comfortable on 32 GB
        self.assertTrue(setup.low_ram_needed(IQ4, RAM24))
        self.assertFalse(setup.low_ram_needed(IQ3, RAM24))                # 24 GB: IQ3_S without the low-RAM mode
        self.assertTrue(setup.low_ram_needed(IQ3, RAM16))
        # 16 GB with UD-IQ3_S's measured 12.9 GB of experts: a 12 GB card fits the low-RAM rule, an 8 GB card misses
        # it by ~0.5 GB (setup then stops and says --model ... --yes)
        self.assertTrue(setup.low_ram_fits(IQ3, RAM16, 11.9))
        self.assertFalse(setup.low_ram_fits(IQ3, RAM16, 7.99))
        self.assertFalse(setup.low_ram_fits(IQ3, 8.0, 7.99))


class FakeGGUF:
    """gguf_reader.GGUFFile: Flash-Next's shard 2 holds the PLE table; Qwen3.6's file has none."""
    def __init__(self, path):
        names = ["per_layer_token_embd.weight"] if "00002-of" in str(path) else ["blk.0.ffn_up_exps.weight"]
        self.tensors = [types.SimpleNamespace(name=n) for n in names]


def nvidia(n=1, vram=11.9):
    return [{"index": i, "name": "NVIDIA GeForce RTX 4070", "vram_gb": vram, "arch": "89", "driver": "580.97"}
            for i in range(n)]


class Base(unittest.TestCase):
    """setup.main(), every outside effect mocked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        (self.t / "data" / "mtp" / "rt").mkdir(parents=True)              # Flash-Next's draft layer (for --family qwen)
        (self.t / "data" / "mtp" / "rt" / "experts.bin").write_bytes(b"")
        self.downloads, self.runs, self.verified, self.asked = [], [], [], []
        self.vram_free = None                                             # nvidia-smi's memory.free: unknown

    def tearDown(self):
        self.tmp.cleanup()

    def main(self, argv, ram=RAM64, version="0.1.40", found=None, amd=(), answers=None, free=500.0, source="local",
             runs_small=True, f16_mtp=False):
        """answers: None = --yes; else {words of a question: its answer} (Enter for the others).
        -> (exit code, printed text, the config written or None)."""
        def fake_input(prompt=""):
            if answers is None:
                raise AssertionError(f"asked {prompt!r}")
            self.asked.append(prompt)
            return next((v for k, v in answers.items() if k in prompt), "")

        eng = self.t / "engine"
        eng.mkdir(exist_ok=True)
        (eng / "BUILD.json").write_text(json.dumps({"version": version, "source": source}))
        # runs_small: qwen35moe's engine-owned keys; f16_mtp: the additional patched draft-loader capability.
        # Separate markers let Huihui's tests distinguish stock PR engines from patched ones at the same version.
        mtp_mark = setup.MTP_F16_ENGINE_MARK if f16_mtp else b""
        (eng / setup.EXE).write_bytes(b"\0" + (setup.SMALL_ENGINE_MARK if runs_small else b"qwen4exp.block_count") + mtp_mark)
        self.builds = []

        def fake_build(*a, **k):                                         # a compile here: this checkout's engine
            self.builds.append(a)
            if json.loads((eng / "BUILD.json").read_text()).get("source") == "local":
                return eng                                               # compiled here, same source: kept as is
            (eng / "BUILD.json").write_text(json.dumps({"version": "0.1.40", "source": "local"}))
            (eng / setup.EXE).write_bytes(setup.SMALL_ENGINE_MARK + mtp_mark)
            return eng
        found = nvidia() if found is None else found
        for old in self.t.glob("strata-*.json"):                          # this run's config only
            old.unlink()

        def fake_download(url, dst, what=None):
            self.downloads.append(url)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)

        patches = [
            mock.patch.object(setup, "ROOT", self.t),
            mock.patch.object(setup, "GPU_PICK", None),
            mock.patch.object(setup, "data_folder", lambda d: (self.t / "data", [])),
            mock.patch.object(setup, "installed_configs", lambda: []),
            mock.patch.object(setup, "gpus", lambda: found),
            mock.patch.object(setup, "gpu_free_gb", lambda i: self.vram_free),
            mock.patch.object(setup, "amd_gpus", lambda *a: list(amd)),
            mock.patch.object(setup, "ram_gb", lambda: ram),
            mock.patch.object(setup, "cpu_info", lambda: ("Test CPU", True, False)),
            mock.patch.object(setup, "cpu_cores", lambda: None),
            mock.patch.object(setup, "page_file_gb", lambda: 16.0),
            mock.patch.object(setup, "free_gb", lambda p: free),
            mock.patch.object(setup, "rotational_disk", lambda p: None),
            mock.patch.object(setup, "is_wsl", lambda: False),
            mock.patch.object(setup, "linux_desktop", lambda env=None: False),
            mock.patch.object(setup, "pip_install", lambda *a, **k: None),
            mock.patch.object(setup, "get_llama_cpp", lambda: self.t / "llama.cpp"),
            mock.patch.object(setup, "get_prebuilt", lambda *a, **k: eng),
            mock.patch.object(setup, "build_engine_hip", lambda *a, **k: eng),
            mock.patch.object(setup, "build_engine", fake_build),
            mock.patch.object(setup, "pip_cuda_libs", lambda *a, **k: None),
            mock.patch.object(setup, "hipblaslt_table", lambda *a, **k: None),
            mock.patch.object(setup, "download", fake_download),
            mock.patch.object(setup, "check_shards", lambda shards: None),
            mock.patch.object(setup, "verify_sha256", lambda s, size, sha: self.verified.append((s.name, size, sha))),
            mock.patch.object(setup, "run", lambda cmd, *a, **k: self.runs.append([str(x) for x in cmd])),
            mock.patch.object(setup, "refresh_draft_vocab", lambda *a, **k: None),
            mock.patch.object(setup, "mtp_corrupt", lambda *a, **k: False),
            mock.patch.object(setup, "write_run_script", lambda tag, cfg, port, *_: self.t / f"run-{tag}.sh"),
            mock.patch.object(setup, "saved_calibration", lambda cfg: None),
            mock.patch.object(setup, "start", mock.Mock(side_effect=AssertionError("started"))),
            mock.patch.dict(sys.modules, {"gguf_reader": types.SimpleNamespace(GGUFFile=FakeGGUF)}),
            mock.patch.object(sys, "argv", ["setup.py", *(["--yes"] if answers is None else []), "--no-start",
                                            "--models-dir", str(self.t / "data" / "models"), *argv]),
            mock.patch("builtins.input", fake_input),
        ]
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            code, out = quiet(setup.main)
        cfgs = sorted(self.t.glob("strata-*.json"))
        self.assertLessEqual(len(cfgs), 1, cfgs)
        self.cfg_path = cfgs[0] if cfgs else None
        return code, out, (json.loads(cfgs[0].read_text()) if cfgs else None)


class Install(Base):
    def test_iq4_xs_engine_args(self):
        code, out, cfg = self.main(["--setup", "--family", "qwen36", "--model", "UD-IQ4_XS"])
        self.assertEqual(code, 0, out)
        data = self.t / "data"
        gguf = data / "models" / "qwen36-UD-IQ4_XS" / F_IQ4
        pack = data / "packs" / "qwen36-ud-iq4_xs"
        self.assertEqual(self.cfg_path.name, "strata-qwen36-ud-iq4_xs.json")
        self.assertEqual(self.downloads, [BASE + F_IQ4])                  # one file; no MTP, no image encoder
        self.assertEqual(self.verified, [(F_IQ4, *setup.QWEN36_FILES[F_IQ4])])
        self.assertEqual(self.runs, [[sys.executable, str(self.t / "tools" / "iq_pack.py"), "--gguf", str(gguf),
                                      "--out", str(pack), "--compat-bf16"]])   # no mtp_fetch/mtp_pack/mtp_rt
        self.assertEqual(cfg["args"], [
            "--pack", str(pack), "--native", str(gguf),
            "--expert-profile", str(self.t / "data" / "expert-profile-qwen36.bin"), "--expert-cache", "auto",
            "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
            "--mtp", str(gguf), "--mtp-draft-vocab", str(self.t / "data" / "draft_vocab.bin"),
            "--max-context", "32768", "--kv", "int8"])
        self.assertEqual(cfg["model_name"], "qwen3.6-35b-a3b-ud-iq4_xs")
        self.assertEqual(cfg["tokenizer"], str(pack / "tokenizer"))
        self.assertEqual(cfg["log"], str(self.t / "strata-qwen36-ud-iq4_xs.log"))
        self.assertNotIn("vision", cfg)
        self.assertIn("MTP draft layer: the model file's own", out)
        self.assertIn("Apache 2.0", out)

    def test_iq3_s(self):
        code, out, cfg = self.main(["--family", "qwen36", "--model", "UD-IQ3_S"], ram=RAM24)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.cfg_path.name, "strata-qwen36-ud-iq3_s.json")
        self.assertEqual(self.downloads, [BASE + F_IQ3])
        self.assertTrue(arg(cfg, "--native").endswith("models/qwen36-UD-IQ3_S/" + F_IQ3))
        self.assertEqual(arg(cfg, "--mtp"), arg(cfg, "--native"))
        self.assertEqual(cfg["model_name"], "qwen3.6-35b-a3b-ud-iq3_s")
        self.assertNotIn("--ple-gguf", cfg["args"])

    def test_draft_vocab_choice(self):
        code, out, cfg = self.main(["--family", "qwen36", "--model", "UD-IQ4_XS", "--draft-vocab", "en"])
        self.assertEqual(code, 0, out)
        self.assertEqual(arg(cfg, "--mtp-draft-vocab"), str(self.t / "data" / "draft_vocab_en.bin"))
        self.assertEqual(cfg["draft_vocab"], "en")

    def test_files_already_there(self):
        # a hand-made install: the file in models/qwen36-UD-IQ4_XS and a finished pack: nothing is downloaded or built
        gguf = self.t / "data" / "models" / "qwen36-UD-IQ4_XS" / F_IQ4
        gguf.parent.mkdir(parents=True)
        gguf.write_bytes(b"")
        setup.mark(gguf)
        pack = self.t / "data" / "packs" / "qwen36-ud-iq4_xs"
        (pack / "tokenizer").mkdir(parents=True)
        (pack / "tokenizer" / "vocab.json").write_text("{}")
        (pack / "native_experts.txt").write_text("")
        code, out, cfg = self.main(["--family", "qwen36", "--model", "UD-IQ4_XS"])
        self.assertEqual(code, 0, out)
        self.assertEqual((self.downloads, self.runs), ([], []))
        self.assertEqual(len(self.verified), 1)                           # the SHA-256 is still checked (once)
        self.assertEqual(arg(cfg, "--pack"), str(pack))

    def test_wrong_size_for_the_family(self):
        code, out, cfg = self.main(["--family", "qwen36", "--model", "IQ3_S"])
        self.assertEqual(code, 1)
        self.assertIn("choose one of: UD-IQ4_XS, UD-IQ3_S", out)
        self.assertIsNone(cfg)

    def test_old_engine_stops_before_the_download(self):
        # an engine compiled here from a checkout without qwen35moe (setup does not recompile it: same source)
        code, out, cfg = self.main(["--family", "qwen36", "--model", "UD-IQ4_XS"], version="0.1.39", runs_small=False)
        self.assertEqual(code, 1)
        self.assertIn("needs an engine with the qwen35moe support", out)
        self.assertEqual(self.downloads, [])

    def test_ready_made_engine_without_it_is_compiled_here(self):
        # the published 0.1.40 engines (Windows) predate qwen35moe: setup compiles this checkout's engine instead
        code, out, cfg = self.main(["--family", "qwen36", "--model", "UD-IQ4_XS"], source="release", runs_small=False)
        self.assertEqual(code, 0, out)
        self.assertEqual(len(self.builds), 1)
        self.assertIn("does not run Qwen3.6-35B-A3B yet: compiling the engine on this PC instead", out)
        self.assertEqual(cfg["exe"], str(self.t / "engine" / setup.EXE))

    def test_ready_made_engine_with_it_is_used(self):
        code, out, cfg = self.main(["--family", "qwen36", "--model", "UD-IQ4_XS"], source="release")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.builds, [])

    def test_flash_next_keeps_the_ready_made_engine(self):
        code, out, cfg = self.main(["--family", "qwen", "--model", "IQ2_XS"], source="release", runs_small=False)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.builds, [])

    def test_no_images(self):
        code, out, cfg = self.main(["--family", "qwen36", "--model", "UD-IQ4_XS", "--vision", "yes"])
        self.assertEqual(code, 0, out)
        self.assertIn("images are not available", out)
        self.assertNotIn("vision", cfg)
        self.assertEqual(self.downloads, [BASE + F_IQ4])

    def test_disk_space_counts_no_mtp(self):
        code, out, cfg = self.main(["--family", "qwen36", "--model", "UD-IQ4_XS"], free=19.0)
        self.assertEqual(code, 1)
        self.assertIn("not enough free disk space", out)
        code, out, cfg = self.main(["--family", "qwen36", "--model", "UD-IQ4_XS"], free=19.5)   # 18.2 + 1
        self.assertEqual(code, 0, out)


class Suggestion(Base):
    """No --family: Qwen3.6 on 16 and 24 GB, the Flash-Next default as before on 32 and 64 GB."""

    def test_16_gb(self):
        # an 8 GB card: UD-IQ3_S (12.9 GB of experts, measured) is just under the low-RAM rule on 16 GB
        code, out, cfg = self.main([], ram=RAM16, found=nvidia(vram=7.99))
        self.assertNotEqual(code, 0, out)
        self.assertIn("--yes", out)

    def test_16_gb_12_gb_card(self):
        code, out, cfg = self.main([], ram=RAM16, found=nvidia(vram=11.9))
        self.assertEqual(code, 0, out)
        self.assertEqual(self.cfg_path.name, "strata-qwen36-ud-iq3_s.json")
        self.assertTrue("--mmap-experts" in cfg["args"] or "--resident-experts" in cfg["args"], cfg["args"])
        self.assertNotIn("--ple-gguf", cfg["args"])

    def test_16_gb_12_gb_card_maps_the_rest(self):
        # the experts a 12 GB card does not hold (~6 GB) leave 15.6 GB short of the resident variant's 10 GB of room:
        # the plain low-RAM mode, read through the file cache
        code, out, cfg = self.main([], ram=RAM16, found=nvidia(vram=11.9))
        self.assertEqual(code, 0, out)
        self.assertEqual(self.cfg_path.name, "strata-qwen36-ud-iq3_s.json")
        self.assertIn("--mmap-experts", cfg["args"])
        self.assertIn([sys.executable, str(self.t / "tools" / "iq_pack.py"), "--gguf",
                       arg(cfg, "--native"), "--out", arg(cfg, "--pack"), "--experts-bin"], self.runs)

    def test_24_gb(self):
        code, out, cfg = self.main([], ram=RAM24)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.cfg_path.name, "strata-qwen36-ud-iq3_s.json")
        for flag in ("--mmap-experts", "--resident-experts"):
            self.assertNotIn(flag, cfg["args"])

    def test_24_gb_menus(self):
        code, out, cfg = self.main([], ram=RAM24, answers={})            # Enter everywhere: the recommendations
        self.assertEqual(code, 0, out)
        line = next(ln for ln in out.splitlines() if "Qwen3.6-35B-A3B" in ln and ln.strip()[:2].rstrip(")").isdigit())
        self.assertIn("(recommended for 23 GB of RAM: UD-IQ3_S)", line)
        family_q = next(q for q in self.asked if "Which model?" in q)
        self.assertIn(f"[{list(setup.FAMILIES).index('qwen36') + 1}]", family_q)
        size_q = next(q for q in self.asked if "Which size?" in q)
        self.assertIn("[2]", size_q)                                      # 1) UD-IQ4_XS  2) UD-IQ3_S
        self.assertIn("  1) UD-IQ4_XS ", out)
        self.assertIn("  2) UD-IQ3_S ~3.5-bit", out)
        self.assertEqual(self.cfg_path.name, "strata-qwen36-ud-iq3_s.json")

    def test_32_gb_default_unchanged(self):
        code, out, cfg = self.main([], ram=RAM32)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.cfg_path.name, "strata-q2_0.json")         # the original model, as before
        self.assertNotIn("recommended for 31 GB", out)

    def test_32_gb_qwen36_picks_iq4_xs(self):
        code, out, cfg = self.main(["--family", "qwen36"], ram=RAM32)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.cfg_path.name, "strata-qwen36-ud-iq4_xs.json")
        for flag in ("--mmap-experts", "--resident-experts"):
            self.assertNotIn(flag, cfg["args"])

    def test_64_gb_default_unchanged(self):
        code, out, cfg = self.main([], ram=RAM64)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.cfg_path.name, "strata-iq3_xxs.json")      # setup's pick from 60 GB, as before
        code, out, cfg = self.main(["--family", "qwen36"], ram=RAM64)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.cfg_path.name, "strata-qwen36-ud-iq4_xs.json")

    def test_check_16_gb(self):
        code, out, cfg = self.main(["--check"], ram=RAM16)
        self.assertEqual(code, 0, out)
        self.assertIn("suggests Qwen3.6-35B-A3B UD-IQ3_S (--family qwen36)", out)
        self.assertIn("This PC can run Strata", out)
        self.assertNotIn("[!]  RAM", out)
        code, out, cfg = self.main(["--check"], ram=RAM64)
        self.assertNotIn("--family qwen36", out)

    def test_flash_next_floor_unchanged_with_a_family(self):
        # --family qwen on 20 GB: the Flash-Next floor (the Coder's 32 GB) stops --yes as before
        code, out, cfg = self.main(["--family", "qwen"], ram=19.9, found=nvidia(vram=8.0))
        self.assertEqual(code, 1)
        self.assertIn("the smallest model (the Coder) needs about 32 GB", out)


class SmallCard(Base):
    """A card under 5.5 GB: setup recommends an 8K context and says what was measured there; a 6 GB card keeps 32K;
    Flash-Next keeps its "less than 12 GB" warning."""

    def test_4_gb_card(self):
        code, out, cfg = self.main(["--family", "qwen36"], ram=RAM32, found=nvidia(vram=3.99))
        self.assertEqual(code, 0, out)
        self.assertEqual(arg(cfg, "--max-context"), "8192")
        self.assertEqual(self.cfg_path.name, "strata-qwen36-ud-iq3_s.json")   # the smaller size (32 GB would take IQ4_XS)
        self.assertIn("at this model's floor", out)
        self.assertNotIn("it will be slow", out)

    def test_6_gb_card(self):
        code, out, cfg = self.main(["--family", "qwen36"], ram=RAM32, found=nvidia(vram=5.99))
        self.assertEqual(code, 0, out)
        self.assertEqual(arg(cfg, "--max-context"), "32768")
        self.assertEqual(self.cfg_path.name, "strata-qwen36-ud-iq4_xs.json")
        self.assertNotIn("at this model's floor", out)
        self.assertNotIn("it will be slow", out)

    def test_context_asked_is_kept(self):
        code, out, cfg = self.main(["--family", "qwen36", "--context", "32768"], ram=RAM32, found=nvidia(vram=3.99))
        self.assertEqual(code, 0, out)
        self.assertEqual(arg(cfg, "--max-context"), "32768")          # a recommendation, never forced (#406)

    def test_flash_next_keeps_its_warning(self):
        code, out, cfg = self.main(["--family", "qwen", "--model", "IQ2_XS"], ram=RAM64, found=nvidia(vram=7.99))
        self.assertIn("it will be slow", out)
        self.assertNotIn("at this model's floor", out)


class KvStreaming(Base):
    """From a 64K context, a 35B-A3B model's KV streaming (it attends to every cell: writing slows down past 32K tokens
    when streamed): from 128K with under 9 GB of VRAM free it is needed, and setup says so; otherwise setup asks and
    recommends no; --kv-streaming on|off answers it.  Flash-Next streams as before, unasked."""
    Q = "Turn on KV streaming?"

    def run_ctx(self, ctx, vram, free, answers=None, extra=()):
        self.vram_free = free
        return self.main(["--family", "qwen36", "--model", "UD-IQ4_XS", "--context", str(ctx), *extra], ram=RAM64,
                         found=nvidia(vram=vram), answers=answers)

    def asked_q(self):
        return [p for p in self.asked if self.Q in p]

    def test_asked_recommends_no(self):
        code, out, cfg = self.run_ctx(65536, 7.99, 7.9, answers={})        # Enter: the recommendation
        self.assertEqual(code, 0, out)
        self.assertEqual(self.asked_q(), [f"{self.Q} [n]: "])
        self.assertNotIn("--kv-resident", cfg["args"])
        self.assertIn("past 32K tokens", out)
        self.assertIn("the 8 GB card has 7.9 GB free now", out)
        self.assertIn("strata-qwen36-ud-iq4_xs.json's args, or run\n  setup again with --kv-streaming on|off.", out)
        self.assertIn("KV streaming off, as you chose: the KV cache stays in VRAM", out)

    def test_asked_yes(self):
        code, out, cfg = self.run_ctx(131072, 11.99, 11.6, answers={self.Q: "y"})
        self.assertEqual(code, 0, out)
        self.assertEqual(self.asked_q(), [f"{self.Q} [n]: "])
        self.assertEqual(arg(cfg, "--kv-resident"), "32768")
        self.assertIn("writing slows down past 32K tokens of context", out)

    def test_yes_takes_the_recommendation(self):
        code, out, cfg = self.run_ctx(65536, 11.99, 11.6)
        self.assertEqual(code, 0, out)
        self.assertNotIn("--kv-resident", cfg["args"])

    def test_needed_is_said_not_asked(self):
        for ctx, vram, free in ((131072, 7.99, 7.9), (262144, 11.99, 8.5)):   # free VRAM decides, not the card's size
            with self.subTest(ctx=ctx, free=free):
                self.asked.clear()
                code, out, cfg = self.run_ctx(ctx, vram, free, answers={})
                self.assertEqual(code, 0, out)
                self.assertFalse(self.asked_q())
                self.assertEqual(arg(cfg, "--kv-resident"), "32768")
                self.assertIn("KV streaming is needed here", out)
                self.assertIn(f"has {free:.1f} GB free now", out)
                self.assertIn("--kv-streaming on|off", out)
                self.assertIn("needed for this context with this much free VRAM", out)

    def test_not_needed_with_room_or_at_64k(self):
        for ctx, free in ((131072, 10.0), (65536, 6.0)):
            with self.subTest(ctx=ctx, free=free):
                self.asked.clear()
                code, out, cfg = self.run_ctx(ctx, 11.99, free, answers={})
                self.assertEqual(code, 0, out)
                self.assertEqual(len(self.asked_q()), 1)
                self.assertNotIn("is needed here", out)

    def test_the_option_answers_it(self):
        code, out, cfg = self.run_ctx(65536, 7.99, 7.9, answers={}, extra=("--kv-streaming", "on"))
        self.assertEqual(code, 0, out)
        self.assertFalse(self.asked_q())
        self.assertEqual(arg(cfg, "--kv-resident"), "32768")
        code, out, cfg = self.run_ctx(131072, 7.99, 7.9, answers={}, extra=("--kv-streaming", "off"))
        self.assertEqual(code, 0, out)
        self.assertFalse(self.asked_q())
        self.assertNotIn("--kv-resident", cfg["args"])
        self.assertIn("the first prompt fails without it", out)

    def test_flash_next_unasked(self):
        self.vram_free = 7.9
        code, out, cfg = self.main(["--family", "qwen", "--model", "IQ2_XS", "--context", "65536"], ram=RAM64,
                                   found=nvidia(vram=11.9), answers={})
        self.assertEqual(code, 0, out)
        self.assertFalse(self.asked_q())
        self.assertEqual(arg(cfg, "--kv-resident"), "32768")
        self.assertNotIn("writing slows down", out)


class FreeVram(unittest.TestCase):
    def test_parse(self):
        for text, want in (("6144\n", 6.0), ("", None), ("[N/A]\n", None)):
            with mock.patch.object(setup, "out", lambda cmd: text):
                self.assertEqual(setup.gpu_free_gb(0), want)


class Amd(Base):
    """The HIP engine compiles, but the qwen35moe batched prompt path is NVIDIA-only and it was never run on AMD."""
    CARD = [{"index": 0, "name": "AMD Radeon RX 7700 XT", "vram_gb": 12.0, "arch": "gfx1101", "driver": "amdgpu"}]

    def test_yes_with_the_family_goes_on(self):
        code, out, cfg = self.main(["--family", "qwen36", "--model", "UD-IQ4_XS", "--backend", "hip"],
                                   found=[], amd=self.CARD)
        self.assertEqual(code, 0, out)
        self.assertIn("Qwen3.6-35B-A3B has not been run on AMD cards yet", out)
        self.assertIn("installing Qwen3.6-35B-A3B UD-IQ4_XS on an AMD card, as you chose", out)
        self.assertEqual(cfg["backend"], "hip")
        self.assertNotIn("--ple-gguf", cfg["args"])

    def test_asked_default_no(self):
        code, out, cfg = self.main(["--family", "qwen36", "--backend", "hip"], found=[], amd=self.CARD, answers={})
        self.assertEqual(code, 1)
        self.assertTrue(any("Try it anyway?" in q and "[n]" in q for q in self.asked), self.asked)
        self.assertIn("Qwen3.6-35B-A3B is untested on AMD", out)
        self.assertEqual(self.downloads, [])
        self.assertIsNone(cfg)

    def test_check_says_untested(self):
        code, out, cfg = self.main(["--check", "--backend", "hip"], ram=RAM32, found=[], amd=self.CARD)
        self.assertEqual(code, 0, out)
        line = next(ln for ln in out.splitlines() if ln.strip().startswith(IQ4))
        self.assertIn("untested on AMD", line)


class Pascal(Base):
    """Below sm_75 the engine reads every prompt through the decode windows, and the model was never run there."""
    CARD = [{"index": 0, "name": "NVIDIA GeForce GTX 1080", "vram_gb": 8.0, "arch": "61", "driver": "535.104"}]

    def test_asked_default_no_before_the_download(self):
        code, out, cfg = self.main(["--family", "qwen36"], found=self.CARD, answers={})
        self.assertEqual(code, 1)
        self.assertTrue(any("Try it anyway?" in q and "[n]" in q for q in self.asked), self.asked)
        self.assertIn("Qwen3.6-35B-A3B is untested below sm_75", out)
        self.assertEqual(self.downloads, [])
        self.assertIsNone(cfg)

    def test_yes_with_the_model_goes_on(self):
        code, out, cfg = self.main(["--family", "qwen36", "--model", "UD-IQ3_S"], found=self.CARD)
        self.assertEqual(code, 0, out)
        self.assertIn("has not been run on GPU 0 (NVIDIA GeForce GTX 1080, 8 GB) (sm_61)", out)
        self.assertIn("installing Qwen3.6-35B-A3B UD-IQ3_S on GPU 0", out)

    def test_turing_is_not_asked(self):
        card = [{**self.CARD[0], "name": "NVIDIA GeForce RTX 2080 SUPER", "arch": "75", "driver": "580.97"}]
        code, out, cfg = self.main(["--family", "qwen36", "--model", "UD-IQ3_S"], found=card)
        self.assertEqual(code, 0, out)
        self.assertNotIn("sm_75", out.replace("(sm_75)", ""))
        self.assertNotIn("untested below", out)


class OneGpu(Base):
    def test_gpus_keeps_one(self):
        code, out, cfg = self.main(["--family", "qwen36", "--model", "UD-IQ4_XS", "--gpus", "0,1"],
                                   found=nvidia(2))
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], 0)
        self.assertNotIn("layer_split", cfg)
        self.assertNotIn("--remote-expert-opt", cfg["args"])
        self.assertIn("runs on one GPU for now", out)

    def test_not_offered_at_a_start(self):
        p = self.t / "strata-qwen36-ud-iq4_xs.json"
        p.write_text(json.dumps({"exe": "x", "args": []}))
        with mock.patch.object(setup, "gpus", mock.Mock(side_effect=AssertionError("looked for GPUs"))):
            code, out = quiet(setup.offer_together, p, {"args": []}, True)
        self.assertEqual(code, {"args": []})                              # unchanged, nothing asked
        self.assertEqual(out, "")

    def test_start_with_gpus_uses_the_first(self):
        exe = self.t / "strata"
        exe.write_bytes(b"")
        p = self.t / "strata-qwen36-ud-iq4_xs.json"
        p.write_text(json.dumps({"exe": str(exe), "args": ["--pack", "p"], "gpu": 0, "gpus_asked": True}))
        call = mock.Mock(return_value=0)
        with mock.patch.object(setup, "gpus", lambda: nvidia(2)), \
                mock.patch.object(setup, "ensure_engine_for", lambda cards, path, cfg, yes: cfg), \
                mock.patch.object(setup, "is_wsl", lambda: False), \
                mock.patch.object(setup.subprocess, "call", call):
            code, out = quiet(setup.start, p, None, [0, 1], False, True)
        self.assertEqual(code, 0, out)
        self.assertIn("runs on one GPU for now", out)
        self.assertEqual(json.loads(p.read_text())["gpu"], 0)
        cmd = call.call_args[0][0]
        self.assertEqual(cmd[cmd.index("--gpu") + 1], "0")


class DraftVocabAtStart(unittest.TestCase):
    def test_points_at_the_chosen_subset(self):
        data = setup.ROOT / "data"
        cfg = {"args": ["--mtp", "m.gguf", "--mtp-draft-vocab", str(data / "draft_vocab.bin")], "draft_vocab": "en"}
        self.assertTrue(setup.point_draft_vocab(cfg))
        self.assertEqual(cfg["args"][-1], str(data / "draft_vocab_en.bin"))
        self.assertFalse(setup.point_draft_vocab(cfg))                    # already there
        own = {"args": ["--mtp-draft-vocab", "/home/me/mine.bin"], "draft_vocab": "en"}
        self.assertFalse(setup.point_draft_vocab(own))                    # a file of the user's own is kept
        self.assertFalse(setup.point_draft_vocab({"args": ["--mtp", "rt"]}))   # Flash-Next: refresh_draft_vocab


if __name__ == "__main__":
    unittest.main()
