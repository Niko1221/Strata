"""Tests for setup.py's choices that depend on the PC (no GPU, no network, nothing installed): the image encoder of a
ready-made engine that has no code for the card (#331), --gguf-dir's shard count (#305), the experimental
Pascal/Volta build (#295), the CPU image encoder with the AMD backend (#304).

    python -m unittest tools.test_setup_choices
"""
from __future__ import annotations

import contextlib
import io
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
import setup  # noqa: E402


def quiet(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*args, **kw), out.getvalue()


class PrebuiltVision(unittest.TestCase):
    """#331: the 0.1.30/0.1.31 zip's encoder has no sm_75 code: an RTX 20 card gets the CPU encoder, not a compile."""
    META = {"version": "0.1.31", "archs": [75, 86, 89, 120], "ptx": True, "vision_archs": [86, 89, 120]}

    def test_a_card_the_encoder_has_no_code_for_gets_the_cpu_encoder(self):
        got, out = quiet(setup.prebuilt_vision, self.META, {"arch": 75}, "gpu")
        self.assertEqual(got, "cpu")
        self.assertIn("runs on the CPU instead", out)

    def test_covered_cards_keep_the_gpu_encoder(self):
        for arch in (86, 89, 120):
            self.assertEqual(quiet(setup.prebuilt_vision, self.META, {"arch": arch}, "gpu")[0], "gpu")
        # a newer card than the newest encoder code: only with PTX in the zip
        self.assertEqual(quiet(setup.prebuilt_vision, {**self.META, "vision_archs": [86, 89]}, {"arch": 120}, "gpu")[0],
                         "gpu")
        self.assertEqual(quiet(setup.prebuilt_vision, {**self.META, "vision_archs": [86, 89], "ptx": False},
                               {"arch": 120}, "gpu")[0], "cpu")
        # 0.1.32's zip: the encoder built with 75-real too
        self.assertEqual(quiet(setup.prebuilt_vision, {**self.META, "vision_archs": [75, 86, 89, 120]},
                               {"arch": 75}, "gpu")[0], "gpu")

    def test_other_choices_are_left_alone(self):
        for v in ("cpu", "none"):
            self.assertEqual(quiet(setup.prebuilt_vision, self.META, {"arch": 75}, v), (v, ""))
        # an older BUILD.json without vision_archs: the engine's archs
        self.assertEqual(quiet(setup.prebuilt_vision, {"archs": [75, 86]}, {"arch": 75}, "gpu")[0], "gpu")


class GgufDirShards(unittest.TestCase):
    """#305: --gguf-dir reads the shard count from the -of-N part of the name instead of assuming two."""

    def shards(self, names, family="qwen", model="IQ3_XXS"):
        with tempfile.TemporaryDirectory() as d:
            for n in names:
                (Path(d) / n).write_bytes(b"")
            return [p.name for p in setup.gguf_dir_shards(Path(d), setup.FAMILIES[family], model)]

    def test_the_published_two_shards(self):
        names = ["Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
                 "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"]
        self.assertEqual(self.shards(names), names)
        self.assertEqual(self.shards(names[:1]), names)              # shard 2 missing: check_shards says so later
        swift = ["Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
                 "Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"]
        self.assertEqual(self.shards(swift + ["mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf"], "swift", "Q2_0"), swift)

    def test_four_unsloth_shards(self):
        names = ["Qwen3.8-Flash-Next-UD-Q4_K_XL-%05d-of-00004.gguf" % i for i in range(1, 5)]
        self.assertEqual(self.shards(names + ["mmproj-F16.gguf"], model="Q4_K_XL"), names)
        self.assertEqual(self.shards(names[:1], model="Q4_K_XL"), names)   # the others named from the first

    def test_another_split_of_a_setup_size(self):
        names = ["my-IQ3_XXS-%05d-of-00003.gguf" % i for i in range(1, 4)]
        other = ["my-Q2_0-%05d-of-00002.gguf" % i for i in range(1, 3)]
        self.assertEqual(self.shards(names + other), names)           # the one with the size in its name
        self.assertEqual(self.shards(other, model="Q2_0"), other)

    def test_nothing_recognisable_keeps_the_published_names(self):
        want = ["Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
                "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"]
        self.assertEqual(self.shards([]), want)
        self.assertEqual(self.shards(["a-00001-of-00002.gguf", "b-00001-of-00002.gguf"]), want)   # ambiguous


class GgufDirUnsupported(unittest.TestCase):
    """#444: --gguf-dir with GGUFs Strata cannot run (Unsloth's UD-IQ3_XXS, UD-Q2_K_XL) says so, names the files it
    runs (ISTA-DASLab's GSQ-RCO, the Coder's, Unsloth's UD-Q4_K_XL) and the --family/--model of the usable ones."""
    UD = ["Qwen3.8-Flash-Next-UD-IQ3_XXS-%05d-of-00003.gguf" % i for i in range(1, 4)] + \
         ["Qwen3.8-Flash-Next-UD-Q2_K_XL-%05d-of-00003.gguf" % i for i in range(1, 4)]

    def problem(self, names, family="qwen", model="IQ3_XXS"):
        with tempfile.TemporaryDirectory() as d:
            for n in names:
                (Path(d) / n).write_bytes(b"")
            fam = setup.FAMILIES[family]
            first = setup.gguf_dir_shards(Path(d), fam, model)[0]
            p = setup.gguf_dir_problem(Path(d), first, fam, model)
            return p and (p[0].replace(d, "<D>"), p[1])

    def test_quant_names(self):
        for name, want in (("Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf", "UD-IQ3_XXS"),
                           ("Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf", "UD-Q2_K_XL"),
                           ("model-Q4_K_M.gguf", "Q4_K_M"),
                           ("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf", None),
                           ("Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", None),
                           ("my-IQ3_XXS-00001-of-00003.gguf", None), ("mmproj-F16.gguf", None),
                           ("a-00001-of-00002.gguf", None)):
            self.assertEqual(setup.gguf_unsupported(name), want, name)

    def test_an_unsloth_ud_file_is_not_taken_for_a_gsq_rco_size(self):
        msg, hint = self.problem(self.UD)                   # IQ3_XXS: UD-IQ3_XXS has the size in its name
        self.assertEqual(msg, "Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf is UD-IQ3_XXS, a GGUF Strata "
                              "cannot run")
        self.assertIn("ISTA-DASLab's GSQ-RCO files", hint)
        self.assertIn("Unsloth's UD-Q4_K_XL only", hint)

    def test_a_folder_without_the_choice_names_what_is_there(self):
        gsq = ["Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-%05d-of-00002.gguf" % i for i in (1, 2)]
        msg, hint = self.problem(self.UD + gsq, "unsloth", "UD-Q4_K_XL")
        self.assertEqual(msg, "<D> has no Qwen3.8-Flash-Next (Unsloth) UD-Q4_K_XL file")
        self.assertIn("Not usable here: Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf, "
                      "Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf", hint)
        self.assertIn("Usable here: --family coder --model IQ1_M", hint)

    def test_usable_folders_are_left_alone(self):
        good = ["Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-%05d-of-00002.gguf" % i for i in (1, 2)]
        self.assertIsNone(self.problem(good + self.UD))
        self.assertIsNone(self.problem(["my-IQ3_XXS-%05d-of-00003.gguf" % i for i in (1, 2, 3)]))
        self.assertIsNone(self.problem([]))                 # nothing there: check_shards' "missing", as before

    def test_the_size_of_another_family(self):
        from test_setup_golden import PROFILES, install
        ram, found = PROFILES["64GB-1x32GB"]
        with tempfile.TemporaryDirectory() as d:
            code, out, _, _ = install(ram, found, ["--gguf-dir", d, "--family", "unsloth", "--model", "IQ3_XXS"])
        self.assertEqual(code, 1)
        self.assertIn("has no IQ3_XXS model file", out)
        self.assertIn("choose one of: UD-Q4_K_XL (or IQ3_XXS: --family qwen --model IQ3_XXS, --family swift --model "
                      "IQ3_XXS)", out)
        self.assertIn("Strata runs ISTA-DASLab's GSQ-RCO files", out)


class ExperimentalSm60(unittest.TestCase):
    """#295: Pascal (6.x) and Volta (7.0) only with STRATA_EXPERIMENTAL_SM60=1, built with -DSTRATA_EXPERIMENTAL_SM60=ON
    and a CUDA 12.x toolkit; nothing changes without the variable."""

    def card(self, arch):
        return {"arch": arch, "vram_gb": 11.0, "index": 0, "name": "card"}

    def test_the_gate(self):
        with mock.patch.dict(os.environ, {"STRATA_EXPERIMENTAL_SM60": ""}):
            for arch in ("60", "61", "70"):
                p = setup.gpu_problem(self.card(arch))
                self.assertIn("not supported", p)
                self.assertIn("STRATA_EXPERIMENTAL_SM60=1", p)
            self.assertNotIn("STRATA_EXPERIMENTAL_SM60", setup.gpu_problem(self.card("52")))
            self.assertIsNone(setup.gpu_problem(self.card("75")))
        with mock.patch.dict(os.environ, {"STRATA_EXPERIMENTAL_SM60": "1"}):
            for arch in ("60", "61", "70", "75", "120"):
                self.assertIsNone(setup.gpu_problem(self.card(arch)), arch)
            for arch in ("52", "72"):
                self.assertIsNotNone(setup.gpu_problem(self.card(arch)), arch)

    def test_the_build_flag(self):
        self.assertEqual(setup.engine_defs([61]), ["-DSTRATA_EXPERIMENTAL_SM60=ON"])
        self.assertEqual(setup.engine_defs([70, 86]), ["-DSTRATA_EXPERIMENTAL_SM60=ON"])
        self.assertEqual(setup.engine_defs([75, 86, 120]), [])

    def test_find_nvcc_below(self):
        with tempfile.TemporaryDirectory() as d:
            new, old = Path(d) / "13" / "nvcc", Path(d) / "12" / "bin" / "nvcc"
            for p in (new, old):
                p.parent.mkdir(parents=True)
                p.write_bytes(b"")
            versions = {str(new): "Cuda compilation tools, release 13.0, V13.0.88",
                        str(old): "Cuda compilation tools, release 12.9, V12.9.86"}
            with mock.patch.object(setup, "WIN", False), mock.patch.object(setup.shutil, "which", lambda n: str(new)), \
                    mock.patch.dict(os.environ, {"CUDA_PATH": str(Path(d) / "12")}), \
                    mock.patch.object(setup, "out", lambda cmd: versions.get(cmd[0], "")):  # #414
                self.assertEqual(setup.find_nvcc(), (str(new), (13, 0)))
                self.assertEqual(setup.find_nvcc(below=(13, 0)), (str(old), (12, 9)))

    def tools(self, archs, nvcc):
        seen = []

        def find(below=None):
            seen.append(below)
            return nvcc(below)

        with mock.patch.object(setup, "find_nvcc", find), mock.patch.object(setup, "find_vcvars", lambda: "vcvars"), \
                mock.patch.object(setup.shutil, "which", lambda n: "/usr/bin/" + n):
            got, _ = quiet(setup.install_build_tools, {"arch": str(archs[0]), "archs": archs}, True)
        return got, seen

    def test_pascal_takes_cuda_12(self):
        got, seen = self.tools([61], lambda below: ("nvcc12", (12, 9)) if below == (13, 0) else ("nvcc13", (13, 0)))
        self.assertEqual(got[0], "nvcc12")
        self.assertEqual(seen, [(13, 0)])
        got, seen = self.tools([86, 120], lambda below: ("nvcc13", (13, 0)))
        self.assertEqual((got[0], seen), ("nvcc13", [None]))           # the default: unchanged

    def test_pascal_without_cuda_12_stops(self):
        for archs in ([61], [70, 120]):
            with self.subTest(archs=archs), self.assertRaises(SystemExit):
                self.tools(archs, lambda below: (None, None) if below else ("nvcc13", (13, 0)))


class HipVision(unittest.TestCase):
    """#304: --vision cpu with --backend hip builds the CPU image encoder beside the HIP engine."""

    @mock.patch.object(setup, "WIN", False)            # Linux: compiled here (Windows has no AMD image encoder yet)
    def test_the_choice(self):
        self.assertEqual(quiet(setup.hip_vision, "cpu"), ("cpu", ""))
        for asked in (None, "no", "none"):
            self.assertEqual(quiet(setup.hip_vision, asked), ("none", ""))
        for asked in ("yes", "gpu"):
            got, out = quiet(setup.hip_vision, asked)
            self.assertEqual(got, "none")
            self.assertIn("--vision cpu", out)

    def build(self, meta, vision, vexe=False):
        """build_engine_hip with an engine that is already built: only the encoder can be missing."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            eng = root / "engine"
            eng.mkdir()
            (eng / setup.EXE).write_bytes(b"engine")
            if vexe:
                (eng / setup.VEXE).write_bytes(b"vision")
            src, vsrc = "S", "V"
            (eng / "BUILD.json").write_text(json.dumps({"backend": "hip", "archs": ["gfx1201"], "src": src, **meta}))
            built = []

            def cmake_build(src_dir, bdir, target, defs, vcvars, bat):
                built.append((target, defs))
                (bdir / "bin").mkdir(parents=True)
                (bdir / "bin" / setup.VEXE).write_bytes(b"vision")

            with mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "cmake_build", cmake_build), \
                    mock.patch.object(setup, "source_hash", lambda paths: vsrc if paths == setup.VISION_SOURCES else src):
                quiet(setup.build_engine_hip, {"arch": "gfx1201"}, "llama", vision)
            return built, json.loads((eng / "BUILD.json").read_text()), (eng / setup.VEXE).exists()

    def test_the_cpu_encoder_is_built_once(self):
        built, meta, have = self.build({}, "cpu")
        self.assertEqual([t for t, _ in built], ["strata-vision"])
        self.assertIn("-DSTRATA_VISION_CUDA=OFF", built[0][1])
        self.assertEqual((meta["vision"], meta["vision_src"], have), ("cpu", "V", True))
        built, meta, _ = self.build({"vision": "cpu", "vision_src": "V"}, "cpu", vexe=True)
        self.assertEqual(built, [])                                    # built and unchanged: nothing to do
        built, meta, _ = self.build({"vision": "cpu", "vision_src": "old"}, "cpu", vexe=True)
        self.assertEqual([t for t, _ in built], ["strata-vision"])     # its source changed: again

    def test_without_images_nothing_changes(self):
        built, meta, have = self.build({}, "none")
        self.assertEqual((built, have), ([], False))
        self.assertNotIn("vision_src", meta)


class VramReserve(unittest.TestCase):
    """#493: --vram-reserve-mib N - VRAM the engine leaves free for other programs - goes into the config only when
    given (the default config is tools/test_setup_golden.py's, unchanged), at setup and on a start."""

    def setUp(self):
        from test_setup_golden import PROFILES
        self.ram, self.found = PROFILES["64GB-1x32GB"]

    def install(self, *extra):
        from test_setup_golden import install
        return install(self.ram, self.found, ["--family", "qwen", "--model", "Q2_0", "--no-start", *extra])

    def test_given_at_setup(self):
        code, out, cfg, _ = self.install("--vram-reserve-mib", "2048")
        self.assertEqual(code, 0, out)
        a = cfg["args"]
        self.assertEqual(a.count("--vram-reserve-mib"), 1)
        self.assertEqual(a[a.index("--vram-reserve-mib") + 1], "2048")
        self.assertIn("VRAM kept free for other programs: 2048 MiB", out)
        code, out, cfg, _ = self.install()
        self.assertNotIn("--vram-reserve-mib", cfg["args"])          # not given: the engine's default, as before

    def test_with_images_one_value(self):
        code, out, cfg, _ = self.install("--vision", "cpu", "--vram-reserve-mib", "1500")
        self.assertEqual(code, 0, out)
        a = cfg["args"]
        self.assertEqual(a.count("--vram-reserve-mib"), 1)
        self.assertEqual(a[a.index("--vram-reserve-mib") + 1], "1500")

    def test_a_negative_value_is_refused(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            code, out, cfg, _ = self.install("--vram-reserve-mib", "-1")
        self.assertEqual(code, 2)
        self.assertIn("--vram-reserve-mib takes a number of MiB", err.getvalue())

    def test_kept_from_an_earlier_install(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-q2_0.json"
            p.write_text(json.dumps({"args": ["--max-context", "8192", "--vram-reserve-mib", "2048"]}))
            self.assertEqual(setup.choices_from_config(p)["vram_reserve_mib"], 2048)
            p.write_text(json.dumps({"args": ["--vision", "--vram-reserve-mib", "700"], "vision": {"gpu": True}}))
            self.assertIsNone(setup.choices_from_config(p)["vram_reserve_mib"])   # the images' own reserve
            p.write_text(json.dumps({"args": ["--max-context", "8192"]}))
            self.assertIsNone(setup.choices_from_config(p)["vram_reserve_mib"])

    def test_given_on_a_start(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "strata.exe"
            exe.write_bytes(b"")
            p = Path(d) / "strata-q2_0.json"
            p.write_text(json.dumps({"exe": str(exe), "args": ["--kv", "int8"], "gpu": 0, "gpus_asked": True}))
            call = mock.Mock(return_value=0)
            with mock.patch.object(setup, "gpus", lambda: self.found), \
                    mock.patch.object(setup, "is_wsl", lambda: False), \
                    mock.patch.object(setup, "ensure_engine_for", lambda cards, path, cfg, yes: cfg), \
                    mock.patch.object(setup.subprocess, "call", call):
                for n in ("2048", "1024"):
                    _, out = quiet(setup.start, p, None, None, False, True, None, {"vram_reserve_mib": int(n)})
                    self.assertEqual(json.loads(p.read_text())["args"], ["--kv", "int8", "--vram-reserve-mib", n])
        self.assertIn("1024 MiB of VRAM kept free", out)
        self.assertTrue(call.called)


class SmallCardTip(unittest.TestCase):
    """#496: a card under 8 GB (a 6 GB laptop RTX 3060) gets a tip for when the start has no room for the expert cache;
    the engine chooses its own reserve there, so the config is the one any card gets.  An 8 GB card: no tip."""

    def install(self, vram, *extra):
        from test_setup_golden import card, install
        return install(63.7, [card(0, "NVIDIA GeForce RTX 3060 Laptop GPU", vram, "86")],
                       ["--family", "qwen", "--model", "Q2_0", "--no-start", *extra])

    def test_a_6gb_card(self):
        code, out, cfg, _ = self.install(6.0)
        self.assertEqual(code, 0, out)
        self.assertNotIn("--vram-reserve-mib", cfg["args"])          # the engine decides (no fixed reserve)
        self.assertIn("no VRAM is left for the expert cache", out)
        self.assertIn("--draft-vocab en", out)
        self.assertIn("--mtp", cfg["args"])                          # the draft layer stays: the server needs it

    def test_an_8gb_card_has_no_tip(self):
        for vram in (8188 / 1024, 8.0):                               # nvidia-smi lists an 8 GB card as 8188 MiB
            code, out, cfg, _ = self.install(vram)
            self.assertEqual(code, 0, out)
            self.assertNotIn("no VRAM is left for the expert cache", out)
            self.assertNotIn("--vram-reserve-mib", cfg["args"])

    def test_a_given_reserve_is_kept(self):
        code, out, cfg, _ = self.install(6.0, "--vram-reserve-mib", "500")
        self.assertEqual(code, 0, out)
        a = cfg["args"]
        self.assertEqual(a[a.index("--vram-reserve-mib") + 1], "500")

    def test_the_tip(self):
        self.assertIn("an 8K context", " ".join(setup.small_card_note(32768, None)))
        self.assertIn("--draft-vocab en", " ".join(setup.small_card_note(32768, "cjk")))
        tip = " ".join(setup.small_card_note(8192, "en"))
        self.assertNotIn("--draft-vocab", tip)
        self.assertIn("close other programs", tip)


class DesktopReserveTip(unittest.TestCase):
    """#560 #516: an AMD card on a Linux desktop gets a recommended reserve (3072 MiB) - a tip, the config is not
    changed."""

    def test_desktop_detection(self):
        with mock.patch.object(setup.sys, "platform", "linux"):
            self.assertTrue(setup.linux_desktop({"WAYLAND_DISPLAY": "wayland-0"}))
            self.assertTrue(setup.linux_desktop({"DISPLAY": ":0"}))
            self.assertFalse(setup.linux_desktop({}))                 # a headless box / ssh
        with mock.patch.object(setup.sys, "platform", "win32"):
            self.assertFalse(setup.linux_desktop({"DISPLAY": ":0"}))  # Windows counts the desktop itself (#497)

    def test_the_tip(self):
        tip = " ".join(setup.desktop_reserve_note())
        self.assertIn("--vram-reserve-mib 3072", tip)
        self.assertIn("desktop", tip)


class VisionDevice(unittest.TestCase):
    """--vision-device: the image encoder's device as its own role next to the engine's cards (its card form is
    #408's "cuda_device" in the config's vision section, the CPU side #304's encoder)."""

    CARDS = [{"index": 0, "name": "A", "vram_gb": 11.0, "arch": "86"},
             {"index": 1, "name": "B", "vram_gb": 11.0, "arch": "86"},
             {"index": 2, "name": "spare", "vram_gb": 16.0, "arch": "89"}]   # GPU 2: the spare one
    CHOSEN = [{"index": 0}, {"index": 1}]                                     # the engine's layer-split cards

    def role(self, vision, text, hip=False):
        with mock.patch.object(setup, "gpus", lambda: self.CARDS), \
                mock.patch.object(setup, "amd_gpus", lambda: self.CARDS):
            return quiet(setup.vision_device_role, vision, text, self.CHOSEN, hip)

    def test_the_flag_absent_changes_nothing(self):
        self.assertEqual(quiet(setup.vision_device_role, "gpu", "", self.CHOSEN), (("gpu", None), ""))
        self.assertEqual(quiet(setup.vision_device_role, "none", "", self.CHOSEN), (("none", None), ""))

    def test_a_card_number_pins_the_spare(self):
        for text in ("2", "cuda:2", " CUDA:2 "):
            (vision, card), out = self.role("gpu", text)
            self.assertEqual((vision, card), ("gpu", 2))
            self.assertIn("its own GPU: spare (GPU 2)", out)               # the most VRAM is not the rule here
        (vision, card), _ = self.role("gpu", "cuda:2", hip=True)          # AMD: the same numbers as setup lists them
        self.assertEqual((vision, card), ("gpu", 2))

    def test_the_engine_card_warns(self):
        (_, card), out = self.role("gpu", "1")
        self.assertEqual(card, 1)
        self.assertIn("shares GPU 1 with the engine", out)

    def test_a_missing_card_stops(self):
        with self.assertRaises(SystemExit):
            self.role("gpu", "3")

    def test_cpu_reads_on_the_cpu(self):
        (vision, card), out = self.role("gpu", "cpu")
        self.assertEqual((vision, card), ("cpu", None))
        self.assertIn("read on the CPU", out)
        self.assertEqual(self.role("cpu", "cpu")[0], ("cpu", None))       # already the CPU encoder: nothing changes

    def test_cpu_needs_the_pictures_on(self):
        with self.assertRaises(SystemExit):
            self.role("none", "cpu")

    def test_auto_needs_the_pictures_on(self):
        with self.assertRaises(SystemExit):
            self.role("none", "auto")

    def test_a_card_with_the_cpu_encoder_stops(self):
        with self.assertRaises(SystemExit):
            self.role("cpu", "1")

    def test_garbage_stops(self):
        with self.assertRaises(SystemExit):
            self.role("gpu", "two")

    def test_auto_takes_the_best_spare(self):
        with mock.patch.object(setup, "gpus", lambda: self.CARDS), \
                mock.patch.object(setup, "amd_gpus", lambda: self.CARDS):
            (_, card), out = quiet(setup.vision_device_auto, "gpu", self.CHOSEN)
            self.assertEqual(card, 2)
            self.assertIn("its own GPU: spare", out)
            (_, card), out = quiet(setup.vision_device_auto, "gpu", [{"index": 0}, {"index": 2}])
            self.assertEqual(card, 1)                                              # the card left over is the spare
            self.assertIn("its own GPU:", out)
            (_, card), out = quiet(setup.vision_device_auto, "gpu",
                                   [{"index": 0}, {"index": 1}, {"index": 2}])     # no spare left
            self.assertEqual((card, "own GPU" in out), (None, False))
            self.assertIn("no spare GPU", out)
            self.assertEqual(quiet(setup.vision_device_auto, "cpu", self.CHOSEN)[0], ("cpu", None))
            self.assertEqual(quiet(setup.vision_device_auto, "none", self.CHOSEN)[0], ("none", None))


VULKANINFO_SUMMARY = """
Devices:
========

        GPU0:\t
                deviceName        = NVIDIA GeForce GTX 1080 Ti
                deviceType        = DISCRETE_GPU

        GPU1:\t
                deviceName        = Intel(R) Iris(R) Xe Graphics
                deviceType        = INTEGRATED_GPU
"""

SYCL_LS = """
[0] level_zero:gpu:0.0 Intel(R) Arc(TM) A770 Graphics (0x56a0)
[1] level_zero:gpu:1.0 Intel(R) UHD Graphics (0x46d1)
[2] cuda:0.0 CUDA 12.2 (NVIDIA GeForce GTX 1080 Ti)
[3] opencl:cpu:0.0 Intel(R) Core(TM) i7
"""


class VisionBackend(unittest.TestCase):
    """--vision-backend vulkan|sycl (the non-CUDA image encoders): the encoder is built for another ggml GPU backend
    and its device number is that backend's own list's (vulkaninfo / sycl-ls), not nvidia-smi's."""

    VK = [{"index": 0, "name": "NVIDIA GeForce GTX 1080 Ti", "integrated": False},
          {"index": 1, "name": "Intel(R) Iris(R) Xe Graphics", "integrated": True}]
    CHOSEN = [{"index": 0}, {"index": 1}]

    def backend_role(self, vision, text, backend, devices):
        with mock.patch.object(setup, "vk_devices", lambda: devices), \
                mock.patch.object(setup, "sycl_devices", lambda: devices):
            return quiet(setup.vision_device_role, vision, text, self.CHOSEN, False, backend)

    def test_a_backend_number_is_its_own_lists_number(self):
        (vision, card), out = self.backend_role("gpu", "1", "vulkan", self.VK)
        self.assertEqual((vision, card), ("gpu", 1))
        self.assertIn("its own GPU: Intel(R) Iris(R) Xe Graphics (GPU 1)", out)     # the CUDA list's "1" is "A"
        (_, card), _ = self.backend_role("gpu", "cuda:1", "vulkan", self.VK)
        self.assertEqual(card, 1)                                       # the prefix does not turn it into CUDA's
        (_, card), _ = self.backend_role("gpu", " 0 ", "sycl", self.VK)
        self.assertEqual(card, 0)

    def test_a_backend_number_outside_its_list_stops(self):
        with self.assertRaises(SystemExit):
            self.backend_role("gpu", "2", "vulkan", self.VK)            # the Vulkan list ends at 1
        with self.assertRaises(SystemExit):
            self.backend_role("gpu", "1", "sycl", [{"index": 0, "name": "Arc"}])

    def test_auto_with_a_backend_stops(self):
        with mock.patch.object(setup, "vk_devices", lambda: self.VK):
            with self.assertRaises(SystemExit):
                quiet(setup.vision_device_auto, "gpu", self.CHOSEN, False, "vulkan")

    def test_vk_devices_reads_vulkaninfo(self):
        with mock.patch.object(setup, "out", lambda cmd: VULKANINFO_SUMMARY):
            found = setup.vk_devices()
        self.assertEqual(found, [{"index": 0, "driver": "vulkan", "name": "NVIDIA GeForce GTX 1080 Ti",
                                  "integrated": False},
                                 {"index": 1, "driver": "vulkan", "name": "Intel(R) Iris(R) Xe Graphics",
                                  "integrated": True}])
        with mock.patch.object(setup, "out", lambda cmd: ""):
            with self.assertRaises(SystemExit):
                setup.vk_devices()

    def test_sycl_devices_reads_sycl_ls_and_keeps_only_level_zero(self):
        with mock.patch.object(setup, "out", lambda cmd: SYCL_LS):
            found = setup.sycl_devices()
        self.assertEqual(found, [{"index": 0, "name": "Intel(R) Arc(TM) A770 Graphics (0x56a0)", "driver": "sycl"},
                                 {"index": 1, "name": "Intel(R) UHD Graphics (0x46d1)", "driver": "sycl"}])
        with mock.patch.object(setup, "out", lambda cmd: "sycl-ls: no device found"):
            with self.assertRaises(SystemExit):
                setup.sycl_devices()

    def build(self, meta, vision="gpu", backend="", vexe=False):
        """build_engine in a fresh dir, with an engine already there (only the encoder can be missing)."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            eng = root / "engine"
            eng.mkdir()
            (eng / setup.EXE).write_bytes(b"engine")
            if vexe:
                (eng / setup.VEXE).write_bytes(b"vision")
            src, vsrc = "S", "V"
            (eng / "BUILD.json").write_text(json.dumps({"source": "local", "archs": [86], "src": src, **meta}))
            built = []

            def cmake_build(src_dir, bdir, target, defs, vcvars, bat):
                built.append((target, defs))
                (bdir / "bin").mkdir(parents=True)
                (bdir / "bin" / setup.VEXE).write_bytes(b"vision")

            with mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "cmake_build", cmake_build), \
                    mock.patch.object(setup, "install_build_tools", lambda gpu_, yes: (str(root / "cuda" / "nvcc"), None)), \
                    mock.patch.object(setup, "source_hash", lambda paths: vsrc if paths == setup.VISION_SOURCES else src), \
                    mock.patch.object(setup, "source_version", lambda: "0.0.0"):
                quiet(setup.build_engine, {"arch": "86", "vram_gb": 24.0}, vision, False, "llama", backend)
            return built, json.loads((eng / "BUILD.json").read_text())

    def test_the_vulkan_encoder_def(self):
        built, meta = self.build({}, backend="vulkan")
        self.assertEqual([t for t, _ in built], ["strata-vision"])
        self.assertIn("-DSTRATA_VISION_VULKAN=ON", built[0][1])
        self.assertEqual((meta["vision_backend"], meta["vision"]), ("vulkan", "gpu"))
        for no in ("-DSTRATA_VISION_CUDA=ON", "-DSTRATA_VISION_CUDA=OFF", "-DCMAKE_CUDA_ARCHITECTURES=86"):
            self.assertNotIn(no, built[0][1])                  # a Vulkan build takes no CUDA arch either

    def test_a_backend_change_rebuilds_the_encoder(self):
        built, meta = self.build({"vision": "gpu", "vision_src": "V", "vision_backend": "cuda"}, backend="vulkan")
        self.assertEqual([t for t, _ in built], ["strata-vision"])     # the CUDA encoder in place is not Vulkan's
        self.assertEqual(meta["vision_backend"], "vulkan")

    def test_the_same_backend_keeps_the_encoder(self):
        built, meta = self.build({"vision": "gpu", "vision_src": "V", "vision_backend": "vulkan"},
                                 backend="vulkan", vexe=True)
        self.assertEqual(built, [])                                    # the Vulkan encoder in place is this build's
        self.assertEqual(meta["vision_backend"], "vulkan")

    def test_dropping_the_backend_rebuilds_the_cuda_encoder(self):
        built, meta = self.build({"vision": "gpu", "vision_src": "V", "vision_backend": "vulkan"})
        self.assertIn("-DSTRATA_VISION_CUDA=ON", built[0][1])          # back to CUDA, with its arch line back too
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=86", built[0][1])
        self.assertIsNone(meta["vision_backend"])

    def test_the_flag_without_images_builds_nothing_backendish(self):
        built, meta = self.build({}, vision="cpu", backend="vulkan")
        self.assertEqual([t for t, _ in built], ["strata-vision"])     # the CPU encoder, as ever
        self.assertIn("-DSTRATA_VISION_CUDA=OFF", built[0][1])         # no backend line in its defs
        self.assertIsNone(meta.get("vision_backend"))


if __name__ == "__main__":
    unittest.main()
