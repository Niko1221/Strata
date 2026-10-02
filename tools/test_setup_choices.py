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


if __name__ == "__main__":
    unittest.main()
