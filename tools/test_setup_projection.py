"""Tests for setup.py's two experimental projections (data/experimental-speed-projection): the existing one
(--experimental-speed-projection, layers 4-44: qwen and coder) and the separate layers 8-33 one
(--experimental-projection-l8-33: qwen, unsloth, nvfp4, never NVIDIA's checkpoint).  Each family gets the right vector
and layer range, both are off by default, the two are never on at once (command line, questions, config), and a config reads back.

    python -m unittest tools.test_setup_projection
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import test_setup_golden as G  # noqa: E402

PC = "128GB-1x24GB"
UP, L8 = "--experimental-speed-projection", "--experimental-projection-l8-33"
CASES = [("qwen", "IQ3_S"), ("coder", "IQ1_M"), ("unsloth", "UD-Q4_K_XL"), ("nvfp4", "NVFP4"), ("nvfp4", "NVFP4-NVIDIA"),
         ("swift", "IQ3_S")]


def install(family, model, extra=(), answers=None):
    ram, found = G.PROFILES[PC]
    return G.install(ram, found, ["--family", family, "--model", model, "--no-start", *extra], answers=answers)


def vector(cfg):
    """(file name, first and last layer) of the control vector in a written config, or None."""
    a = cfg["args"]
    if "--control-vector-scaled" not in a:
        return None
    r = a.index("--control-vector-layer-range")
    return (a[a.index("--control-vector-scaled") + 1].rsplit(":", 1)[0].replace("\\", "/").rsplit("/", 1)[-1],
            int(a[r + 1]), int(a[r + 2]))


UP_V = (setup.ESP_VECTOR.name, 4, 44)
L8_V = (setup.ESP_L8_VECTOR.name, 8, 33)


class Files(unittest.TestCase):
    def test_the_vectors_ship_and_upstreams_is_unchanged(self):
        self.assertTrue(setup.ESP_VECTOR.is_file() and setup.ESP_L8_VECTOR.is_file())
        self.assertEqual(setup.ESP_VECTOR.name, "Qwen3.8-Flash-Next-experimental-speed-projection.gguf")
        self.assertEqual(setup.ESP_L8_LAYERS, (8, 33))


class Install(unittest.TestCase):
    def test_off_by_default(self):
        for fam, model in CASES:
            with self.subTest(fam=fam, model=model):
                code, out, cfg, _ = install(fam, model)
                self.assertEqual(code, 0, out[-1500:])
                self.assertIsNone(vector(cfg))

    def test_existing_option_is_as_before(self):
        for fam in ("qwen", "coder"):
            with self.subTest(fam):
                code, out, cfg, _ = install(fam, "IQ1_M" if fam == "coder" else "IQ3_S", [UP, "on"])
                self.assertEqual(code, 0, out[-1500:])
                self.assertEqual(vector(cfg), UP_V)
        for fam, model in (("unsloth", "UD-Q4_K_XL"), ("nvfp4", "NVFP4"), ("swift", "IQ3_S")):
            with self.subTest(fam):
                code, out, cfg, _ = install(fam, model, [UP, "on"])
                self.assertEqual(code, 0, out[-1500:])
                self.assertIsNone(vector(cfg))             # still left off there, with the warning
                self.assertIn("left off", out)

    def test_ours_for_the_original_model_in_every_quantization(self):
        for fam, model in (("qwen", "IQ3_S"), ("unsloth", "UD-Q4_K_XL"), ("nvfp4", "NVFP4")):
            with self.subTest(fam=fam, model=model):
                code, out, cfg, _ = install(fam, model, [L8, "on"])
                self.assertEqual(code, 0, out[-1500:])
                self.assertEqual(vector(cfg), L8_V)
                a = cfg["args"]
                self.assertEqual((a[a.index("--cvec-mode") + 1], a[a.index("--cvec-dir") + 1]), ("project", "per-layer"))
                self.assertTrue(a[a.index("--control-vector-scaled") + 1].endswith(":1.0"))
                self.assertEqual(a.count("--control-vector-scaled"), 1)

    def test_not_for_nvidias_checkpoint(self):
        # the NVIDIA Open Model License does not allow reducing that model's safeguards: never loaded, said why
        code, out, cfg, _ = install("nvfp4", "NVFP4-NVIDIA", [L8, "on"])
        self.assertEqual(code, 0, out[-1500:])
        self.assertIsNone(vector(cfg))
        self.assertIn("not offered for NVFP4-NVIDIA", out)
        self.assertIn("NVIDIA Open Model License", out)

    def test_not_for_the_coder_or_swift(self):
        for fam in ("coder", "swift"):
            with self.subTest(fam):
                code, out, cfg, _ = install(fam, "IQ1_M" if fam == "coder" else "IQ3_S", [L8, "on"])
                self.assertEqual(code, 0, out[-1500:])
                self.assertIsNone(vector(cfg))
                self.assertIn("left off", out)

    def test_both_on_on_the_command_line_stops(self):
        for fam, model in (("qwen", "IQ3_S"), ("unsloth", "UD-Q4_K_XL"), ("nvfp4", "NVFP4")):
            with self.subTest(fam):
                code, out, cfg, _ = install(fam, model, [UP, "on", L8, "on"])
                self.assertNotEqual(code, 0)
                self.assertIsNone(cfg)
                self.assertIn(UP, out)
                self.assertIn(L8, out)
                self.assertIn("cannot both be on", out)

    def test_a_bad_answer(self):
        code, out, _, _ = install("qwen", "IQ3_S", [L8, "maybe"])
        self.assertNotEqual(code, 0)


class Questions(unittest.TestCase):
    EXISTING, OURS = "Turn on the experimental speed projection?", "Turn on the experimental layers 8-33 projection?"

    def run_with(self, fam, model, answers):
        code, out, cfg, asked = install(fam, model, answers=answers)
        self.assertEqual(code, 0, out[-1500:])
        return cfg, [q for q in asked if "projection" in q]

    def test_qwen_asks_both_and_never_ends_with_both(self):
        for a1, a2 in (("n", "n"), ("y", "n"), ("n", "y"), ("y", "y")):
            with self.subTest(a1=a1, a2=a2):
                cfg, qs = self.run_with("qwen", "IQ3_S", {self.EXISTING: a1, self.OURS: a2})
                want = UP_V if a1 == "y" else L8_V if a2 == "y" else None
                self.assertEqual(vector(cfg), want)
                self.assertEqual(len(qs), 1 if a1 == "y" else 2)   # ours is asked only when the existing one is off

    def test_default_is_off(self):
        cfg, qs = self.run_with("qwen", "IQ3_S", "")
        self.assertIsNone(vector(cfg))
        self.assertEqual(len(qs), 2)

    def test_unsloth_and_nvfp4_ask_only_ours(self):
        for fam, model in (("unsloth", "UD-Q4_K_XL"), ("nvfp4", "NVFP4")):
            with self.subTest(fam=fam, model=model):
                cfg, qs = self.run_with(fam, model, {self.OURS: "y"})
                self.assertEqual(vector(cfg), L8_V)
                self.assertEqual(len(qs), 1)
                self.assertIn("layers 8-33", qs[0])

    def test_nvidias_checkpoint_is_not_asked(self):
        cfg, qs = self.run_with("nvfp4", "NVFP4-NVIDIA", {self.OURS: "y"})
        self.assertIsNone(vector(cfg))
        self.assertEqual(qs, [])

    def test_coder_asks_only_the_existing_one(self):
        cfg, qs = self.run_with("coder", "IQ1_M", {self.EXISTING: "y"})
        self.assertEqual(vector(cfg), UP_V)
        self.assertEqual(len(qs), 1)

    def test_swift_asks_neither(self):
        cfg, qs = self.run_with("swift", "IQ3_S", "")
        self.assertEqual(qs, [])

    def test_ours_given_on_the_command_line_skips_the_existing_question(self):
        code, out, cfg, asked = install("qwen", "IQ3_S", [L8, "on"], answers="")
        self.assertEqual(code, 0, out[-1500:])
        self.assertEqual(vector(cfg), L8_V)
        self.assertEqual([q for q in asked if "projection" in q], [])


class ReadBack(unittest.TestCase):
    def read(self, tag, args):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / f"strata-{tag}.json"
            f.write_text(json.dumps({"args": args}), encoding="utf-8")
            return setup.choices_from_config(f)

    def test_each_vector_is_recognised_by_its_file(self):
        up = ["--control-vector-scaled", f"{setup.ESP_VECTOR}:1.0", "--control-vector-layer-range", "4", "44"]
        l8 = setup.esp_l8_flags(setup.ESP_L8_VECTOR)
        for tag, fam in (("iq3_s", "qwen"), ("coder-iq3_s", "coder"), ("unsloth-ud-q4_k_xl", "unsloth"),
                         ("nvfp4-nvfp4", "nvfp4")):
            with self.subTest(tag=tag):
                ch = self.read(tag, up)
                self.assertEqual((ch["family"], ch["esp"], ch["esp_l8"]), (fam, "on", "off"))
                ch = self.read(tag, l8)
                self.assertEqual((ch["family"], ch["esp"], ch["esp_l8"]), (fam, "off", "on"))
                ch = self.read(tag, [])
                self.assertEqual((ch["esp"], ch["esp_l8"]), ("off", "off"))
        self.assertTrue(self.read("iq3_s", ["--control-vector-scaled", "x/mine.gguf:1.0"])["esp"].endswith("mine.gguf"))


class Server(unittest.TestCase):
    def test_a_config_with_both_is_refused(self):
        from serve.server import refuse_two_projections
        d = ROOT / "data" / "experimental-speed-projection"
        one = {"args": ["--control-vector-scaled", f"{d}/{setup.ESP_VECTOR.name}:1.0"]}
        refuse_two_projections(one)
        refuse_two_projections({"args": setup.esp_l8_flags(setup.ESP_L8_VECTOR)})
        refuse_two_projections({})
        both = {"args": one["args"] + ["--control-vector-layer-range", "4", "44"] + setup.esp_l8_flags(setup.ESP_L8_VECTOR)}
        with self.assertRaises(SystemExit) as cm:
            refuse_two_projections(both)
        self.assertIn(setup.ESP_VECTOR.name, str(cm.exception))
        self.assertIn(setup.ESP_L8_VECTOR.name, str(cm.exception))
        win = {"args": ["--control-vector-scaled", "C:\\S\\data\\x\\" + setup.ESP_VECTOR.name + ":1.0",
                        "--control-vector-scaled", "C:\\S\\data\\x\\" + setup.ESP_L8_VECTOR.name + ":1.0"]}
        with self.assertRaises(SystemExit):
            refuse_two_projections(win)

    def test_the_per_request_switch_is_one_key_for_whichever_vector_is_loaded(self):
        from serve.server import StrataEngine
        self.assertEqual(StrataEngine.projection_key({"experimental_speed_projection": False}), " cvec=0")
        self.assertEqual(StrataEngine.projection_key({}), "")


if __name__ == "__main__":
    unittest.main()
