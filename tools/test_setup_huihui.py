"""Setup's Huihui abliterated Qwen3.6 family: pins, RAM recommendations, selection, output names and local reuse.
The qwen36 harness mocks the GPU, downloads, builds and pack tools; all writes stay in a temporary directory.
Filesystem assertions use pathlib / os.path.join so the same tests work on Windows and Linux.

    python -m unittest tools.test_setup_huihui -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_setup_qwen36 as q36  # noqa: E402

setup = q36.setup
REPO = "huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF"
REV = "acd5abe0c5ebd064f3863ca7d4bba2d924ec6544"
BASE = f"{setup.hf_endpoint()}/{REPO}/resolve/{REV}/"
STEM = "Huihui-Qwen3.6-35B-A3B-abliterated-ggml-model-"
# The Hub's LFS pointers from the family brief, independent of setup's table.
PINS = {
    "Q2_K": (13246129376, "85f446fff19406aa81bd8333433fa11921038ba1b80523c2432d82623b73fb73"),
    "Q3_K": (17165606112, "a2b82acc9d3e0f1de711cac93deaebc6f07ee999ab82a400ef0dd8d8ca97338f"),
    "Q4_K": (21712409824, "63b75afb4b68fc61059c78115b580037074de88146e0762f4a496081e410cf81"),
    "Q5_K": (25346479328, "01a919f50fdffb78713423f5883709808bc0672f257f5b9627c98181723fb45b"),
    "Q6_K": (29207678176, "06aa67c944362aff75878ba732b18054485c3a1c89b7085796cfde5ad3c5041a"),
}
SIZES = ("Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K")  # biggest first, as small_size expects


def key(size):
    return f"Huihui-Qwen3.6-{size}"


def filename(size):
    return f"{STEM}{size}.gguf"


class Tables(unittest.TestCase):
    def test_family(self):
        fam = setup.FAMILIES["huihui"]
        self.assertEqual(list(setup.FAMILIES)[:4], ["qwen", "swift", "coder", "unsloth"])
        self.assertEqual(list(setup.FAMILIES)[-2:], ["qwen36", "ornith"])
        for field, value in {"tag": "huihui-", "name": "huihui-qwen3.6-35b-a3b-abliterated",
                             "architecture": "qwen35moe", "mtp": False, "own_mtp": False, "ple": False,
                             "one_gpu": True, "nvidia_only": True, "vision": False, "mmproj": None,
                             "profile": "expert-profile-qwen36.bin", "shards": 1}.items():
            self.assertEqual(fam[field], value, field)
        self.assertEqual(fam["hf"], BASE)
        self.assertEqual(fam["mmproj_hf"], BASE)
        self.assertEqual(fam["pack_args"], ["--compat-bf16"])
        self.assertIn("BF16/F32", fam["mtp_note"])
        for family in ("huihui", "qwen36", "ornith"):
            self.assertTrue(setup.small_family(family), family)
        for family in ("qwen", "swift", "coder", "unsloth", "unknown"):
            self.assertFalse(setup.small_family(family), family)

    def test_sizes_and_estimates(self):
        models = [m for m, d in setup.MODELS.items() if "huihui" in d.get("families", ())]
        self.assertEqual(models, [key(s) for s in SIZES])
        ram_lines = {"Q2_K": 24, "Q3_K": 28, "Q4_K": 30, "Q5_K": 36, "Q6_K": 40}
        for size in SIZES:
            with self.subTest(size=size):
                model = setup.MODELS[key(size)]
                self.assertEqual(model["engine"], setup.QWEN36_ENGINE)
                self.assertEqual(model["families"], ("huihui",))
                self.assertEqual(model["ram_gb"], ram_lines[size])
                self.assertFalse(model.get("budget") or model.get("experimental"))
                self.assertAlmostEqual(model["download_gb"], PINS[size][0] / 1e9, places=1)
                self.assertGreaterEqual(model["ram_gb"], model["arena_gb"] + setup.LOW_RAM_HEADROOM_GB)
                if size != "Q4_K":
                    self.assertGreaterEqual(model["arena_gb"], PINS[size][0] / 1e9)  # whole-file upper bound
        self.assertAlmostEqual(setup.MODELS[key("Q4_K")]["arena_gb"], 19505610752 / 1e9, places=1)

    def test_pins(self):
        fam = setup.FAMILIES["huihui"]
        self.assertEqual(setup.HF_REVISIONS[REPO], REV)
        self.assertRegex(REV, r"\A[0-9a-f]{40}\Z")
        self.assertIs(fam["sha256"], setup.HUIHUI_FILES)
        self.assertEqual(setup.HUIHUI_FILES, {filename(s): pin for s, pin in PINS.items()})
        for name, (size, digest) in setup.HUIHUI_FILES.items():
            with self.subTest(file=name):
                self.assertIsInstance(size, int)
                self.assertGreater(size, 0)
                self.assertRegex(digest, r"\A[0-9a-f]{64}\Z")
                self.assertNotIn("-of-", name)  # all are single-file GGUFs

    def test_model_keys(self):
        for size in SIZES:
            with self.subTest(size=size):
                self.assertEqual(setup.model_key("huihui", size), key(size))
                self.assertEqual(setup.model_key("huihui", size.lower()), key(size))
                self.assertEqual(setup.model_key("huihui", key(size)), key(size))
                self.assertEqual(setup.size_of(key(size)), size)
                self.assertIn(size, setup.model_choices())
                self.assertIn(key(size), setup.model_choices())
                self.assertIsNone(setup.model_key("qwen36", size))
                self.assertIsNone(setup.model_key("qwen", size))
        for size in ("Q8_0", "f16", "IQ3_S"):
            self.assertIsNone(setup.model_key("huihui", size))
        self.assertEqual(setup.model_key("qwen", "IQ3_S"), "IQ3_S")
        self.assertEqual(setup.model_key("ornith", "IQ3_XXS"), "Ornith-1.5-IQ3_XXS")

    def test_file_discovery(self):
        fam = setup.FAMILIES["huihui"]
        for size in SIZES:
            with self.subTest(size=size):
                name = filename(size)
                self.assertEqual(setup.model_file(fam, key(size), 1), name)
                self.assertEqual(setup.model_shards(fam, key(size)), 1)
                self.assertEqual(setup.gguf_choice(name), ("huihui", size))
                self.assertIsNone(setup.gguf_unsupported(name))
        for size in ("Q8_0", "f16"):
            self.assertIsNone(setup.gguf_choice(filename(size)))
            self.assertEqual(setup.gguf_unsupported(filename(size)).upper(), size.upper())

    def test_config_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            for size in SIZES:
                with self.subTest(size=size):
                    path = Path(tmp) / f"strata-huihui-{size.lower()}.json"
                    path.write_text(json.dumps({"args": ["--max-context", "32768", "--kv", "int8"]}),
                                    encoding="utf-8")
                    choices = setup.choices_from_config(path)
                    self.assertEqual((choices["family"], choices["model"], choices["context"], choices["kv"]),
                                     ("huihui", key(size), 32768, "int8"))
                    self.assertTrue(setup.one_gpu_family(path))
        self.assertFalse(setup.one_gpu_family(Path("strata-iq2_xs.json")))

    def test_ram_recommendations(self):
        for ram, size in ((q36.RAM16, "Q2_K"), (q36.RAM24, "Q2_K"),
                          (q36.RAM32, "Q4_K"), (q36.RAM64, "Q6_K")):
            with self.subTest(ram=ram):
                self.assertEqual(setup.small_size("huihui", ram), key(size))
        self.assertEqual(setup.small_size("huihui", q36.RAM32, 3.99), key("Q2_K"))
        self.assertFalse(setup.low_ram_needed(key("Q4_K"), q36.RAM32))
        self.assertTrue(setup.low_ram_needed(key("Q4_K"), q36.RAM24))
        self.assertFalse(setup.low_ram_needed(key("Q2_K"), q36.RAM24))
        self.assertTrue(setup.low_ram_needed(key("Q2_K"), q36.RAM16))
        self.assertTrue(setup.low_ram_fits(key("Q2_K"), q36.RAM16, 11.9))
        self.assertFalse(setup.low_ram_fits(key("Q2_K"), q36.RAM16, 7.99))
        self.assertTrue(setup.qwen36_recommended(q36.RAM24))
        self.assertFalse(setup.qwen36_recommended(q36.RAM32))
        self.assertEqual(setup.qwen36_size(q36.RAM24), q36.IQ3)

    def test_no_draft_tips(self):
        tip = " ".join(setup.small_card_note(32768, None, mtp=False))
        self.assertIn("an 8K context", tip)
        self.assertNotIn("--draft-vocab", tip)


class Install(q36.Base):
    def test_all_sizes(self):
        for size in SIZES:
            with self.subTest(size=size):
                self.downloads.clear()
                self.runs.clear()
                self.verified.clear()
                code, out, cfg = self.main(["--family", "huihui", "--model", size])
                self.assertEqual(code, 0, out)
                self.assertEqual(self.cfg_path.name, f"strata-huihui-{size.lower()}.json")
                gguf = self.t / "data" / "models" / f"huihui-{size}" / filename(size)
                pack = self.t / "data" / "packs" / f"huihui-{size.lower()}"
                native_suffix = os.path.join("models", f"huihui-{size}", filename(size))
                pack_suffix = os.path.join("packs", f"huihui-{size.lower()}")
                self.assertTrue(q36.arg(cfg, "--native").endswith(native_suffix))
                self.assertTrue(q36.arg(cfg, "--pack").endswith(pack_suffix))
                self.assertEqual(self.downloads, [BASE + filename(size)])
                self.assertEqual(self.verified, [(filename(size), *PINS[size])])
                self.assertEqual(self.runs, [[sys.executable, str(self.t / "tools" / "iq_pack.py"),
                                              "--gguf", str(gguf), "--out", str(pack), "--compat-bf16"]])
                self.assertEqual(cfg["args"], [
                    "--pack", str(pack), "--native", str(gguf),
                    "--expert-profile", str(self.t / "data" / "expert-profile-qwen36.bin"), "--expert-cache", "auto",
                    "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
                    "--max-context", "32768", "--kv", "int8"])
                self.assertEqual(cfg["model_name"], f"huihui-qwen3.6-35b-a3b-abliterated-{size.lower()}")
                self.assertEqual(cfg["tokenizer"], str(pack / "tokenizer"))
                self.assertEqual(cfg["log"], str(self.t / f"strata-huihui-{size.lower()}.log"))
                self.assertNotIn("vision", cfg)
                self.assertNotIn("draft_vocab", cfg)
                self.assertIn("MTP is off", out)
                self.assertIn("BF16/F32", out)

    def test_explicit_family_uses_ram_recommendation(self):
        code, out, cfg = self.main(["--family", "huihui"], ram=q36.RAM32)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.cfg_path.name, "strata-huihui-q4_k.json")
        self.assertNotIn("--mmap-experts", cfg["args"])
        self.assertNotIn("--resident-experts", cfg["args"])

    def test_menu_selection(self):
        number = str(list(setup.FAMILIES).index("huihui") + 1)
        code, out, cfg = self.main([], ram=q36.RAM32, answers={"Which model?": number})
        self.assertEqual(code, 0, out)
        self.assertEqual(self.cfg_path.name, "strata-huihui-q4_k.json")
        self.assertEqual(cfg["model_name"], "huihui-qwen3.6-35b-a3b-abliterated-q4_k")
        self.assertTrue(any("Which size?" in q and "[3]" in q for q in self.asked), self.asked)
        self.assertIn("huihui-ai's abliterated", out)

    def test_no_mtp_even_without_existing_draft(self):
        (self.t / "data" / "mtp" / "rt" / "experts.bin").unlink()
        with mock.patch.object(setup, "saved_draft_vocab", side_effect=AssertionError("read draft choice")), \
                mock.patch.object(setup, "draft_vocab_note", side_effect=AssertionError("offered draft tip")):
            code, out, cfg = self.main(["--family", "huihui", "--model", "Q4_K"], free=22.8)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.downloads, [BASE + filename("Q4_K")])
        self.assertEqual(len(self.runs), 1)  # iq_pack only, never mtp_fetch / mtp_pack / mtp_rt
        self.assertNotIn("--mtp", cfg["args"])
        self.assertNotIn("--mtp-draft-vocab", cfg["args"])

    def test_draft_vocab_is_unused(self):
        code, out, cfg = self.main(["--family", "huihui", "--model", "Q4_K", "--draft-vocab", "en"])
        self.assertEqual(code, 0, out)
        self.assertNotIn("draft_vocab", cfg)
        self.assertNotIn("--mtp-draft-vocab", cfg["args"])
        self.assertIn("--draft-vocab is not used", out)

    def test_gguf_dir_local_reuse(self):
        folder = self.t / "existing-ggufs"
        folder.mkdir()
        gguf = folder / filename("Q4_K")
        gguf.write_bytes(b"")  # a hand-copied file without a .done mark
        with mock.patch.object(setup, "whole_shard", return_value=True) as whole:
            code, out, cfg = self.main(["--family", "huihui", "--model", "Q4_K", "--gguf-dir", str(folder)])
        self.assertEqual(code, 0, out)
        whole.assert_called_once_with(gguf)
        self.assertTrue(setup.done(gguf))
        self.assertEqual(self.downloads, [])
        self.assertEqual(self.verified, [(filename("Q4_K"), *PINS["Q4_K"])])
        self.assertEqual(q36.arg(cfg, "--native"), str(gguf))
        self.assertEqual(q36.arg(cfg, "--pack"), str(self.t / "data" / "packs" / "huihui-q4_k"))
        self.assertNotIn("--mtp", cfg["args"])

    def test_gguf_dir_hints_at_the_family(self):
        folder = self.t / "existing-ggufs"
        folder.mkdir()
        (folder / filename("Q4_K")).write_bytes(b"")
        code, out, cfg = self.main(["--family", "qwen36", "--model", "UD-IQ4_XS", "--gguf-dir", str(folder)])
        self.assertEqual(code, 1, out)
        self.assertIsNone(cfg)
        self.assertIn("Usable here: --family huihui --model Q4_K", out)
        self.assertEqual(self.downloads, [])

    def test_already_prepared(self):
        gguf = self.t / "data" / "models" / "huihui-Q4_K" / filename("Q4_K")
        gguf.parent.mkdir(parents=True)
        gguf.write_bytes(b"")
        setup.mark(gguf)
        pack = self.t / "data" / "packs" / "huihui-q4_k"
        (pack / "tokenizer").mkdir(parents=True)
        (pack / "tokenizer" / "vocab.json").write_text("{}", encoding="utf-8")
        (pack / "native_experts.txt").write_text("", encoding="utf-8")
        code, out, cfg = self.main(["--family", "huihui", "--model", "Q4_K"])
        self.assertEqual(code, 0, out)
        self.assertEqual((self.downloads, self.runs), ([], []))
        self.assertEqual(self.verified, [(filename("Q4_K"), *PINS["Q4_K"])])
        self.assertEqual(q36.arg(cfg, "--pack"), str(pack))

    def test_wrong_size_for_the_family(self):
        code, out, cfg = self.main(["--family", "huihui", "--model", "IQ3_S"])
        self.assertEqual(code, 1, out)
        self.assertIsNone(cfg)
        self.assertIn("choose one of: Q6_K, Q5_K, Q4_K, Q3_K, Q2_K", out)
        self.assertEqual(self.downloads, [])

    def test_no_images(self):
        code, out, cfg = self.main(["--family", "huihui", "--model", "Q4_K", "--vision", "yes"])
        self.assertEqual(code, 0, out)
        self.assertNotIn("vision", cfg)
        self.assertIn("images are not available", out)
        self.assertEqual(self.downloads, [BASE + filename("Q4_K")])

    def test_engine_support_required_without_mtp(self):
        code, out, cfg = self.main(["--family", "huihui", "--model", "Q4_K"], version="0.1.41", runs_small=False)
        self.assertEqual(code, 1, out)
        self.assertIsNone(cfg)
        self.assertIn("needs an engine with the qwen35moe support", out)
        self.assertEqual(self.downloads, [])

    def test_ready_made_engine_with_support_is_used(self):
        code, out, cfg = self.main(["--family", "huihui", "--model", "Q4_K"], version="0.1.41", source="release")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.builds, [])
        self.assertEqual(cfg["exe"], str(self.t / "engine" / setup.EXE))

    def test_one_gpu(self):
        code, out, cfg = self.main(["--family", "huihui", "--model", "Q4_K", "--gpus", "0,1"], found=q36.nvidia(2))
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], 0)
        self.assertNotIn("layer_split", cfg)
        self.assertIn("runs on one GPU for now", out)

    def test_small_card_has_no_draft_advice(self):
        code, out, cfg = self.main(["--family", "huihui"], ram=q36.RAM32, found=q36.nvidia(vram=3.99))
        self.assertEqual(code, 0, out)
        self.assertEqual(self.cfg_path.name, "strata-huihui-q2_k.json")
        self.assertEqual(q36.arg(cfg, "--max-context"), "8192")
        self.assertIn("has not been measured on a 4 GB card", out)
        self.assertNotIn("--draft-vocab en", out)
        self.assertNotIn("--mtp", cfg["args"])

    def test_disk_space_counts_no_draft(self):
        (self.t / "data" / "mtp" / "rt" / "experts.bin").unlink()
        code, out, cfg = self.main(["--family", "huihui", "--model", "Q4_K"], free=22.6)
        self.assertEqual(code, 1, out)
        self.assertIn("not enough free disk space", out)
        self.assertIsNone(cfg)
        code, out, cfg = self.main(["--family", "huihui", "--model", "Q4_K"], free=22.8)  # 21.7 + 1, no draft
        self.assertEqual(code, 0, out)
        self.assertEqual(self.cfg_path.name, "strata-huihui-q4_k.json")

    def test_default_family_stays_unchanged(self):
        code, out, cfg = self.main([], ram=q36.RAM16, found=q36.nvidia(vram=7.99))
        self.assertEqual(code, 1, out)
        self.assertIsNone(cfg)
        for ram, name in ((q36.RAM24, "strata-qwen36-ud-iq3_s.json"),
                          (q36.RAM32, "strata-q2_0.json"), (q36.RAM64, "strata-iq3_xxs.json")):
            with self.subTest(ram=ram):
                code, out, cfg = self.main([], ram=ram)
                self.assertEqual(code, 0, out)
                self.assertEqual(self.cfg_path.name, name)

    def test_run_script_names(self):
        cfg_path = self.t / "strata-huihui-q4_k.json"
        for win, suffix in ((True, "bat"), (False, "sh")):
            with self.subTest(windows=win), mock.patch.object(setup, "ROOT", self.t), mock.patch.object(setup, "WIN", win):
                script = setup.write_run_script("huihui-Q4_K", cfg_path, 8080, open_browser=False)
                self.assertEqual(script, self.t / f"run-huihui-q4_k.{suffix}")
                text = script.read_text(encoding="utf-8")
                self.assertIn(str(cfg_path), text)
                self.assertNotIn("--open", text)

    def test_help(self):
        code, out, cfg = self.main(["--help"])
        self.assertEqual(code, 0, out)
        self.assertIsNone(cfg)
        help_text = " ".join(out.split())  # argparse wraps at the terminal width
        self.assertIn("huihui = Huihui's abliterated", help_text)
        self.assertIn("Q2_K-Q6_K, MTP off", help_text)
        self.assertIn("the chosen single file", help_text)


if __name__ == "__main__":
    unittest.main()
