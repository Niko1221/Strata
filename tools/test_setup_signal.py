"""agentionai's Signal in setup: its two one-file sizes at the pinned revision, the BF16 token embedding setup writes
for it and the --embd-gguf that hands it to the engine, its own image encoder under a name of its own, and the file
and config names that find its size again.  No GPU, no downloads: setup.main() runs on a mocked PC.

    python -m unittest tools.test_setup_signal
"""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import setup  # noqa: E402

SIZES = ("AP-Q4_K_XL", "AP-IQ4_XS")
FAM = setup.FAMILIES["signal"]


class FakeGGUF:
    """gguf_reader.GGUFFile: Signal's one file holds the PLE table itself."""
    def __init__(self, path):
        self.tensors = [types.SimpleNamespace(name=n) for n in ("per_layer_token_embd.weight", "token_embd.weight")]


def signal_install(model, *more):
    """setup.main() with --yes for a Signal size on a 93 GB PC with an RTX 4090 -> (code, output, config, run calls)."""
    from test_setup_golden import card, install
    calls = []
    code, out, cfg, _ = install(
        93.0, [card(0, "NVIDIA GeForce RTX 4090", 24.0, "89")],
        ["--family", "signal", "--model", model, "--no-start", *more],
        extra=[mock.patch.object(setup, "run", lambda cmd, **k: calls.append([str(c).replace("\\", "/") for c in cmd])),
               mock.patch.dict(sys.modules, {"gguf_reader": types.SimpleNamespace(GGUFFile=FakeGGUF)})])
    return code, out, cfg, calls


class Files(unittest.TestCase):
    def test_one_pinned_file_per_size(self):
        for m in SIZES:
            with self.subTest(m):
                name = setup.model_file(FAM, m, 1)
                self.assertEqual(name, f"Signal-3.8-Flash-Next-{m}.gguf")
                self.assertEqual(setup.model_shards(FAM, m), 1)
                self.assertIn(name, FAM["sha256"])                 # size and SHA-256 checked after the download
                self.assertEqual(FAM["hf"].format(q=m) + name,
                                 f"{setup.hf_endpoint()}/agentionai/Signal-3.8-Flash-Next-GGUF/resolve/"
                                 f"eba565bea194b899480ebd8a8ab63d537917004b/{m}/{name}")

    def test_the_sizes_belong_to_signal_alone(self):
        self.assertEqual([m for m in setup.MODELS if "signal" in setup.MODELS[m].get("families", ("qwen", "swift"))],
                         list(SIZES))
        # every format has AMD kernels: Q4_K experts run on an R9700 (UD-Q4_K_XL), IQ3_S / IQ4_NL on a Strix Halo
        self.assertFalse(setup.MODELS["AP-Q4_K_XL"].get("nvidia_only"))  # (docs/QUANTS.md), so neither size asks on AMD
        self.assertFalse(setup.MODELS["AP-IQ4_XS"].get("nvidia_only"))   # IQ3_S / IQ4_NL: UD-IQ4_XS's types

    def test_the_ram_budget_holds_at_most_its_experts(self):
        self.assertEqual(setup.resident_budget_gib("AP-Q4_K_XL", 93.0), 63)     # 67.9 GB of experts
        self.assertEqual(setup.resident_budget_gib("AP-IQ4_XS", 93.0), 53)      # 57.3 GB
        self.assertEqual(setup.resident_budget_gib("AP-Q4_K_XL", 63.7), 40)     # 64 GB: the measured setting


class Names(unittest.TestCase):
    def test_a_config_s_file_names_its_size(self):
        for m in SIZES:
            cfg = {"args": ["--native", f"/m/signal-{m}/Signal-3.8-Flash-Next-{m}.gguf"]}
            self.assertEqual(setup.budget_model(cfg), m)
        cfg = {"args": ["--native", "/m/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"]}
        self.assertEqual(setup.budget_model(cfg), "UD-IQ4_XS")         # Unsloth's names as before

    def test_gguf_dir_takes_signal_s_files_and_not_its_q8_0(self):
        for m in SIZES:
            name = f"Signal-3.8-Flash-Next-{m}.gguf"
            self.assertIsNone(setup.gguf_unsupported(name))
            self.assertEqual(setup.gguf_choice(name), ("signal", m))
        # Q8_0 is a size of its own now (Unsloth's, for a 192 GB PC), so its name alone is no longer refused by
        # gguf_unsupported; what keeps Signal's Q8_0 file out of a --gguf-dir install is gguf_choice, which knows
        # only the two sizes Signal ships.
        self.assertIsNone(setup.gguf_unsupported("Signal-3.8-Flash-Next-Q8_0.gguf"))
        self.assertIsNone(setup.gguf_choice("Signal-3.8-Flash-Next-Q8_0.gguf"))
        self.assertNotIn("Q8_0", [m for m in setup.MODELS if "signal" in setup.MODELS[m].get("families", ())])
        self.assertIn("Signal AP-Q4_K_XL and AP-IQ4_XS", setup.SUPPORTED_GGUFS)

    def test_gguf_dir_finds_the_one_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "Signal-3.8-Flash-Next-AP-IQ4_XS.gguf"
            f.write_bytes(b"")
            self.assertEqual(setup.gguf_dir_shards(Path(tmp), FAM, "AP-IQ4_XS"), [f])

    def test_setup_run_again_reads_the_size_from_the_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "strata-signal-ap-iq4_xs.json"
            p.write_text(json.dumps({"args": ["--max-context", "262144", "--kv", "int8"],
                                     "vision": {"gpu": True}}), encoding="utf-8")
            ch = setup.choices_from_config(p)
        self.assertEqual((ch["family"], ch["model"], ch["vision"], ch["kv"]), ("signal", "AP-IQ4_XS", "gpu", "int8"))


class Install(unittest.TestCase):
    def test_the_pack_the_embedding_and_the_config(self):
        for m in SIZES:
            with self.subTest(m):
                code, out, cfg, calls = signal_install(m)
                self.assertEqual(code, 0, out[-3000:])
                gguf = f"<T>/models/signal-{m}/Signal-3.8-Flash-Next-{m}.gguf"
                pack = f"<T>/data/packs/signal-{m.lower()}"
                embd = f"{pack}/token-embd-bf16.gguf"
                iq = next(c for c in calls if c[1].endswith("tools/iq_pack.py"))
                self.assertIn("--compat-bf16", iq)                 # its Q6_K dense weights become BF16
                self.assertTrue(iq[iq.index("--gguf") + 1].endswith(gguf[len("<T>"):]))
                emb = next(c for c in calls if c[1].endswith("tools/embd_bf16_pack.py"))
                self.assertEqual(emb[2:3] + emb[4:5], ["--gguf", "--out"])
                self.assertTrue(emb[5].endswith(embd[len("<T>"):]))
                a = cfg["args"]
                self.assertEqual(a[a.index("--native") + 1], gguf)
                self.assertEqual(a[a.index("--ple-gguf") + 1], gguf)      # the one file holds the PLE table
                self.assertEqual(a[a.index("--embd-gguf") + 1], embd)
                # 93 GB holds its experts and the 16 GiB beside them, so every expert stays in RAM and setup writes no
                # RAM budget (setup.unsloth_budget, the rule upstream added for Unsloth's files)
                self.assertNotIn("--resident-budget-gib", a)
                self.assertEqual(a[a.index("--kv") + 1], "int8")         # the authors: no 4-bit V cache
                self.assertEqual(cfg["tokenizer"], f"{pack}/tokenizer")

    def test_a_ram_budget_when_asked_for_one(self):
        """A 93 GB PC holds Signal's experts with the 16 GiB beside them, so setup keeps every expert in RAM
        (setup.unsloth_budget, the same rule as Unsloth's files); --resident-budget-gib still asks for the budget
        mode, which is how the numbers in docs/SIGNAL_AP_Q4_K_XL.md were measured (a 63 GiB budget on a 96 GB PC)."""
        for m in SIZES:
            with self.subTest(m):
                code, out, cfg, _ = signal_install(m, "--resident-budget-gib", "40")
                self.assertEqual(code, 0, out[-3000:])
                a = cfg["args"]
                self.assertIn("--resident-budget-gib", a)
                self.assertEqual(a[a.index("--resident-budget-gib") + 1], "40")

    def test_its_own_image_encoder_under_its_own_name(self):
        downloads = []

        def fake_download(url, dst, what=None):
            downloads.append((url, dst.name))
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)

        with mock.patch.object(setup, "VEXE", setup.EXE):         # the mocked engine folder has no strata-vision
            from test_setup_golden import card, install
            code, out, cfg, _ = install(
                93.0, [card(0, "NVIDIA GeForce RTX 4090", 24.0, "89")],
                ["--family", "signal", "--model", "AP-Q4_K_XL", "--no-start", "--vision", "gpu"],
                extra=[mock.patch.object(setup, "download", fake_download),
                       mock.patch.object(setup, "get_prebuilt", lambda *a, **k: setup.ROOT / "engine"),
                       mock.patch.dict(sys.modules, {"gguf_reader": types.SimpleNamespace(GGUFFile=FakeGGUF)})])
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn((f"{setup.hf_endpoint()}/agentionai/Signal-3.8-Flash-Next-GGUF/resolve/"
                       "eba565bea194b899480ebd8a8ab63d537917004b/mmproj-F16.gguf",
                       "mmproj-Signal-3.8-Flash-Next-F16.gguf"), downloads)
        self.assertTrue(cfg["vision"]["mmproj"].endswith("/models/mmproj-Signal-3.8-Flash-Next-F16.gguf"))
        self.assertTrue(cfg["vision"]["gpu"])

    def test_the_other_families_get_no_embedding_file(self):
        self.assertEqual([f for f, d in setup.FAMILIES.items() if d.get("embd_bf16")], ["signal"])


if __name__ == "__main__":
    unittest.main()
