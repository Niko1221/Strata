"""launcher/test_presets.py - what a preset is and what it turns into, without a GPU or a download.

    python -m unittest launcher.test_presets -v

A preset is only useful if it says exactly what setup.py's flags say, so these tests check the two directions that
matter: the choices a user cannot make (a size a family does not publish, a context setup does not offer, an address
without a key, images on a family that has none) are refused with the reason, and the flags a preset writes are the
flags setup would be given - including that a preset's own flag wins over the one the install planner picked, and
that a field left empty is not passed at all, so setup's own default for this PC still decides.
"""
from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from launcher import presets as P  # noqa: E402

CAT = {
    "models": {
        "Q2_0": {"about": "2-bit, the fastest", "download_gb": 66.4, "ram_gb": 48, "arena_gb": 34.0,
                 "families": ("qwen", "swift")},
        # like setup.py's own table: a size that names no families belongs to the original model and its fine-tunes
        "IQ3_XXS": {"about": "3-bit i-quant", "download_gb": 75.8, "ram_gb": 60, "arena_gb": 42.9},
        "IQ3_S": {"about": "3.5-bit", "download_gb": 83.6, "ram_gb": 62, "arena_gb": 50.3, "families": ("qwen",)},
        "IQ1_M": {"about": "the Coder", "download_gb": 58.4, "ram_gb": 32, "arena_gb": 23.4,
                  "families": ("coder",)},
        "UD-Q4_K_XL": {"about": "4-bit", "download_gb": 111.3, "ram_gb": 48, "arena_gb": 77.0,
                       "families": ("unsloth",), "budget": True},
    },
    "families": {
        "qwen": {"title": "Qwen3.8-Flash-Next", "by": "Qwen", "hf": "…", "file": "…", "tag": "",
                 "name": "qwen3.8-flash-next"},
        "swift": {"title": "Swift 1.5", "by": "UkisAI's fine-tune", "hf": "…", "file": "…", "tag": "swift-",
                  "name": "swift-1.5"},
        "coder": {"title": "Qwen3.8-Flash-Next Coder", "by": "ISTA", "hf": "…", "file": "…", "tag": "coder-",
                  "name": "qwen3.8-flash-next-coder"},
        "unsloth": {"title": "Unsloth", "by": "Unsloth", "hf": "…", "file": "…", "tag": "unsloth-",
                    "name": "qwen3.8-flash-next-unsloth", "experimental": True, "vision": False},
    },
    "contexts": [8192, 32768, 65536, 131072, 262144, 393216],
    "ram_gb": 64.0,
}
REC = {"family": "qwen", "model": "IQ3_S", "context": 65536, "backend": "cuda"}


def base(**over):
    """A valid preset with one thing changed."""
    p = {"name": "My 32K", "family": "qwen", "model": "Q2_0", "context": 32768, "vision": "no", "calibrate": "ask"}
    return dict(p, **over)


def errors(**over):
    try:
        P.clean(base(**over), CAT)
    except P.PresetError as e:
        return str(e)
    return None


def args(**over):
    return P.setup_args(P.clean(base(**over), CAT))


class Choices(unittest.TestCase):
    def test_a_valid_preset_is_accepted(self):
        p = P.clean(base(), CAT)
        self.assertEqual(p["family"], "qwen")
        self.assertEqual(p["context"], 32768)
        self.assertEqual(p["id"], "my-32k")
        self.assertEqual(p["calibrate"], "ask")

    def test_every_problem_is_reported_at_once(self):
        msg = errors(family="coder", model="IQ3_S", context=400, host="0.0.0.0", gpu=0, gpus="0,1")
        self.assertIn("Coder has no IQ3_S", msg)
        self.assertIn("outside what Strata takes", msg)
        self.assertIn("needs an API key", msg)
        self.assertIn("choose one GPU or a list", msg)

    def test_a_context_between_setup_s_sizes_or_of_your_own_is_taken(self):
        # setup takes any --context integer; its menu only lists some lengths, and the launcher adds 192K to them
        self.assertEqual(P.clean(base(context=196608), CAT)["context"], 196608)
        self.assertEqual(P.clean(base(context="200000"), CAT)["context"], 200000)
        self.assertEqual(args(context=196608)[args(context=196608).index("--context") + 1], "196608")
        self.assertIn("outside what Strata takes", errors(context=400))
        self.assertIn("outside what Strata takes", errors(context=2000000))
        self.assertIn("whole number of tokens", errors(context="a lot"))

    def test_a_size_is_only_offered_for_the_families_that_publish_it(self):
        self.assertIn("Unsloth has no IQ3_S", errors(family="unsloth", model="IQ3_S"))
        self.assertIsNone(errors(family="coder", model="IQ1_M"))

    def test_the_backend_field_offers_no_value_setup_refuses(self):
        # setup's --backend is cuda, hip or sycl: there is no "auto" value, so "leave it to setup" is the empty
        # field, which writes no flag; a preset saved when the page still said "auto" means the same thing
        self.assertEqual(args(backend="auto").count("--backend"), 0)
        self.assertEqual(args(backend="").count("--backend"), 0)
        self.assertEqual(args(backend="hip")[args(backend="hip").index("--backend") + 1], "hip")
        self.assertIn("not one of cuda, hip, sycl", errors(backend="metal"))

    def test_the_kv_labels_do_not_claim_a_streaming_setup_does_not_write(self):
        # 0.1.40 (#711) streams k8v4's KV too: --kv-resident is written for it.  A label that says a precision
        # "does not stream" is a promise the page makes about setup, so it is checked against setup's own rule
        import setup as S
        for kv, label in P.KV.items():
            if S.kv_streaming_wanted("IQ3_S", 196608, kv, 400.0):        # a PC with room for any of these caches
                self.assertNotIn("does not stream", label, f"{kv} streams: the label says otherwise")

    def test_no_select_offers_a_choice_setup_does_not_take(self):
        # a value setup's parser refuses reaches the user as argparse's own "invalid choice", which is how
        # --backend auto looked like a working choice when it never was one
        src = (Path(__file__).resolve().parents[1] / "setup.py").read_text(encoding="utf-8")
        defined = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', src))
        takes = {m.group(1): set(re.findall(r'"([^"]+)"', m.group(3)))
                 for m in re.finditer(r'add_argument\(\s*"(--[a-z0-9-]+)"([^)]*?)choices=(\[[^\]]*\])', src)}
        for f in P.FIELDS:
            flag = f.get("flag")
            if not flag:
                continue
            self.assertIn(flag, defined, f"the page writes {flag}, which setup.py does not take")
            if "choices" in f and flag in takes:
                extra = {v for v in f["choices"] if v and v not in takes[flag]}
                self.assertFalse(extra, f"{flag}: setup takes {sorted(takes[flag])}, the page offers {sorted(extra)}")

    def test_a_size_that_names_no_families_belongs_to_the_fine_tunes_too(self):
        # setup.py lists Swift's IQ2_XS and IQ3_XXS without a "families" key.  Reading that table as if such a size
        # were the original's alone makes the page offer a size the save then refuses, with an empty list of sizes
        # to choose from ("Swift 1.5 has no IQ3_XXS chosen; its sizes:").
        p = P.clean(base(family="swift", model="IQ3_XXS"), CAT)
        self.assertEqual((p["family"], p["model"]), ("swift", "IQ3_XXS"))
        self.assertEqual(P.model_id(p, CAT), "swift-iq3_xxs")
        self.assertEqual(P.model_name(p, CAT), "swift-1.5-iq3_xxs")
        self.assertEqual(args(family="swift", model="IQ3_XXS")[:6],
                         ["--yes", "--no-start", "--family", "swift", "--model", "IQ3_XXS"])
        # a size the family really does not publish is still refused, and now says what its sizes are
        self.assertIn("Swift 1.5 has no IQ3_S chosen; its sizes: Q2_0, IQ3_XXS", errors(family="swift", model="IQ3_S"))

    def test_images_follow_what_the_family_has(self):
        self.assertIn("images are not available", errors(family="unsloth", model="UD-Q4_K_XL", vision="gpu"))
        self.assertIn("AMD backend has no image encoder", errors(vision="gpu", backend="hip"))

    def test_a_context_past_the_trained_one_is_setup_s_to_scale(self):
        # the page does not offer --rope-scaling: setup adds yarn by itself, so a 384K preset is simply passed on
        self.assertIsNone(errors(context=393216))

    def test_the_ram_budget_is_only_for_the_size_that_needs_it(self):
        self.assertIn("only for", errors(resident_budget_gib=32))
        self.assertIsNone(errors(family="unsloth", model="UD-Q4_K_XL", resident_budget_gib=32))

    def test_numbers_are_numbers(self):
        self.assertIn("is not a number", errors(vram_reserve_mib="a lot"))
        self.assertIn("must be 0 or more", errors(vram_reserve_mib=-5))
        self.assertIn("between 1024 and 65535", errors(port=80))
        self.assertEqual(P.clean(base(port="8123"), CAT)["port"], 8123)

    def test_unknown_keys_are_dropped(self):
        p = P.clean(dict(base(), colour="blue", context="32768"), CAT)
        self.assertNotIn("colour", p)
        self.assertEqual(p["context"], 32768)


class Flags(unittest.TestCase):
    def test_an_empty_field_is_not_passed(self):
        a = args()
        for flag in ("--kv", "--gpus", "--vram-reserve-mib", "--api-key"):
            self.assertNotIn(flag, a)
        self.assertEqual(a[:7], ["--yes", "--no-start", "--family", "qwen", "--model", "Q2_0", "--context"])
        self.assertEqual(a[a.index("--context") + 1], "32768")

    def test_a_filled_field_is_passed(self):
        a = args(kv="q4_0", vram_reserve_mib=1500, gpus="0,1", layer_split="18",
                 kv_streaming="on", low_ram="resident", resident_budget_gib=None)
        for flag, value in (("--kv", "q4_0"), ("--vram-reserve-mib", "1500"),
                            ("--gpus", "0,1"), ("--layer-split", "18"), ("--kv-streaming", "on"),
                            ("--low-ram", "resident")):
            self.assertIn(flag, a)
            self.assertEqual(a[a.index(flag) + 1], value)

    def test_images_yes_means_the_gpu_encoder(self):
        a = args(vision="gpu")
        self.assertEqual(a[a.index("--vision") + 1], "gpu")

    def test_the_preset_wins_over_the_install_plan(self):
        plan = ["--yes", "--no-start", "--family", "qwen", "--model", "IQ3_S", "--context", "32768",
                "--vision", "no", "--experimental-speed-projection", "off", "--port", "8080"]
        merged = P.merge_args(plan, P.setup_args(P.clean(base(speed_projection="on", port=8123), CAT)))
        self.assertEqual(merged.count("--experimental-speed-projection"), 1)
        self.assertEqual(merged[merged.index("--experimental-speed-projection") + 1], "on")
        self.assertEqual(merged[merged.index("--port") + 1], "8123")
        self.assertEqual(merged.count("--family"), 1)

    def test_a_flag_only_in_the_preset_is_appended(self):
        merged = P.merge_args(["--yes", "--model", "Q2_0"], P.setup_args(P.clean(base(kv="k8v4"), CAT)))
        self.assertEqual(merged[merged.index("--kv") + 1], "k8v4")


class Names(unittest.TestCase):
    def test_the_model_id_is_the_config_and_script_tag(self):
        self.assertEqual(P.model_id(base(), CAT), "q2_0")
        self.assertEqual(P.model_id(base(family="coder", model="IQ1_M"), CAT), "coder-iq1_m")

    def test_the_model_name_is_what_the_server_answers_to(self):
        self.assertEqual(P.model_name(base(), CAT), "qwen3.8-flash-next-q2_0")
        self.assertEqual(P.model_name(base(family="coder", model="IQ1_M"), CAT),
                         "qwen3.8-flash-next-coder-iq1_m")

    def test_two_names_do_not_collide_in_the_store(self):
        with tempfile.TemporaryDirectory() as d:
            s = P.Store(Path(d) / "presets.json")
            s.put(P.clean(base(), CAT))
            s.put(P.clean(base(name="My 32K!"), CAT))          # slugs the same
            ids = sorted(p["id"] for p in s.load())
            self.assertEqual(ids, ["my-32k", "my-32k-2"])
            self.assertTrue(s.delete("my-32k-2"))
            self.assertEqual([p["id"] for p in s.load()], ["my-32k"])

    def test_a_broken_file_is_not_a_store(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "presets.json"
            f.write_text("{", encoding="utf-8")
            self.assertEqual(P.Store(f).load(), [])


class Derived(unittest.TestCase):
    def test_builtin_presets_only_offer_sizes_this_pc_has_room_for(self):
        items = P.builtin(CAT, REC)
        names = [p["name"] for p in items]
        self.assertIn("Recommended for this PC", names)
        self.assertIn("Coding", names)
        for p in items:
            self.assertTrue(p["id"].startswith("builtin-"))
            self.assertEqual(p["context"], 65536)          # the context setup would pick for this card
        big = dict(CAT, ram_gb=32.0)          # a PC where only the Coder fits: the derived presets follow that
        low = P.builtin(big, {"family": "coder", "model": "IQ1_M", "context": 32768, "why": "32 GB of RAM"})
        self.assertNotIn("IQ3_S", [p["model"] for p in low])
        self.assertIn("IQ1_M", [p["model"] for p in low])

    def test_a_preset_from_an_installed_model_reads_its_config(self):
        desc = {"model": "coder-iq1_m", "config": "strata-coder-iq1_m.json", "context": 65536, "kv": "int8",
                "images": "off", "port": 8080, "host": "127.0.0.1", "gpu": [0, 1], "layer_split": [18]}
        cfg = {"args": ["--max-context", "65536", "--kv", "int8", "--vram-reserve-mib", "1500",
                        "--control-vector-scaled", "x.gguf:1.0", "--mmap-experts", "1"]}
        p = P.from_config(desc, cfg, CAT)
        self.assertEqual((p["family"], p["model"]), ("coder", "IQ1_M"))
        self.assertEqual(p["context"], 65536)
        self.assertEqual(p["vram_reserve_mib"], 1500)
        self.assertEqual(p["speed_projection"], "on")
        self.assertEqual(p["low_ram"], "mmap")
        self.assertEqual(p["gpus"], "0,1")
        self.assertEqual(p["layer_split"], "18")
        self.assertEqual(P.model_id(p, CAT), "coder-iq1_m")

    def test_the_summary_says_what_the_preset_is(self):
        line = P.summary(P.clean(base(kv="q4_0", vision="gpu", gpus="0,1"), CAT), CAT)
        self.assertIn("Qwen3.8-Flash-Next Q2_0", line)
        self.assertIn("32K", line)
        self.assertIn("kv q4_0", line)
        self.assertIn("images", line)
        self.assertIn("GPUs 0,1", line)


if __name__ == "__main__":
    unittest.main()
