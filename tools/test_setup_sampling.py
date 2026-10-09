"""Tests for #1129: the sampling defaults a client that asks for none gets (--thinking / --instruct).  No GPU, no
network, nothing installed: the numbers are the model card's own, setup writes them into strata-<model>.json, thinking
is what a setup run with no flag writes while numbers of the user's own survive it, a start saves a new preset for the
model, and the settings line of a start says which numbers are in use.

    python -m unittest tools.test_setup_sampling
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from serve import runconfig  # noqa: E402
from serve.server import sampling_defaults_from_config  # noqa: E402
from test_setup_golden import PROFILES, install  # noqa: E402

# https://huggingface.co/Qwen/Qwen3.8-Flash-Next, "Sampling Parameters" (the numbers issue #1129 quotes)
THINKING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0, "repetition_penalty": 1.0}
INSTRUCT = {"temperature": 0.7, "top_p": 0.80, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 1.5, "repetition_penalty": 1.0}


def quiet(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*args, **kw), out.getvalue()


class Preset(unittest.TestCase):
    """The two sets are the card's numbers, and the server takes them as they are."""

    def test_the_cards_numbers(self):
        self.assertEqual(runconfig.SAMPLING_PRESETS, {"thinking": THINKING, "instruct": INSTRUCT})

    def test_the_server_reads_the_block_as_setup_wrote_it(self):
        for name, block in runconfig.SAMPLING_PRESETS.items():
            with self.subTest(name):
                self.assertEqual(sampling_defaults_from_config({"sampling": dict(block)}), block)

    def test_a_preset_is_named_and_their_own_numbers_are_not(self):
        self.assertEqual(setup.sampling_choice("thinking"), THINKING)
        self.assertEqual(setup.sampling_choice("instruct"), INSTRUCT)
        self.assertIsNone(setup.sampling_choice(None))          # no preset in the name
        self.assertEqual(runconfig.DEFAULT_SAMPLING_PRESET, "thinking")     # what a run that names no flag writes
        self.assertIsNone(setup.sampling_choice("nope"))
        self.assertEqual(runconfig.preset_of(dict(THINKING)), "thinking")
        self.assertEqual(runconfig.preset_of(dict(INSTRUCT)), "instruct")
        for own in ({}, {"temperature": 1.0, "top_p": 0.95, "top_k": 20},   # the shorter block of an earlier setup
                    {"temperature": 0.7, "top_p": 0.8, "top_k": 20}):       # half of the instruct set
            self.assertIsNone(runconfig.preset_of(own), own)


class Install(unittest.TestCase):
    """setup.main() on a mocked PC: the flag writes that preset into the new config, no flag writes the default
    thinking preset, and numbers the file already has of its own stay."""
    RAM, CARDS = PROFILES["64GB-1x32GB"]
    ARGV = ["--family", "qwen", "--model", "IQ3_S", "--no-start"]

    def written(self, flags):
        code, out, cfg, asked = install(self.RAM, self.CARDS, self.ARGV + flags)
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(asked, [])
        return cfg, out

    def test_thinking(self):
        cfg, out = self.written(["--thinking"])
        self.assertEqual(cfg["sampling"], THINKING)
        self.assertIn("sampling for requests that send none: thinking", out)

    def test_instruct(self):
        cfg, out = self.written(["--instruct"])
        self.assertEqual(cfg["sampling"], INSTRUCT)
        self.assertIn("sampling for requests that send none: instruct", out)

    def test_no_flag_writes_the_default_thinking_block(self):
        cfg, out = self.written([])
        self.assertEqual(cfg["sampling"], THINKING)                 # thinking is the default
        self.assertIn("sampling for requests that send none: thinking", out)

    def test_numbers_written_by_hand_survive_a_run_that_names_no_flag(self):
        own = {"temperature": 0.5, "top_k": 8}
        code, out, cfg, _ = install(self.RAM, self.CARDS, self.ARGV, configs=(
            ("strata-iq3_s.json", {"args": ["--max-context", "131072"], "sampling": dict(own)}),))
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["sampling"], own)                      # the default does not replace them
        self.assertIn("sampling for requests that send none: your own numbers", out)

    def test_both_flags_stop(self):
        code, _, cfg, _ = install(self.RAM, self.CARDS, self.ARGV + ["--thinking", "--instruct"])
        self.assertEqual(code, 2)                                # argparse's own error
        self.assertIsNone(cfg)


class SetupRunAgain(unittest.TestCase):
    """#629 with #1129: the block is the user's, so setup run again keeps it - unless this run names a preset."""
    OLD = {"args": ["--max-context", "65536"], "sampling": {"temperature": 0.5, "top_k": 8}}

    def test_a_block_of_their_own_survives_a_run_that_names_no_preset(self):
        cfg = {"args": ["--max-context", "131072"]}
        kept = setup.carry_over(json.loads(json.dumps(self.OLD)), cfg)
        self.assertIn("sampling", kept)
        self.assertEqual(cfg["sampling"], self.OLD["sampling"])

    def test_a_flag_beats_the_earlier_block(self):
        cfg = {"args": [], "sampling": setup.sampling_choice("instruct")}
        kept = setup.carry_over(json.loads(json.dumps(self.OLD)), cfg)
        self.assertNotIn("sampling", kept)
        self.assertEqual(cfg["sampling"], INSTRUCT)

    def test_the_choices_of_a_config_include_the_preset(self):
        with tempfile.TemporaryDirectory() as tmp:
            for block, want in ((THINKING, "thinking"), (INSTRUCT, "instruct"),
                                ({"temperature": 0.5, "top_k": 8}, None), (None, None)):
                cfg = {"args": ["--max-context", "65536", "--kv", "int8"]}
                if block is not None:
                    cfg["sampling"] = block
                p = Path(tmp) / "strata-qwen-iq3_s.json"
                p.write_text(json.dumps(cfg), encoding="utf-8")
                with self.subTest(block=block):
                    self.assertEqual(setup.choices_from_config(p)["sampling"], want)


    def test_a_block_that_is_not_a_set_of_numbers_is_replaced_and_said(self):
        """A "sampling" that is a name, a list or an empty block is not numbers: a run that names no flag says which
        value it replaces instead of writing thinking quietly (the server would only refuse to start on such one)."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "strata-iq3_s.json"
            for bad in ("warm", 5, ["temperature"], {}):
                with self.subTest(bad=bad):
                    p.write_text(json.dumps({"sampling": bad}), encoding="utf-8")
                    preset, block, dropped = setup.sampling_for_setup(None, p)
                    self.assertEqual(preset, "thinking")
                    self.assertEqual(block, THINKING)                        # and what it writes in its place
                    self.assertEqual(dropped, bad)

    def test_nothing_is_reported_when_no_block_of_ours_is_replaced(self):
        """No key, a key set to null, no config file at all: thinking with no word about it, because nothing the user
        wrote is going away; and a block that is numbers stays the user's own."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "strata-iq3_s.json"
            p.write_text(json.dumps({"sampling": None}), encoding="utf-8")
            self.assertEqual(setup.sampling_for_setup(None, p)[:2],
                             ("thinking", setup.sampling_choice("thinking")))
            self.assertIsNone(setup.sampling_for_setup(None, p)[2])
            self.assertIsNone(setup.sampling_for_setup(None, Path(tmp) / "strata-nothing.json")[2])
            p.write_text(json.dumps({"sampling": {"temperature": 0.5}}), encoding="utf-8")
            self.assertIsNone(setup.sampling_for_setup(None, p)[2])

    def test_an_earlier_config_counts_for_its_own_model_only(self):
        """#1129 with #629: another folder's config lends its sampling only when it names this model - the same
        condition its other keys are carried over on, so a block cannot survive by itself and mislead."""
        with tempfile.TemporaryDirectory() as tmp:
            here = Path(tmp) / "strata-iq3_s.json"
            there = Path(tmp) / "earlier" / "strata-coder-q2_0.json"
            there.parent.mkdir()
            self.assertIsNone(setup.sampling_older(here, None))               # nothing installed yet
            there.write_text("{}", encoding="utf-8")
            self.assertIsNone(setup.sampling_older(here, there))              # another model's config
            same = there.parent / here.name
            same.write_text("{}", encoding="utf-8")
            self.assertEqual(setup.sampling_older(here, same), same)          # this model's earlier install
            here.write_text(json.dumps({"sampling": {"temperature": 0.5}}), encoding="utf-8")
            self.assertEqual(setup.sampling_older(here, None), here)          # the config being rewritten wins

class StartSavesTheChoice(unittest.TestCase):
    """A model already installed: ./setup.sh --instruct saves the preset for it, like --vram-reserve-mib does."""
    CFG = {"exe": "engine/strata", "args": ["--max-context", "65536"], "model_name": "qwen3.8-flash-next-iq3_s"}

    def config(self, tmp: Path, block=None) -> Path:
        p = tmp / "strata-qwen-iq3_s.json"
        p.write_text(json.dumps({**self.CFG, **({"sampling": block} if block else {})}), encoding="utf-8")
        return p

    def test_a_start_with_a_flag_saves_it_for_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.config(Path(tmp), {"temperature": 0.5})
            _, out = quiet(setup.save_sampling_choice, p, json.loads(p.read_text(encoding="utf-8")), "thinking")
            self.assertEqual(json.loads(p.read_text(encoding="utf-8"))["sampling"], THINKING)
            self.assertIn("saved for this model: sampling thinking", out)
            self.assertIn("from its next start", out)           # the server reads the config when it starts

            # what it replaced was the user's own numbers, so the file is kept first, as a setup run keeps it (#629)
            self.assertIn("kept as strata-qwen-iq3_s.json.bak", out)
            self.assertEqual(json.loads((p.parent / (p.name + ".bak")).read_text(encoding="utf-8"))["sampling"],
                             {"temperature": 0.5})

    def test_a_start_without_a_flag_or_with_the_preset_it_already_has_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode, block in ((None, THINKING), ("thinking", THINKING), (None, None), ("nope", None)):
                with self.subTest(mode=mode, block=block):
                    p = self.config(Path(tmp), block)
                    before = p.read_text(encoding="utf-8")
                    done, out = quiet(setup.save_sampling_choice, p, json.loads(before), mode)
                    self.assertFalse(done)
                    self.assertEqual(p.read_text(encoding="utf-8"), before)
                    self.assertEqual(out, "")

            self.assertFalse((p.parent / (p.name + ".bak")).exists())   # a start that changes nothing leaves no file

    def test_a_start_saves_the_other_settings_it_named(self):
        # #179 #493: --host / --api-key / --draft-vocab / --no-browser and --vram-reserve-mib ride the same write as
        # --thinking / --instruct (save_start_settings, which sycl/setup_intel.py calls before its own run script).
        with tempfile.TemporaryDirectory() as tmp:
            p = self.config(Path(tmp))
            named = {"host": "0.0.0.0", "api_key": "secret", "draft_vocab": "en", "open_browser": False,
                     "vram_reserve_mib": 2048, "sampling_mode": None, "layer_split": None}
            _, out = quiet(setup.save_start_settings, p, json.loads(p.read_text(encoding="utf-8")), named)
            wrote = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual({k: wrote[k] for k in ("host", "api_key", "draft_vocab", "open_browser")},
                             {"host": "0.0.0.0", "api_key": "secret", "draft_vocab": "en", "open_browser": False})
            self.assertEqual(wrote["args"][-2:], ["--vram-reserve-mib", "2048"])
            self.assertNotIn("sampling", wrote)                        # no --thinking / --instruct on this start
            self.assertNotIn("layer_split", wrote)                     # not given: no key, no empty value
            self.assertIn("2048 MiB of VRAM kept free", out)
            self.assertIn("api key", out)
            self.assertIn("no browser", out)

            # the same settings named again: keys that already hold them write nothing and say nothing
            before = p.read_text(encoding="utf-8")
            _, out = quiet(setup.save_start_settings, p, json.loads(before), {k: v for k, v in named.items()
                                                                             if k != "vram_reserve_mib"})
            self.assertEqual(p.read_text(encoding="utf-8"), before)
            self.assertEqual(out, "")

    def test_the_settings_line_of_a_start_shows_the_numbers(self):
        self.assertIn("sampling thinking: temperature=1.0, top_p=0.95",
                      setup.settings_summary({"args": ["--max-context", "65536"], "sampling": THINKING}, 8080))
        self.assertIn("sampling own: temperature=0.5",
                      setup.settings_summary({"args": [], "sampling": {"temperature": 0.5}}, 8080))
        self.assertNotIn("sampling", setup.settings_summary({"args": ["--max-context", "65536"]}, 8080))
        # the card's numbers with a key set to null are still the user's own: the server's start line names that block
        # the same way (pinned in serve/test_runconfig.py), so neither line calls it a preset the other does not
        self.assertIn("sampling own: temperature=1.0",
                      setup.settings_summary({"args": [], "sampling": {**THINKING, "seed": None}}, 8080))


if __name__ == "__main__":
    unittest.main()
