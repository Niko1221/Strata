"""One drafter per run, from every way in (owner, 2026-11-02): DFlash2 alone or the MTP alone, never both.

strata_control/drafting.py decides; the app (apps/host, through Control and serve/server.py) and the Terminal (strata launch, its Drafting step, a saved preset)
both end at serve/server.py engine_args(). These tests lock in the engine arguments of every choice on the main backends, the old saved values, and the Terminal
path from the launch settings to the argument list. (tests/terminal/test_drafting.py has the table, the flags and the downloads one by one.)
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.runner import derive_config
from strata_control import drafting

NAMES = {"qwen": "qwen3.8-flash-next-iq3_s", "gyro": "qwen3.8-flash-next-gyro-gyro-s", "unsloth": "qwen3.8-flash-next-unsloth-ud-q4_k_xl",
         "coder": "qwen3.8-flash-next-coder-iq1_m", "nvfp4": "qwen3.8-flash-next-nvfp4-nvfp4", "swift": "qwen3.8-flash-next-swift-q4_k_m"}
HARDWARE = (("cuda", ["sm_120"]), ("cuda", ["sm_86"]), ("cuda", ["sm_89"]), ("hip", ["gfx1151"]), ("hip", ["gfx1201"]), ("metal", []), ("sycl", ["arc_a"]))
MTP_ONLY = ("--mtp", "--mtp-q4", "--batch-mtp", "--spec-min-p", "--mtp-draft-vocab")


def flag(args, name):
    return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) else None


def loaded(args):
    """The drafters this argument list loads: 'dflash2' for `--draft dflash2`, 'mtp' for an MTP layer folder (`--mtp <folder>` with no DFlash2)."""
    found = set()
    if flag(args, "--draft") == "dflash2":
        found.add("dflash2")
    elif "--mtp" in args:
        found.add("mtp")
    return found


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name) / "data"
        (self.data / "dflash2").mkdir(parents=True)
        (self.data / "mtp" / "rt").mkdir(parents=True)
        (self.data / "packs" / "tag").mkdir(parents=True)
        (self.data / "dflash2" / drafting.PACK_NAME).write_bytes(b"x" * (2 << 20))
        (self.data / "dflash2" / drafting.VOCAB_NAME).write_bytes(b"v" * 100)
        (self.data / "mtp" / "rt" / "experts.bin").write_bytes(b"e" * 100)
        (self.data / "mtp" / "rt" / drafting.VOCAB_NAME).write_bytes(b"v" * 100)
        self.exe = Path(self.tmp.name) / "strata-engine"
        self.exe.write_bytes(b"--mtp-draft-vocab FILE (or --mtp DIR ... --draft-fallback")   # an engine of the usual form (rc1 and later)
        self.args = ["--native", "m.gguf", "--pack", str(self.data / "packs" / "tag"), "--mtp", str(self.data / "mtp" / "rt"), "--spec", "4", "--batch-mtp",
                     "--mtp-q4", "all", "--spec-min-p", "0.5", "--max-context", "131072"]

    def cfg(self, family="qwen", **extra):
        return {"model_name": NAMES[family], "args": list(self.args), **extra}

    def apply(self, cfg, family="qwen", backend="cuda", archs=("sm_120",), **kw):
        d = drafting.apply(cfg, data_dir=self.data, family=family, backend=backend, archs=list(archs), engine=self.exe, **kw)
        return d, cfg["args"]


class EveryChoiceOnEveryBackend(Base):
    def test_auto_dflash2_and_the_unset_choice_load_dflash2_alone(self):
        for backend, archs in HARDWARE:
            for family in ("qwen", "gyro", "unsloth"):
                for mode in (None, "auto", "dflash2"):
                    cfg = self.cfg(family, **({} if mode is None else {"drafting": mode}))
                    d, args = self.apply(cfg, family, backend, archs)
                    where = (backend, archs, family, mode)
                    self.assertEqual(d.drafter, "dflash2", where)
                    self.assertEqual(loaded(args), {"dflash2"}, where)
                    self.assertEqual(flag(args, "--draft-fallback"), "none", where)
                    self.assertEqual(flag(args, "--draft-model"), str(drafting.pack_path(self.data, self.exe)), where)
                    self.assertTrue(flag(args, "--mtp-draft-vocab"), where)
                    for name in ("--mtp", "--mtp-q4", "--batch-mtp", "--spec-min-p"):
                        self.assertNotIn(name, args, (name, where))

    def test_mtp_loads_the_mtp_alone(self):
        for backend, archs in HARDWARE:
            for family in ("qwen", "gyro", "unsloth"):
                d, args = self.apply(self.cfg(family, drafting="mtp"), family, backend, archs)
                where = (backend, archs, family)
                self.assertEqual((d.drafter, loaded(args)), ("mtp", {"mtp"}), where)
                self.assertFalse([a for a in args if a.startswith("--draft")], where)
                self.assertNotIn("dflash2", " ".join(args), where)

    def test_off_loads_neither(self):
        for backend, archs in HARDWARE:
            d, args = self.apply(self.cfg(drafting="off"), "qwen", backend, archs)
            self.assertEqual((d.drafter, loaded(args)), ("off", set()), (backend, archs))
            self.assertFalse([a for a in args if a in MTP_ONLY or a.startswith("--draft")])

    def test_no_engine_with_the_code_means_the_mtp_for_auto(self):
        for backend in (None, "cpu"):
            for mode in (None, "auto", "dflash2"):
                d, args = self.apply(self.cfg(**({} if mode is None else {"drafting": mode})), "qwen", backend, ())
                self.assertEqual((d.drafter, loaded(args)), ("mtp", {"mtp"}), (backend, mode))

    def test_other_families_never_get_dflash2_whatever_is_saved(self):
        for family in ("coder", "nvfp4", "swift"):
            for mode in (None, "auto", "mtp", "dflash2"):
                cfg = self.cfg(family, **({} if mode is None else {"drafting": mode}))
                d, args = self.apply(cfg, family, "cuda", ("sm_120",))
                self.assertNotIn("--draft", args, (family, mode))
                self.assertEqual(loaded(args), {"mtp"}, (family, mode))

    def test_a_layer_split_or_several_requests_load_the_mtp_alone_for_auto_and_dflash2(self):
        for mode in ("auto", "dflash2"):
            for kw in ({"split": True}, {"slots": 4}):
                d, args = self.apply(self.cfg(drafting=mode), **kw)
                self.assertEqual((d.drafter, d.note, loaded(args)), ("mtp", drafting.NOT_HERE, {"mtp"}), (mode, kw))
                self.assertFalse([a for a in args if a.startswith("--draft")])
        for kw in ({"split": True}, {"slots": 4}):
            d, args = self.apply(self.cfg(drafting="mtp"), **kw)
            self.assertEqual((d.drafter, d.note, loaded(args)), ("mtp", "", {"mtp"}), kw)

    def test_a_missing_file_runs_the_other_drafter_alone_and_says_so(self):
        (self.data / "dflash2" / drafting.PACK_NAME).unlink()
        for mode in (None, "auto", "dflash2"):
            d, args = self.apply(self.cfg(**({} if mode is None else {"drafting": mode})))
            self.assertEqual((d.drafter, d.note, loaded(args)), ("mtp", drafting.NOT_DOWNLOADED, {"mtp"}), mode)
        (self.data / "dflash2" / drafting.PACK_NAME).write_bytes(b"x" * (2 << 20))
        (self.data / "mtp" / "rt" / "experts.bin").unlink()
        d, args = self.apply(self.cfg(drafting="mtp"))
        self.assertEqual((d.drafter, d.note, loaded(args)), ("dflash2", drafting.NO_MTP, {"dflash2"}))


class OldSavedValues(Base):
    def test_a_value_nobody_writes_any_more_is_auto(self):
        for old in ("both", "mtp+dflash2", "dflash2+mtp", "dflash", "", None, True, 1, "fastest"):
            self.assertEqual(drafting.normalize(old), "auto", old)
        for same in ("DFlash2", " dflash2 "):
            self.assertEqual(drafting.normalize(same), "dflash2", same)
        for same in ("MTP", " mtp "):
            self.assertEqual(drafting.normalize(same), "mtp", same)

    def test_every_old_value_loads_one_drafter(self):
        for old in ("both", "mtp+dflash2", "dflash2+mtp", "dflash", "", None, True, "DFlash2", "MTP"):
            d, args = self.apply(self.cfg(drafting=old))
            self.assertEqual(len(loaded(args)), 1, old)
            self.assertEqual(loaded(args), {"mtp"} if str(old).strip().lower() == "mtp" else {"dflash2"}, old)

    def test_a_saved_argument_list_that_names_both_drafters_starts_with_the_one_chosen(self):
        both = [*self.args, "--draft", "dflash2", "--draft-model", "old.pack", "--draft-fallback", "mtp", "--draft-conf", "0.5"]
        for mode, want in ((None, "dflash2"), ("auto", "dflash2"), ("dflash2", "dflash2"), ("mtp", "mtp"), ("both", "dflash2")):
            cfg = {"model_name": NAMES["qwen"], "args": list(both), **({} if mode is None else {"drafting": mode})}
            d, args = self.apply(cfg)
            self.assertEqual(loaded(args), {want}, mode)
            if want == "dflash2":
                self.assertEqual(flag(args, "--draft-fallback"), "none", mode)
                self.assertNotIn("--mtp", args, mode)
            else:
                self.assertNotIn("--draft", args, mode)
                self.assertNotIn("--draft-fallback", args, mode)

    def test_the_mtp_switch_off_with_no_choice_is_no_drafter_but_a_choice_wins(self):
        d, args = self.apply(self.cfg(mtp=False))
        self.assertEqual((d.drafter, loaded(args)), ("off", set()))
        d, args = self.apply(self.cfg(mtp=False, drafting="mtp"))
        self.assertEqual((d.drafter, loaded(args)), ("mtp", {"mtp"}))
        d, args = self.apply(self.cfg(mtp=False, drafting="dflash2"))
        self.assertEqual((d.drafter, loaded(args)), ("dflash2", {"dflash2"}))


class TheServerAndTheTerminalEndAtTheSameArguments(Base):
    """serve/server.py engine_args() is where the app's Load (through Control's saved profile), `strata launch` and the Terminal's steps all arrive."""

    def server_args(self, cfg, backend="cuda", archs=("sm_120",)):
        from serve.server import engine_args
        with mock.patch.object(drafting, "detect_hardware", return_value=(backend, list(archs))), mock.patch("serve.server.drafting_vram_mib", return_value=None), \
                mock.patch.object(drafting, "installed_engine_exe", return_value=self.exe):
            return engine_args(cfg, str(self.exe), announce=False)

    def test_each_choice_through_the_server(self):
        for backend, archs in HARDWARE:
            for mode, want in ((None, {"dflash2"}), ("auto", {"dflash2"}), ("dflash2", {"dflash2"}), ("mtp", {"mtp"}), ("off", set())):
                args = self.server_args(self.cfg(**({} if mode is None else {"drafting": mode})), backend, archs)
                self.assertEqual(loaded(args), want, (backend, archs, mode))
                if want == {"dflash2"}:
                    self.assertNotIn("--mtp", args)
                elif want == {"mtp"}:
                    self.assertFalse([a for a in args if a.startswith("--draft")])

    def test_deepseek_gets_no_drafter_arguments(self):
        cfg = {"model_name": "deepseek-v4.1-flash", "arch": "deepseek41", "args": ["--pack", "/d/ds", "--spec", "2"], "drafting": "dflash2"}
        args = self.server_args(cfg)
        self.assertEqual((loaded(args), [a for a in args if a.startswith("--draft")]), (set(), []))

    def test_the_terminal_choices_reach_the_engine_alone(self):
        base = {"exe": "component:strata-engine", "model_name": NAMES["qwen"], "port": 8080, "args": list(self.args)}
        for choice, want in ((None, {"dflash2"}), ("auto", {"dflash2"}), ("dflash2", {"dflash2"}), ("mtp", {"mtp"}), ("off", set())):
            with mock.patch.object(drafting, "detect_hardware", return_value=("cuda", ["sm_120"])):
                cfg = derive_config(base, S.LaunchSettings(drafting=choice), bool_flags=frozenset())
            args = self.server_args(cfg)
            self.assertEqual(loaded(args), want, choice)

    def test_auto_in_the_terminal_overrides_a_choice_the_app_saved_in_the_model_file(self):
        """The app writes its Drafting choice into the model's own settings file; the Terminal derives its run config from that file. Choosing Auto in the Terminal's
        Drafting step has to mean Auto, not the app's older choice (the step stores what modelstate.drafting_to_save says)."""
        from strata_cli.launch import modelstate
        for saved, key, want in (("mtp", "auto", {"dflash2"}), ("off", "auto", {"dflash2"}), ("dflash2", "mtp", {"mtp"}), ("mtp", "dflash2", {"dflash2"}),
                                 ("mtp", "off", set()), (None, "auto", {"dflash2"}), ("auto", "auto", {"dflash2"}), (None, "mtp", {"mtp"})):
            base = {"exe": "component:strata-engine", "model_name": NAMES["qwen"], "port": 8080, "args": list(self.args), **({} if saved is None else {"drafting": saved})}
            with mock.patch.object(drafting, "detect_hardware", return_value=("cuda", ["sm_120"])):
                cfg = derive_config(base, S.LaunchSettings(drafting=modelstate.drafting_to_save(key, saved)), bool_flags=frozenset())
            self.assertEqual(loaded(self.server_args(cfg)), want, (saved, key))

    def test_the_step_writes_nothing_for_auto_unless_the_file_holds_another_choice(self):
        from strata_cli.launch import modelstate
        for saved in (None, "auto", "AUTO", "both", "fastest", ""):
            self.assertIsNone(modelstate.drafting_to_save("auto", saved) if drafting.normalize(saved) == "auto" else None, saved)
        for saved in ("mtp", "dflash2", "off", "MTP"):
            self.assertEqual(modelstate.drafting_to_save("auto", saved), "auto", saved)
        for key in ("mtp", "dflash2", "off"):
            self.assertEqual(modelstate.drafting_to_save(key, "mtp"), key)


if __name__ == "__main__":
    unittest.main()
