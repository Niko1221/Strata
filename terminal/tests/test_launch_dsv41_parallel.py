"""DeepSeek-V4.1-Flash `strata launch --parallel N`: offered, off by default.

No flag is exactly what it was (one request at a time, nothing about parallel in the plan, no override in the profile).  With the flag the number
reaches the engine's command line through the profile and the server, `--plan` says in plain words what it does with the measured numbers, a pick
beyond what was run (Windows, a Mac, a small card) is a warning and never a refusal, a DeepSeek engine from before the flag is asked and the
launch goes on one at a time, the saved preset and the saved profile keep the number, and the engine gets the restore-point setting the
server gives it.  Nothing is downloaded or started.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch import store
from strata_cli.launch.catalog import Catalog
from strata_cli.launch.runner import Runner
from strata_control import dsv41_flags, dsv41_parallel
from tests.terminal.launch_support import HW_12, Home, catalog, run_cli

R9700 = {"name": "AMD Radeon AI PRO R9700", "vram_gb": 32.0, "usable": True, "vendor": "amd", "arch": "gfx1201"}
HW_4X32 = {**HW_12, "ram_gb": 251.0, "disk_free_gb": 900.0, "engine_vendor": "amd", "gpus": [{**R9700, "index": i} for i in range(4)]}
A6000 = {"name": "NVIDIA RTX A6000", "vram_gb": 48.0, "usable": True, "vendor": "nvidia"}
HW_2XA6000 = {**HW_12, "ram_gb": 125.0, "disk_free_gb": 900.0, "engine_vendor": "nvidia", "gpus": [{**A6000, "index": i} for i in range(2)]}
LINUX = ("linux", "x86_64")


def flat(text: str) -> str:
    """A wrapped plan paragraph on one line."""
    return re.sub(r"\s+", " ", text)


class Base(unittest.TestCase):
    hw = HW_2XA6000
    host = LINUX

    def setUp(self):
        type(self).hw, type(self).host = HW_2XA6000, LINUX
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        for p in (mock.patch.object(Catalog, "hardware", lambda s, fresh=False: dict(type(self).hw)),
                  mock.patch("strata_cli.launch.cmd._host_platform", lambda: type(self).host)):
            p.start()
            self.addCleanup(p.stop)

    def plan(self, *args):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", *args])
        self.assertEqual(code, 0, err + out)
        return json.loads(out)["plan"]

    def human(self, *args):
        code, out, err = run_cli(["launch", "--plan", "--no-tui", *args])
        self.assertEqual(code, 0, err + out)
        return out


class NoFlagIsWhatItWas(Base):
    def test_the_plan_has_nothing_about_parallel(self):
        p = self.plan("--model", "ist-das-2.00bpw")
        self.assertIsNone(p["settings"]["parallel"])
        self.assertNotIn("parallel", p)                              # no new key in the plan
        self.assertEqual([w for w in p["warnings"] if "parallel" in w.lower()], [])
        out = self.human("--model", "ist-das-2.00bpw")
        self.assertIn("parallel one at a time", out)
        self.assertNotIn("conversations", out)
        self.assertNotIn("--parallel", out.split("same as")[-1])

    def test_one_is_one_at_a_time(self):
        p = self.plan("--model", "ist-das-2.00bpw", "--parallel", "1")
        self.assertNotIn("parallel", p)
        self.assertIn("parallel one at a time", self.human("--model", "ist-das-2.00bpw", "--parallel", "1"))

    def test_nothing_is_asked_of_the_engine_and_no_override_is_made(self):
        s = S.LaunchSettings(family="deepseek41", model="IST-DAS-2.00bpw", context=8192, tune="skip")
        r = Runner(s, catalog=catalog(self.hw))
        self.assertIsNone(r.catalog_parallel())
        self.assertEqual(S.flags_text(s).count("--parallel"), 0)


class AskedFor(Base):
    def test_the_number_reaches_the_plan_and_is_not_dropped(self):
        p = self.plan("--model", "ist-das-2.25bpw", "--parallel", "4", "--context", "32k")
        self.assertEqual(p["settings"]["parallel"], 4)
        self.assertEqual(p["parallel"]["conversations"], 4)
        self.assertEqual(p["parallel"]["requested"], 4)
        text = " ".join(p["parallel"]["lines"])
        self.assertIn("4 conversations are answered at the same time", text)
        self.assertIn("73 tokens a second", text)                    # numbers, with what they were measured on
        self.assertIn("RTX A6000", text)
        self.assertIn("graphics memory", text)
        self.assertNotIn("is not used", " ".join(p["warnings"]))
        self.assertIn("--parallel 4", p["flags"])

    def test_the_human_plan_says_what_it_does(self):
        out = flat(self.human("--model", "ist-das-2.25bpw", "--parallel", "3", "--context", "32k"))
        self.assertIn("parallel 3 conversations at the same time", out)
        self.assertIn("One answer is slower while the others run", out)
        self.assertIn("without --parallel requests take turns", out)
        self.assertIn("Measured on", out)

    def test_no_verdict_words(self):
        text = " ".join(dsv41_parallel.explain(4, 32768)).lower()
        for word in ("better", "worse", "recommend", "worth", "best", "should", "great", "poor"):
            self.assertNotIn(word, text)

    def test_more_than_eight_is_the_engines_limit_not_a_refusal(self):
        self.assertEqual(dsv41_parallel.clamp(12), 8)
        self.assertIsNone(dsv41_parallel.clamp(1))
        self.assertIsNone(dsv41_parallel.clamp(None))
        w = dsv41_parallel.warnings(12, cards=[A6000], os_name="linux", machine="x86_64", vendor="nvidia")
        self.assertTrue(any("will use 8" in what for what, _ in w))

    def test_a_linux_run_on_the_cards_it_was_run_on_has_no_warning(self):
        p = self.plan("--model", "ist-das-2.25bpw", "--parallel", "2")
        self.assertEqual([w for w in p["warnings"] if "--parallel" in w], [])
        type(self).hw = HW_4X32
        p = self.plan("--model", "ist-das-2.25bpw", "--parallel", "4")
        self.assertEqual([w for w in p["warnings"] if "--parallel" in w], [])


class AtLaunch(Base):
    def _launch(self, *args):
        from strata_cli.launch.runner import Result
        fake = Result(profile="launch-x", urls=["http://127.0.0.1:8080"], key=None, tune_word=None, notices=[], config="/x")
        with mock.patch.object(Runner, "run", return_value=fake):
            code, out, err = run_cli(["launch", "--no-tui", "--family", "deepseek41", "--model", "ist-das-2.25bpw", "--context", "32k", *args])
        self.assertEqual(code, 0, err + out)
        return flat(out)

    def test_the_launch_says_what_it_does_when_asked(self):
        out = self._launch("--parallel", "4")
        self.assertIn("4 conversations are answered at the same time", out)
        self.assertIn("73 tokens a second", out)
        self.assertIn("Strata is serving.", out)

    def test_the_launch_without_the_flag_says_nothing_about_it(self):
        out = self._launch()
        self.assertNotIn("conversations are answered", out)
        self.assertNotIn("parallel", out.lower().replace("--parallel", "").replace("(parallel", ""))

    def test_a_risky_pick_is_said_and_the_launch_goes_on(self):
        type(self).host = ("win32", "AMD64")
        out = self._launch("--parallel", "2")
        self.assertIn("run on Linux only", out)
        self.assertIn("Strata is serving.", out)


class RiskyPicksWarnNeverRefuse(Base):
    def test_windows_is_a_warning(self):
        type(self).host = ("win32", "AMD64")
        p = self.plan("--model", "ist-das-2.25bpw", "--parallel", "4")
        self.assertEqual(p["settings"]["parallel"], 4)               # the number stays: nothing was refused
        self.assertTrue(any("run on Linux only" in w for w in p["warnings"]), p["warnings"])

    def test_a_mac_is_a_warning(self):
        w = dsv41_parallel.warnings(2, cards=[{"name": "Apple M5 Max", "vram_gb": 96.0, "vendor": "apple", "unified_gb": 128.0}],
                                    os_name="darwin", machine="arm64", vendor="apple")
        self.assertEqual(len(w), 1)
        self.assertIn("not run on a Mac (Metal)", w[0][0])
        self.assertIn("without --parallel", w[0][1])

    def test_a_small_card_is_a_warning_with_the_cost(self):
        type(self).hw = HW_12
        p = self.plan("--model", "ist-das-2.00bpw", "--parallel", "4", "--context", "16k")
        self.assertEqual(p["settings"]["parallel"], 4)
        small = [w for w in p["warnings"] if "12 GB" in w]
        self.assertTrue(small, p["warnings"])
        self.assertIn("runs fewer conversations", small[0])

    def test_an_unknown_amd_card_is_a_warning(self):
        w = dsv41_parallel.warnings(2, cards=[{"name": "AMD Radeon RX 7900 XTX", "vram_gb": 24.0, "vendor": "amd", "arch": "gfx1100"}],
                                    os_name="linux", machine="x86_64", vendor="amd")
        self.assertTrue(any("gfx1100" in what for what, _ in w))

    def test_a_shared_memory_pc_is_a_warning(self):
        w = dsv41_parallel.warnings(2, cards=[{"name": "AMD Radeon 8060S", "vram_gb": 96.0, "vendor": "amd", "uma": True}],
                                    os_name="linux", machine="x86_64", vendor="amd")
        self.assertTrue(any("shared memory" in what for what, _ in w))

    def test_a_dgx_spark_has_no_shared_memory_warning(self):
        w = dsv41_parallel.warnings(2, cards=[{"name": "NVIDIA GB10", "vram_gb": 120.0, "vendor": "nvidia", "uma": True}],
                                    os_name="linux", machine="aarch64", vendor="nvidia")
        self.assertEqual(w, [])


class TheEngineInUse(Base):
    def _cfg(self):
        return {"arch": "deepseek41", "exe": "/x/engine-dsv41/0.1.55-linux/strata-dsv41", "args": ["--pack", "/p"], "parallel": 4}

    def _runner(self, parallel):
        s = S.LaunchSettings(family="deepseek41", model="IST-DAS-2.25bpw", context=8192, tune="skip", parallel=parallel)
        return s, Runner(s, catalog=catalog(self.hw))

    def test_an_engine_from_before_the_flag_is_asked_and_the_launch_goes_on_one_at_a_time(self):
        s, r = self._runner(4)
        with mock.patch.object(dsv41_flags, "takes_parallel", return_value=False):
            r.gate_parallel(self._cfg())
        self.assertTrue(r.parallel_dropped)
        self.assertTrue(any("is from before --parallel" in n for n in r.notices), r.notices)
        self.assertEqual(s.parallel, 4)                              # the person's choice stays in the preset

    def test_a_current_engine_or_an_engine_that_cannot_be_asked_goes_on(self):
        for answer in (True, None):
            _s, r = self._runner(4)
            with mock.patch.object(dsv41_flags, "takes_parallel", return_value=answer):
                r.gate_parallel(self._cfg())
            self.assertFalse(r.parallel_dropped, answer)
            self.assertEqual(r.notices, [])

    def test_without_the_flag_the_engine_is_not_even_asked(self):
        _s, r = self._runner(None)
        with mock.patch.object(dsv41_flags, "takes_parallel", side_effect=AssertionError("asked")):
            r.gate_parallel(self._cfg())
        self.assertFalse(r.parallel_dropped)

    def test_takes_parallel_reads_the_usage_text(self):
        usage_old = "maya-ds4 --pack <model>/maya-ds4 [--gpus 0,1] (--ids I,J | --serve | --pack-info)\n"
        usage_new = usage_old.replace("(--ids", "[--parallel N] (--ids")
        for n, (text, want) in enumerate(((usage_old, False), (usage_new, True), ("segfault", None))):
            exe = Path(self.home.base) / f"e{n}" / "strata-dsv41"
            exe.parent.mkdir()
            exe.write_text("x" * (n + 1), encoding="utf-8")
            with mock.patch("subprocess.run", return_value=mock.Mock(stdout="", stderr=text, returncode=0)) as run:
                self.assertIs(dsv41_flags.takes_parallel(exe), want, text)
                self.assertIs(dsv41_flags.takes_parallel(exe), want, text)
                self.assertEqual(run.call_count, 1)                  # asked once per file
                self.assertEqual(run.call_args[0][0], [str(exe), "--help"])
        self.assertIsNone(dsv41_flags.takes_parallel(Path(self.home.base) / "no" / "such" / "strata-dsv41"))

    def test_the_server_leaves_the_flag_out_for_an_engine_from_before_it(self):
        from serve.server import parallel_args
        cfg = {"arch": "deepseek41", "exe": "/x/strata-dsv41", "parallel": 4}
        with mock.patch.object(dsv41_flags, "takes_parallel", return_value=False), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(parallel_args(cfg, []), [])
        self.assertIn("from before --parallel", out.getvalue())
        for answer in (True, None):
            with mock.patch.object(dsv41_flags, "takes_parallel", return_value=answer):
                self.assertEqual(parallel_args(cfg, []), ["--parallel", "4"])
        with mock.patch.object(dsv41_flags, "takes_parallel", return_value=True), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(parallel_args({**cfg, "parallel": 12}, []), ["--parallel", "8"])


class SavedAndRestarted(Base):
    def test_a_saved_preset_keeps_the_number(self):
        store.save(S.LaunchSettings(name="Agents", family="deepseek41", model="IST-DAS-2.25bpw", context=32768, parallel=4, tune="skip"))
        p = self.plan("--preset", "Agents")
        self.assertEqual(p["settings"]["parallel"], 4)
        self.assertEqual(p["parallel"]["conversations"], 4)
        p2 = self.plan("--preset", "Agents", "--parallel", "2")      # a flag on top changes this start only
        self.assertEqual(p2["settings"]["parallel"], 2)
        self.assertEqual(store.load("Agents").parallel, 4)

    def test_a_preset_without_it_is_one_at_a_time(self):
        store.save(S.LaunchSettings(name="Plain", family="deepseek41", model="IST-DAS-2.25bpw", context=32768, tune="skip"))
        p = self.plan("--preset", "Plain")
        self.assertIsNone(p["settings"]["parallel"])
        self.assertNotIn("parallel", p)

    def test_the_profile_override_becomes_the_engines_flag_after_a_restart(self):
        """launch writes parallel=N into the profile's overrides; `strata serve --profile` (and so a restart of the saved config) turns it into
        the config's "parallel" key and the engine's `--parallel N`; absent, nothing."""
        from serve.server import engine_args
        from strata_cli import cli
        cfg = {"arch": "deepseek41", "exe": "/x/strata-dsv41", "args": ["--pack", "/p", "--max-context", "8192"], "model_name": "m", "log": "/x/e.log"}
        for overrides, want in (({"host": "127.0.0.1", "port": 8080, "parallel": 4}, ["--parallel", "4"]), ({"host": "127.0.0.1", "port": 8080}, None)):
            values, _origins = cli._effective(cfg, {"overrides": overrides}, None)
            self.assertEqual(values["parallel"], overrides.get("parallel"))
            eff = {**cfg, "args": list(cfg["args"])}
            if values["parallel"] is not None:
                eff["parallel"] = values["parallel"]
            with contextlib.redirect_stdout(io.StringIO()):
                argv = engine_args(eff)
            if want:
                self.assertEqual(argv[argv.index("--parallel"):argv.index("--parallel") + 2], want)
            else:
                self.assertNotIn("--parallel", argv)

    def test_the_profile_gets_the_override_only_when_asked(self):
        from strata_control import profiles
        cfgfile = Path(self.home.base) / "cfg.json"
        cfgfile.write_text(json.dumps({"arch": "deepseek41", "exe": "/x/strata-dsv41", "tokenizer": "/p/tokenizer", "args": ["--pack", "/p"], "model_name": "m"}), encoding="utf-8")
        for asked, want in ((4, 4), (None, None)):
            s = S.LaunchSettings(family="deepseek41", model="IST-DAS-2.25bpw", context=8192, tune="skip", parallel=asked)
            r = Runner(s, catalog=catalog(self.hw))
            r.ensure_profile(cfgfile)
            prof, _rev = profiles.load(r.profile)
            self.assertEqual((prof.get("overrides") or {}).get("parallel"), want)

    def test_an_old_engine_leaves_the_override_out_of_the_profile(self):
        from strata_control import profiles
        cfgfile = Path(self.home.base) / "cfg.json"
        cfgfile.write_text(json.dumps({"arch": "deepseek41", "exe": "/x/strata-dsv41", "tokenizer": "/p/tokenizer", "args": ["--pack", "/p"], "model_name": "m"}), encoding="utf-8")
        s = S.LaunchSettings(family="deepseek41", model="IST-DAS-2.25bpw", context=8192, tune="skip", parallel=4)
        r = Runner(s, catalog=catalog(self.hw))
        r.parallel_dropped = True
        r.ensure_profile(cfgfile)
        prof, _rev = profiles.load(r.profile)
        self.assertNotIn("parallel", prof.get("overrides") or {})


class RestorePointsSetting(unittest.TestCase):
    def test_the_engine_gets_the_servers_restore_point_setting_with_parallel(self):
        """The launcher starts the engine through serve/server.py, which gives a DeepSeek engine STRATA_DS4_SLOT_KEEP=1 and a slot folder next to the log;
        a launch with --parallel gets the same environment as one without."""
        from serve.server import child_env, engine_args
        base = {"arch": "deepseek41", "exe": "/x/strata-dsv41", "args": ["--pack", "/p"], "log": "/data/launch/m.engine.log"}
        for extra in ({}, {"parallel": 4}):
            env = child_env({**base, **extra})
            self.assertEqual(env["STRATA_DS4_SLOT_KEEP"], "1", extra)
            self.assertTrue(env["STRATA_DS4_SLOT_DIR"].replace("\\", "/").endswith("/data/launch/ds4-slots"), env["STRATA_DS4_SLOT_DIR"])
        with contextlib.redirect_stdout(io.StringIO()):
            a, b = engine_args(base), engine_args({**base, "parallel": 4})
        self.assertIn("--parallel", b)
        self.assertNotIn("--parallel", a)
        self.assertEqual([x for x in b if x not in ("--parallel", "4")], a)      # nothing else moves with the flag


if __name__ == "__main__":
    unittest.main()
