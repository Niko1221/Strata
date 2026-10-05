"""launcher/test_api.py - what the launcher answers, with no GPU, no download and no model loaded.

    python -m unittest launcher.test_api -v

The Strata folder is the real one (its setup.py supplies the model tables), but everything that could touch this PC
is pinned or faked: the hardware check returns a fixed description of a PC, no process is tracked, nothing answers on
any port, and the launcher's own state (presets, jobs) lives in a temporary folder.  What is checked is the
launcher's part: that a preset is stored and comes back, that a plan carries the preset's flags and only once each,
that starting or measuring something that is not there says so plainly, and that the model card says what the model
here really runs with.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from launcher.api import Launcher  # noqa: E402
from launcher import app as launcher_app  # noqa: E402
from launcher.jobs import Job  # noqa: E402

HW = {"os": "Windows Test", "ram_gb": 64.0, "cpu": {"name": "a Ryzen", "cores": 8, "avx2": True, "avx512": True},
      "gpus": [{"index": 0, "name": "Test GPU", "vendor": "nvidia", "vram_gb": 12.0, "arch": 120,
                "usable": True, "problem": None}], "disk_free_gb": 900.0}
PRESET = {"name": "IQ3_S 128K", "family": "qwen", "model": "IQ3_S", "context": 131072, "vision": "no",
          "kv": "int8", "vram_reserve_mib": 1500, "speed_projection": "on", "calibrate": "never"}


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.a = Launcher(ROOT, state_dir=Path(self.dir.name) / "state")
        import time
        self.a.s._hw_cache = (time.time(), dict(HW))            # the hardware check: a fixed PC
        self.a.s.probe = lambda port, deep=True: None           # nothing answers on any port
        self.a.s.configs = lambda: []                           # nothing is installed in this folder
        self.a.s.state = lambda name: {}                        # no install or server was started by this controller
        self.a.s.save_state = lambda name, d: None

    def tearDown(self):
        self.a.calibrate_job._save(None)
        self.dir.cleanup()


class Presets(Base):
    def test_a_preset_is_stored_and_comes_back(self):
        saved = self.a.save_preset(PRESET)
        again = Launcher(ROOT, state_dir=self.a.state_dir)      # a new process reads the same file
        self.assertEqual([p["id"] for p in again.presets.load()], [saved["id"]])
        self.assertEqual(again.preset(saved["id"])["vram_reserve_mib"], 1500)

    def test_a_preset_the_model_page_would_refuse_is_refused_with_the_reason(self):
        with self.assertRaises(ValueError) as e:
            self.a.save_preset(dict(PRESET, family="coder", model="IQ3_S"))
        self.assertIn("has no IQ3_S", str(e.exception))

    def test_a_fine_tune_size_saves_though_setup_lists_it_without_families(self):
        # Swift 1.5's IQ2_XS and IQ3_XXS carry no "families" key in setup's table: the page lists them under Swift,
        # so saving one of them has to be accepted by the same rule, not refused with an empty list of sizes
        fam = next(f for f in self.a.ui_catalog()["families"] if f["id"] == "swift")
        self.assertIn("IQ3_XXS", fam["sizes"])
        self.assertIn("IQ3_XXS", self.a.catalog()["family_sizes"]["swift"])
        saved = self.a.save_preset(dict(PRESET, name="Swift 3-bit", family="swift", model="IQ3_XXS"))
        self.assertEqual((saved["family"], saved["model"]), ("swift", "IQ3_XXS"))
        self.assertEqual(self.a.preset(saved["id"])["model"], "IQ3_XXS")
        args = self.a.plan(saved["id"])["setup_args"]
        self.assertEqual(args[args.index("--family") + 1], "swift")
        self.assertEqual(args[args.index("--model") + 1], "IQ3_XXS")

    def test_the_page_gets_the_form_and_the_tables_from_the_same_source(self):
        schema = self.a.schema()
        keys = [f["key"] for f in schema["fields"]]
        self.assertIn("context", keys)
        self.assertIn("calibrate", keys)
        cat = self.a.ui_catalog()
        self.assertEqual([f["id"] for f in cat["families"]], ["qwen", "swift", "coder", "unsloth"])
        self.assertTrue(cat["sizeInfo"]["IQ3_S"]["download_gb"] > 0)
        self.assertEqual(cat["sizeInfo"]["IQ3_S"]["on_this_pc"], "fits")

    def test_the_built_in_presets_are_not_stored(self):
        state = self.a.state()
        self.assertEqual(state["presets"], [])
        self.assertTrue(state["builtin"])
        self.assertTrue(all(p["id"].startswith("builtin-") for p in state["builtin"]))
        self.assertTrue(any(p["name"] == "Recommended for this PC" for p in state["builtin"]))
        self.assertEqual(self.a.presets.load(), [])

    def test_a_preset_can_be_made_from_an_installed_model(self):
        cfg = {"exe": "x", "args": ["--max-context", "131072", "--kv", "int8", "--pack", "p"],
               "model_name": "qwen3.8-flash-next-iq3_s", "port": 8080}
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "strata-iq3_s.json").write_text(json.dumps(cfg), encoding="utf-8")
            b = Launcher(ROOT, state_dir=Path(self.dir.name) / "state2")
            import time
            b.s._hw_cache = (time.time(), dict(HW))
            b.s.root = Path(root)                              # the configs and the data folder: another install
            p = b.preset_from_model("iq3_s")
            self.assertEqual((p["family"], p["model"], p["context"]), ("qwen", "IQ3_S", 131072))
            self.assertEqual(p["kv"], "int8")

    def test_the_setup_this_folder_has_is_a_preset_of_its_own(self):
        # the page must show what the installed model actually runs with, not only what setup would recommend
        cfg = {"exe": "x", "args": ["--max-context", "262144", "--kv", "int8", "--pack", "p"],
               "model_name": "qwen3.8-flash-next-iq3_s", "port": 8080}
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "strata-iq3_s.json").write_text(json.dumps(cfg), encoding="utf-8")
            b = Launcher(ROOT, state_dir=Path(self.dir.name) / "state3")
            import time
            b.s._hw_cache = (time.time(), dict(HW))
            b.s.root = Path(root)
            listed = [p for p in b.state()["builtin"] if p["id"] == "builtin-installed-iq3_s"]
            self.assertEqual(len(listed), 1, "the installed model is not in the list")
            self.assertEqual(listed[0]["context"], 262144)
            self.assertIn("262144", b.plan("builtin-installed-iq3_s")["setup_command"])
            self.assertEqual(b.preset("builtin-installed-iq3_s")["context"], 262144)
            self.assertEqual(b.presets.load(), [])            # still nothing stored


class Plans(Base):
    def test_the_plan_is_what_setup_would_be_told(self):
        p = self.a.save_preset(PRESET)
        plan = self.a.plan(p["id"])
        self.assertEqual(plan["model_id"], "iq3_s")
        self.assertEqual(plan["setup_args"].count("--experimental-speed-projection"), 1)
        self.assertEqual(plan["setup_args"][plan["setup_args"].index("--experimental-speed-projection") + 1], "on")
        self.assertEqual(plan["setup_args"][plan["setup_args"].index("--vram-reserve-mib") + 1], "1500")
        self.assertEqual(plan["setup_args"].count("--context"), 1)
        self.assertTrue(plan["setup_command"].startswith("START-HERE.bat") or
                        plan["setup_command"].startswith("./setup.sh"))
        self.assertTrue(plan["disk_needed_gb"] > 0)
        self.assertTrue(plan["ok"], plan["why"])                # the fake PC has 64 GB, enough for this size

    def test_a_plan_says_what_stops_it(self):
        # a disk too small for the download: the plan says so, and the download button is not offered. The data
        # folder is pointed at an empty temp folder so this does not depend on what this PC already has.
        self.a.s.data_dir = lambda: Path(self.dir.name) / "no-models-here"
        self.a.s.disk_free = lambda path: 10.0
        p = self.a.save_preset(PRESET)
        plan = self.a.plan(p["id"])
        self.assertFalse(plan["ok"])
        self.assertIn("not enough free disk space", plan["why"])
        with self.assertRaises(Exception) as e:
            self.a.install(p["id"])
        self.assertIn("not enough free disk space", str(e.exception))

    def test_a_preset_that_does_not_fit_says_so_before_downloading(self):
        p = self.a.save_preset(dict(PRESET, name="IQ3_S on 32 GB", model="IQ3_S"))
        import time
        self.a.s._hw_cache = (time.time(), dict(HW, ram_gb=32.0))
        plan = self.a.plan(p["id"])
        self.assertIn("needs ~62 GB", plan["ram"])

    def test_a_plan_says_what_it_would_change_on_a_model_that_is_already_installed(self):
        # an install rewrites the model's own strata-<model>.json: a preset with a smaller context must not be
        # allowed to lower a 256K setup quietly
        cfg = {"exe": "x", "args": ["--max-context", "262144", "--kv", "int8", "--kv-resident", "32768",
                                    "--pcie-frac", "0.29", "--pack", "p"],
               "model_name": "qwen3.8-flash-next-iq3_s", "port": 8080}
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "strata-iq3_s.json").write_text(json.dumps(cfg), encoding="utf-8")
            b = Launcher(ROOT, state_dir=Path(self.dir.name) / "state4")
            import time
            b.s._hw_cache = (time.time(), dict(HW))
            b.s.root = Path(root)
            p = b.save_preset(PRESET)                                  # same size, 128K context
            changes = b.plan(p["id"])["changes"]
            ctx = [c for c in changes if c["key"] == "context"]
            self.assertEqual([(c["config"], c["preset"]) for c in ctx], [("262144", "131072")], changes)
            self.assertTrue(any(c["key"] == "kv_streaming" for c in changes), changes)
            same = b.save_preset(dict(PRESET, name="as installed", context=262144, kv_streaming="on",
                                      vram_reserve_mib="", speed_projection=""))
            self.assertEqual(b.plan(same["id"])["changes"], [], "a preset that matches changes nothing")

    def test_the_page_sees_that_start_runs_the_config_and_not_the_preset(self):
        cfg = {"exe": "x", "args": ["--max-context", "262144", "--kv", "int8", "--control-vector-scaled", "v.gguf:1.0",
                                    "--pack", "p"],
               "model_name": "qwen3.8-flash-next-iq3_s", "port": 8080}
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "strata-iq3_s.json").write_text(json.dumps(cfg), encoding="utf-8")
            b = Launcher(ROOT, state_dir=Path(self.dir.name) / "state5")
            import time
            b.s._hw_cache = (time.time(), dict(HW))
            b.s.root = Path(root)
            off = b.preset_diff(dict(PRESET, context=262144, speed_projection="off"))
            self.assertEqual(off["model_id"], "iq3_s")
            self.assertTrue(off["installed"])
            esp = [c for c in off["changes"] if c["key"] == "speed_projection"]
            self.assertEqual([(c["config"], c["preset"]) for c in esp], [("on", "off")], off["changes"])
            other = b.preset_diff({"family": "qwen", "model": "Q2_0"})       # a size nothing is installed for
            self.assertFalse(other["installed"])
            self.assertEqual(other["changes"], [])

    def test_the_diff_resolves_auto_with_setup_s_own_rules(self):
        # "auto" is not a setting: it is setup's RAM rule for this PC.  A preset that says what the rule decides is
        # the model as it runs (no warning, and Start does not run setup); one that says something else is a real
        # difference and says what it changes.  The rules are setup's, so the two cannot drift.
        cfg = {"exe": "x", "args": ["--max-context", "196608", "--kv", "int8", "--kv-resident", "32768",
                                    "--pack", "p"], "model_name": "swift-1.5-iq3_xxs", "port": 8080, "gpu": 0}
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "strata-swift-iq3_xxs.json").write_text(json.dumps(cfg), encoding="utf-8")
            b = Launcher(ROOT, state_dir=Path(self.dir.name) / "state7")
            import time
            b.s._hw_cache = (time.time(), dict(HW))
            b.s.root = Path(root)
            b.s._setup, b.s._setup_tried = self.a.s.setup_module(), True      # the rules are setup.py's
            base = dict(PRESET, name="Swift 1.5 IQ3_XXS", family="swift", model="IQ3_XXS", context=196608, gpu=0,
                        vram_reserve_mib="", speed_projection="")

            def keys(over):
                return sorted(c["key"] for c in b.preset_diff(dict(base, **over))["changes"])

            # 64 GB of RAM: at 192K IQ3_XXS streams its KV (60 + 2.7 + 1) and does not need the low-RAM mode
            self.assertEqual(keys({"kv_streaming": "auto", "low_ram": "off"}), [],
                             "spelling out what setup decides here is the model as it runs")
            self.assertEqual(keys({"kv_streaming": "on", "low_ram": "auto"}), [])
            self.assertEqual(keys({"kv_streaming": "off"}), ["kv_streaming"], "the model streams: off is a change")
            self.assertEqual(keys({"low_ram": "on"}), ["low_ram"], "auto here is off: on is a change")
            self.assertEqual(keys({"context": 65536}), ["context"], "64K streams too: only the length differs")
            self.assertEqual(keys({"context": 32768}), ["context", "kv_streaming"], "32K is not streamed")

    def test_a_k8v4_preset_is_read_with_setup_s_rule_like_every_other_kv(self):
        # 0.1.40 lets k8v4 stream its KV (#711), so the launcher keeps no k8v4 case of its own: on a PC with room,
        # a config carrying --kv-resident and a preset saying "auto" are the same setting.  This is the guard
        # against re-introducing the old "k8v4 never streams" special case, which is now a lie about setup.
        cfg = {"exe": "x", "args": ["--max-context", "196608", "--kv", "k8v4", "--kv-resident", "32768",
                                    "--pack", "p"], "model_name": "swift-1.5-iq3_xxs", "port": 8080, "gpu": 0}
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "strata-swift-iq3_xxs.json").write_text(json.dumps(cfg), encoding="utf-8")
            b = Launcher(ROOT, state_dir=Path(self.dir.name) / "state-k8v4")
            import time
            b.s._hw_cache = (time.time(), dict(HW, ram_gb=128.0))
            b.s.root = Path(root)
            b.s._setup, b.s._setup_tried = self.a.s.setup_module(), True
            base = dict(PRESET, name="Swift 1.5 IQ3_XXS k8v4", family="swift", model="IQ3_XXS", context=196608,
                        kv="k8v4", gpu=0, vram_reserve_mib="", speed_projection="")
            import setup as S
            self.assertTrue(S.kv_streaming_wanted("IQ3_XXS", 196608, "k8v4", 128.0),
                            "0.1.40: k8v4 streams on a PC with room for its 2.1 GB cache at 192K")

            def keys(over):
                return sorted(c["key"] for c in b.preset_diff(dict(base, **over))["changes"])

            self.assertEqual(keys({"kv_streaming": "auto"}), [], "auto is what setup writes here: no change")
            self.assertEqual(keys({"kv_streaming": "on"}), [])
            self.assertEqual(keys({"kv_streaming": "off"}), ["kv_streaming"], "off is a real ask")

    def test_a_preset_with_no_disk_room_is_refused_before_it_starts(self):
        p = self.a.save_preset(dict(PRESET, name="no room"))
        empty = Path(self.dir.name) / "empty-data"
        empty.mkdir()
        self.a.s.data_dir = lambda: empty                       # none of it is downloaded yet
        self.a.s.disk_free = staticmethod(lambda path: 10.0)    # and the drive has 10 GB
        with self.assertRaises(Exception) as e:
            self.a.install(p["id"])
        self.assertIn("not enough free disk space", str(e.exception))

    def test_the_page_offers_the_extra_length_and_a_value_of_your_own(self):
        cats = self.a.ui_catalog()["contexts"]
        self.assertIn(196608, cats)                       # 192K, between setup's 128K and 256K
        self.assertIn(131072, cats)
        self.assertEqual(cats, sorted(cats))
        p = self.a.save_preset(dict(PRESET, name="IQ3_S 192K", context=196608))
        plan = self.a.plan(p["id"])
        self.assertTrue(plan["own_context"], "192K is not in setup's own menu, the plan says so")
        self.assertEqual(plan["setup_args"][plan["setup_args"].index("--context") + 1], "196608")
        listed = self.a.save_preset(dict(PRESET, name="IQ3_S 128K", context=131072))
        self.assertFalse(self.a.plan(listed["id"])["own_context"])

    def test_starting_a_model_applies_a_preset_that_differs_from_its_config(self):
        # a model runs what its own strata-<model>.json says, so applying the preset is part of starting it
        cfg = {"exe": "x", "args": ["--max-context", "262144", "--kv", "int8", "--pack", "p"],
               "model_name": "qwen3.8-flash-next-iq3_s", "port": 8080}
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "strata-iq3_s.json").write_text(json.dumps(cfg), encoding="utf-8")
            b = Launcher(ROOT, state_dir=Path(self.dir.name) / "state6")
            import time
            b.s._hw_cache = (time.time(), dict(HW))
            b.s.root = Path(root)
            applied, started = [], []
            b.apply_preset = lambda p: applied.append(p["id"]) or {"install": {"result": "installed"}}
            b.t.strata_start = lambda **kw: started.append(kw) or {"summary": "Strata is running", "server": {}}
            p = b.save_preset(PRESET)                                  # 128K over a 262K config
            out = b.start(p["id"])
            self.assertEqual(applied, [p["id"]], "the preset was not applied before the start")
            self.assertIn("applied the preset to iq3_s first", out["summary"])
            self.assertIn("Context 131072", out["summary"])
            self.assertEqual(len(started), 1)
            same = b.save_preset(dict(PRESET, name="as installed", context=262144, vram_reserve_mib="",
                                      speed_projection=""))
            out2 = b.start(same["id"])
            self.assertEqual(applied, [p["id"]], "a preset that matches must not run setup again")
            self.assertNotIn("applied", out2)
            b.start(p["id"], apply_first=False)                        # a plain start stays possible
            self.assertEqual(applied, [p["id"]])
            spelled = b.save_preset(dict(PRESET, name="spelled defaults", context=262144, vram_reserve_mib="",
                                         speed_projection="off", kv_streaming="auto", low_ram="auto"))
            self.assertEqual(b.plan(spelled["id"])["changes"], [],
                             "saying what setup would do anyway is not a difference")
            b.start(spelled["id"])
            self.assertEqual(len(applied), 1, "a preset that only spells out defaults must not run setup")
            quiet = b.apply_by_id(same["id"])
            self.assertEqual(quiet["applied"], [])
            self.assertIn("already runs with these settings", quiet["summary"])
            r = b.apply_by_id(p["id"])
            self.assertEqual([c["key"] for c in r["applied"]], ["context", "vram_reserve_mib", "speed_projection"],
                             r["applied"])
            self.assertIn("applied to iq3_s", r["summary"])


class Running(Base):
    def test_starting_a_model_that_is_not_installed_says_so(self):
        p = self.a.save_preset(PRESET)
        with self.assertRaises(Exception) as e:
            self.a.start(p["id"])
        self.assertIn("is not installed here", str(e.exception))

    def test_measuring_refuses_while_a_model_runs(self):
        self.a.s.probe = lambda port, deep=True: {"port": port, "model": "m", "loaded": True}
        with self.assertRaises(Exception) as e:
            self.a.calibrate("anything")
        self.assertIn("stop it first", str(e.exception))

    def test_a_measurement_asked_for_before_a_start_ends_with_that_start(self):
        # the page's answer to a preset's "ask": measure, then start. Only the wiring is tested here: no
        # measurement is run.
        self.a._save_hook({"model": "iq3_s", "start": True})
        self.a.calibrate_job.status = lambda *a, **kw: {"running": False, "result": {"ok": True}}
        started = []
        self.a.t.strata_start = lambda **kw: started.append(kw) or {"summary": "running"}
        out = self.a.calibrate_status()["calibrate"]
        self.assertEqual(started, [{"model": "iq3_s", "wait_seconds": 5}])
        self.assertEqual(out["then_start"]["summary"], "running")
        self.assertEqual(self.a._hook(), {})                 # asked once, not again on the next poll
        self.a.calibrate_status()["calibrate"]
        self.assertEqual(len(started), 1)


class Jobs(Base):
    def test_a_job_writes_its_header_and_keeps_its_output(self):
        # a real detached process (a print, no model): the log header, the child's own lines, and the end state
        import time
        job = Job(self.a.state_dir, "measure")
        try:
            job.start([sys.executable, "-c", "print('measured 41.2 tokens/s')"], ROOT, "a short measurement")
            self.assertIn("launcher: a short measurement at", job.log_path.read_text(encoding="utf-8"))
            for _ in range(80):
                if not job.running():
                    break
                time.sleep(0.25)
            self.assertFalse(job.running())
            self.assertIn("measured 41.2 tokens/s", "\n".join(job.status(lines=20)["lines"]))
        finally:
            job._save(None)

    def test_the_measurement_child_refuses_a_config_that_is_not_there(self):
        import contextlib
        import io
        from launcher import calibrate as cal
        out = Path(self.dir.name) / "res.json"
        with contextlib.redirect_stdout(io.StringIO()) as said:
            code = cal.main(["no-such-config.json", str(out)])
        self.assertEqual(code, 2)
        self.assertIn("is not a model config", said.getvalue())
        self.assertFalse(out.exists())

    def test_the_measurement_job_is_told_where_to_write_its_result(self):
        # the child and the job must name the same file, or the card shows nothing when it ends
        captured = []
        self.a.s.find_config = lambda mid: Path("strata-iq3_s.json")
        self.a.s.run_python = lambda: sys.executable
        self.a.calibrate_job.start = lambda cmd, cwd, label, **kw: captured.append((cmd, kw.get("result")))
        self.a.calibrate("iq3_s")
        cmd, result = captured[0]
        self.assertEqual(cmd[:4], [sys.executable, "-m", "launcher.calibrate", "strata-iq3_s.json"])
        self.assertEqual(str(result), cmd[4])
        self.assertEqual(Path(cmd[4]).parent, self.a.state_dir)

    def test_a_measurement_that_never_started_does_not_start_the_model_afterwards(self):
        self.a.s.find_config = lambda mid: Path("strata-iq3_s.json")
        self.a.s.run_python = lambda: sys.executable
        self.a.calibrate_job.start = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("the job could not start"))
        started = []
        self.a.t.strata_start = lambda **kw: started.append(kw)
        with self.assertRaises(RuntimeError):
            self.a.calibrate("iq3_s", then_start=True)
        self.assertEqual(self.a._hook(), {})                 # nothing was asked, so nothing starts later
        self.a.calibrate_status()
        self.assertEqual(started, [])


class ModelCard(Base):
    """What the page shows next to the preset: the model behind it on this PC, not a speed number."""

    def test_the_model_card_names_the_model_and_says_whether_it_is_installed(self):
        p = self.a.save_preset(PRESET)
        card = self.a.model_for_preset(p["id"])
        self.assertEqual(card["model"], "iq3_s")
        self.assertEqual(card["model_name"], "qwen3.8-flash-next-iq3_s")
        self.assertFalse(card["installed"])

    def test_the_model_card_says_which_model_the_server_is_serving(self):
        # the Start button and the "another model is running" warning follow this, so it has to be the running one
        p = self.a.save_preset(PRESET)
        self.a.s.probe = lambda port, deep=True: {"port": 8080, "model": "qwen3.8-flash-next-iq3_s", "loaded": True}
        card = self.a.model_for_preset(p["id"])
        self.assertEqual(card["live"]["model"], "qwen3.8-flash-next-iq3_s")
        self.assertEqual(card["live"]["port"], 8080)


class Page(unittest.TestCase):
    """The routes: the page, its files, and an answer that is an error but still JSON."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name) / "Strata"
        (root / "serve").mkdir(parents=True)
        (root / "serve" / "server.py").write_text("# a stand-in install\n", encoding="utf-8")
        (root / "setup.py").write_text("# a stand-in setup\n", encoding="utf-8")
        launcher_app.Handler.api = Launcher(root, state_dir=root / ".strata-launcher")
        self.httpd = launcher_app.Server(("127.0.0.1", 0), launcher_app.Handler)   # the class the launcher really uses
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.dir.cleanup()

    def get(self, path):
        """A request's status, type and body - an error status is an answer too."""
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=10) as r:
                return r.status, r.headers.get("Content-Type"), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type"), e.read()

    def test_the_page_and_its_files_are_served(self):
        status, ctype, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        self.assertIn(b"Strata Launcher", body)
        for path, kind in (("/web/launcher.css", "text/css"), ("/web/launcher.js", "text/javascript"),
                           ("/ui/tokens.css", "text/css")):
            status, ctype, _ = self.get(path)
            self.assertEqual(status, 200, path)
            self.assertIn(kind, ctype)

    def test_a_path_cannot_escape_its_folder(self):
        self.assertEqual(self.get("/web/../setup.py")[0], 404)
        self.assertEqual(self.get("/api/nothing")[0], 404)

    def test_a_browser_that_vanishes_mid_answer_leaves_no_stack_on_the_console(self):
        # the page asks for the state every 3 seconds; a reload or a closed tab cuts an answer in half, and that is
        # not a launcher problem: no 500, no traceback on the console
        import contextlib
        import http.client
        import io

        def gone(hardware=False):
            raise ConnectionAbortedError("the browser went away")

        launcher_app.Handler.api.state = gone
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
            c.request("GET", "/api/state")
            try:
                c.getresponse()
            except (http.client.BadStatusLine, http.client.RemoteDisconnected, ConnectionError, OSError):
                pass
            c.close()
        self.assertNotIn("Traceback", err.getvalue(), err.getvalue())

    def test_an_error_is_a_message_the_page_can_show(self):
        status, _, body = self.get("/api/plan?id=not-a-preset")
        self.assertEqual(status, 400)
        self.assertIn("no preset", json.loads(body)["error"]["message"])


if __name__ == "__main__":
    unittest.main()
