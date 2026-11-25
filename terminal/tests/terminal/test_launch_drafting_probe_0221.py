"""0221 APPTUI, wiring DSPARK's launch-time drafting probe into `strata launch` (Runner.probe_drafting): before the first start of a placement, when the run config
wants the measured choice, the engine is probed and the numbers are saved in the run config. Every backend is probed (the bit-exact gate is gone, owner 2026-11-23). The engine is DSPARK's fake one (tests/terminal/test_inkling_probe.py): no real engine is started."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve import inkling_kind
from strata_cli.launch import settings as S
from strata_cli.launch.runner import Runner
from strata_control import inkling_family, inkling_probe as P
from tests.terminal.launch_support import HW_12, Home, catalog
from tests.terminal.test_inkling_probe import FakeEngine

CFG = {"exe": "component:strata-inkling", "arch": "inkling2_loop", "backend": "cuda", "parallel": 1,
       "args": ["--native", "m.strata", "--max-context", "32768"], "cwd": "."}
SPEEDS = {"off": {"code": 40.0, "chat": 40.0}, "on": {"code": 80.0, "chat": 44.0}}


class ProbeWiring(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        t = tempfile.TemporaryDirectory(prefix="strata-probe-")
        self.addCleanup(t.cleanup)
        self.path = Path(t.name) / "launch.json"
        self.started = []

    def runner(self, cfg=None):
        self.path.write_text(json.dumps(cfg or CFG), encoding="utf-8")
        r = Runner(S.LaunchSettings(family="inkling-2-mini", model="SQ-10", context=32768), catalog=catalog(HW_12))
        r.config_path = self.path
        r.said = []
        r.emit_cb = lambda e: r.said.append((e.level if hasattr(e, "level") else None, e.text))
        return r

    def fake_engine(self, speeds=SPEEDS):
        def open_engine(exe, args, env=None, cwd=None):
            self.started.append((exe, list(args)))
            return FakeEngine(args, speeds)
        return [mock.patch.object(P, "Engine", open_engine),
                mock.patch("serve.server._component_exe", lambda ref, cfg: "/engine/strata-inkling"),
                mock.patch("serve.server.child_env", lambda cfg, **kw: {})]

    def saved(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def run_probe(self, r, speeds=SPEEDS):
        ps = self.fake_engine(speeds)
        for p in ps:
            p.start()
            self.addCleanup(p.stop)
        r.probe_drafting()

    def test_a_backend_is_probed_once_and_the_numbers_are_saved(self):
        r = self.runner()
        self.run_probe(r)
        self.assertEqual(len(self.started), 2)                       # plain, then drafting
        self.assertEqual(self.started[0][0], "/engine/strata-inkling")
        cfg = self.saved()
        self.assertEqual(cfg["drafting_probe"]["decision"], "on")
        self.assertEqual(cfg["drafting_probe"]["key"], P.placement_key(cfg))
        self.assertTrue(any("Measuring whether drafting makes this model faster" in t for _lv, t in r.said))
        self.assertFalse(inkling_kind.needs_drafting_probe(cfg))     # valid now
        self.started.clear()
        self.run_probe(self.runner(cfg))                             # the second start of the same placement: no probe
        self.assertEqual(self.started, [])

    def test_a_card_where_drafting_loses_is_saved_as_off(self):
        self.run_probe(self.runner(), {"off": {"code": 11.0, "chat": 11.0}, "on": {"code": 9.5, "chat": 6.3}})
        self.assertEqual(self.saved()["drafting_probe"]["decision"], "off")

    def test_another_placement_is_measured_again(self):
        r = self.runner()
        self.run_probe(r)
        cfg = self.saved()
        cfg["args"] = ["--native", "m.strata", "--max-context", "65536"]      # another context: another placement (the slot count is not part of the key)
        self.started.clear()
        self.run_probe(self.runner(cfg))
        self.assertEqual(len(self.started), 2)

    def test_an_explicit_choice_and_other_families_are_never_probed(self):
        for cfg in (dict(CFG, drafting="dspark"), dict(CFG, drafting="off")):
            self.run_probe(self.runner(cfg))
        qwen = Runner(S.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768), catalog=catalog(HW_12))
        qwen.config_path = self.path
        qwen.probe_drafting()
        self.assertEqual(self.started, [])

    def test_start_probes_before_the_engine_starts(self):
        order = []
        r = self.runner()
        with mock.patch.object(Runner, "probe_drafting", lambda self: order.append("probe")), \
                mock.patch.object(Runner, "stop_others", lambda self: None), \
                mock.patch.object(Runner, "check_port_free", lambda self: None), \
                mock.patch.object(Runner, "log_marks", lambda self: {}), \
                mock.patch.object(Runner, "report_drafting", lambda self: None), \
                mock.patch("strata_control.runtime.status", lambda name: {"state": "stopped"}), \
                mock.patch("strata_control.runtime.start", lambda name, wait=None: order.append("start") or {"ok": True, "generation_ready": True}):
            r.start()
        self.assertEqual(order, ["probe", "start"])

    def test_a_probe_that_fails_never_stops_the_launch(self):
        r = self.runner()
        with mock.patch.object(P, "probe_run_config", side_effect=RuntimeError("engine would not start")), \
                mock.patch("serve.server._component_exe", lambda ref, cfg: "/engine/strata-inkling"), \
                mock.patch("serve.server.child_env", lambda cfg, **kw: {}):
            r.probe_drafting()                                       # no exception
        self.assertTrue(any("could not be made (RuntimeError: engine would not start); drafting stays off" in t for _lv, t in r.said))
        self.assertNotIn("drafting_probe", self.saved())


if __name__ == "__main__":
    unittest.main()
