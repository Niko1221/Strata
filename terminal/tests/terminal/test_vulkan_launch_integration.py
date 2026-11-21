"""The Vulkan research path now shares Strata's launch lifecycle.

These tests pin the integration points: management flags must never start a
server, a launch must own a normal ``launch-vulkan-*`` profile so --status and
--stop find it, and a saved preset must route back to the Vulkan branch.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strata_cli.launch import cmd, vulkan_common  # noqa: E402
from strata_cli.launch.settings import LaunchSettings  # noqa: E402


def args(**override):
    a = dict(output="json", backend="vulkan", plan=False,
             model_path="/tmp/model.gguf", data_dir=None, port=8123, context=None,
             list_presets=False, delete_preset=None, stop=False, status=False,
             show_key=False, preset=None, save_preset=None, vulkan_impl=None)
    a.update(override)
    return argparse.Namespace(**a)


class ManagementBeforeVulkan(unittest.TestCase):
    def _run(self, a):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = cmd.run_launch(a)
        return rc, out.getvalue()

    def test_status_with_vulkan_backend_does_not_start_a_server(self):
        with mock.patch.object(cmd, "running_launches", return_value=[]), \
             mock.patch("strata_cli.launch.vulkan_native.launch") as launch:
            rc, out = self._run(args(status=True))
        self.assertEqual(rc, 0, out)
        self.assertIn('"running": []', out)
        launch.assert_not_called()

    def test_stop_with_vulkan_backend_does_not_start_a_server(self):
        with mock.patch.object(cmd, "_stop_all", return_value=0) as stop, \
             mock.patch("strata_cli.launch.vulkan_native.launch") as launch:
            rc, _ = self._run(args(stop=True, preset="vulkan"))
        self.assertEqual(rc, 0)
        stop.assert_called_once_with("launch-vulkan")
        launch.assert_not_called()

    def test_plan_is_still_read_only(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), \
             mock.patch.object(vulkan_common, "start") as start:
            rc = cmd.run_launch(args(plan=True))
        self.assertEqual(rc, 0)
        start.assert_not_called()
        self.assertEqual(json.loads(out.getvalue())["backend"], "vulkan")


class VulkanProfileLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-vulkan-launch-")
        self.addCleanup(self.temp.cleanup)
        self.config = Path(self.temp.name) / "config.json"
        self.config.write_text(json.dumps({"exe": "/tmp/engine", "tokenizer": "/tmp/tok",
                                           "args": [], "cwd": self.temp.name}))
        self.env = mock.patch.dict("os.environ", {
            "STRATA_PROFILE_DIR": str(Path(self.temp.name) / "profiles"),
            "STRATA_LAUNCH_PRESET_DIR": str(Path(self.temp.name) / "presets"),
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_ensure_profile_creates_and_updates(self):
        from strata_control import profiles
        name = vulkan_common.ensure_profile(self.config, "native", args())
        self.assertEqual(name, "launch-vulkan")
        prof, _ = profiles.load(name)
        self.assertEqual(prof["source_config"], str(self.config.resolve()))
        self.assertEqual(prof["overrides"], {"host": "127.0.0.1", "port": 8123})
        # A second call with a new port updates the same profile in place.
        vulkan_common.ensure_profile(self.config, "native", args(port=9999))
        prof, _ = profiles.load(name)
        self.assertEqual(prof["overrides"]["port"], 9999)

    def test_two_impls_get_two_profiles(self):
        from strata_control import profiles
        vulkan_common.ensure_profile(self.config, "native", args())
        vulkan_common.ensure_profile(self.config, "bridge", args())
        self.assertEqual(profiles.load("launch-vulkan-bridge")[0]["source_config"],
                         str(self.config.resolve()))

    def test_start_uses_runtime_and_reports_urls(self):
        with mock.patch("strata_control.runtime.start", return_value={
                "ok": True, "state": "ready"}) as start:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = vulkan_common.start(args(), "native", Path("/tmp/model.gguf"), self.config,
                                         machine=False)
        self.assertEqual(rc, 0, out.getvalue())
        start.assert_called_once()
        self.assertEqual(start.call_args[0][0], "launch-vulkan")
        self.assertIn("8123", out.getvalue())
        self.assertIn("strata launch --status", out.getvalue())

    def test_failed_start_reports_and_returns_nonzero(self):
        with mock.patch("strata_control.runtime.start", return_value={
                "ok": False, "state": "failed", "error": "engine_exited"}):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = vulkan_common.start(args(), "native", Path("/tmp/model.gguf"), self.config)
        self.assertEqual(rc, 1)
        self.assertIn("engine_exited", err.getvalue())

    def test_save_preset_stores_backend_for_restart(self):
        a = args(save_preset="my-vulkan")
        with mock.patch("strata_control.runtime.start", return_value={"ok": True, "state": "ready"}):
            with contextlib.redirect_stdout(io.StringIO()):
                vulkan_common.start(a, "bridge", Path("/tmp/model.gguf"), self.config)
        from strata_cli.launch import store
        saved = store.load("my-vulkan")
        self.assertEqual(saved.extra["backend"], "vulkan")
        self.assertEqual(saved.extra["vulkan_impl"], "bridge")

    def test_preset_routes_back_to_vulkan(self):
        preset = LaunchSettings(name="my-vulkan", model_path="/tmp/other.gguf",
                                port=7777, extra={"backend": "vulkan", "vulkan_impl": "bridge"})
        with mock.patch("strata_cli.launch.store.load", return_value=preset), \
             mock.patch.object(cmd, "running_launches", return_value=[]), \
             mock.patch("strata_cli.launch.vulkan_bridge.launch", return_value=0) as launch:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = cmd.run_launch(args(preset="my-vulkan", backend=None, model_path=None, port=None))
        self.assertEqual(rc, 0)
        launch.assert_called_once()
        a = launch.call_args[0][0]
        self.assertEqual(a.model_path, "/tmp/other.gguf")
        self.assertEqual(a.port, 7777)


if __name__ == "__main__":
    unittest.main()
