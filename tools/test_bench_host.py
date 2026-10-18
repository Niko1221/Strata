"""Tests for the server address tools/strata_bench.py polls, without starting one.

    python -m unittest tools.test_bench_host
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_bench as SB  # noqa: E402


class ServerBase(unittest.TestCase):
    def test_uses_the_configured_host(self):
        # The readiness poll was hardcoded to 127.0.0.1 while the server was started on the config's
        # host, so a LAN bind left BENCH saying "still loading" until it was killed (#2330).
        self.assertEqual(SB.server_base("192.168.1.10", 8080), "http://192.168.1.10:8080")

    def test_defaults_to_loopback_without_a_host(self):
        self.assertEqual(SB.server_base("", 9000), "http://127.0.0.1:9000")
        self.assertEqual(SB.server_base(None, 9000), "http://127.0.0.1:9000")

    def test_wildcard_binds_are_connected_through_loopback(self):
        self.assertEqual(SB.server_base("0.0.0.0", 8080), "http://127.0.0.1:8080")
        self.assertEqual(SB.server_base("::", 8080), "http://[::1]:8080")

    def test_ipv6_literals_are_bracketed(self):
        self.assertEqual(SB.server_base("fd00::5", 8080), "http://[fd00::5]:8080")
        self.assertEqual(SB.server_base("[fd00::5]", 8080), "http://[fd00::5]:8080")

    def test_a_hostname_is_left_alone(self):
        self.assertEqual(SB.server_base("bench.local", 8080), "http://bench.local:8080")


class _NoProc:
    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        pass


class StartServerPort(unittest.TestCase):
    """The path that starts its own server names the port it starts it on. 69fe4055 dropped the
    `port` assignment when it inlined the value into server_base, and the three uses of `port`
    below it died with a NameError; the bench could no longer start its own server."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        cfg = Path(self.tmp.name) / "strata-test.json"
        cfg.write_text('{"port": 8123}', encoding="utf-8")
        self.cfg = str(cfg)
        self.started = []
        fakes = {
            "is_strata": lambda base, key: False,
            "try_json": lambda *a, **k: None,
            "wait_ready": lambda base, key, proc, timeout, log=None: False,
            "start_server": lambda cfg_path, port, gpu, log_path: self.started.append(port) or _NoProc(),
        }
        for name, fn in fakes.items():
            real = getattr(SB, name)
            setattr(SB, name, fn)
            self.addCleanup(setattr, SB, name, real)

    def test_the_config_port_reaches_start_server(self):
        self.assertEqual(SB.main(["--config", self.cfg, "--quiet"]), 1)   # not ready: the stub says so
        self.assertEqual(self.started, [8123])

    def test_the_port_option_wins(self):
        self.assertEqual(SB.main(["--config", self.cfg, "--port", "9111", "--quiet"]), 1)
        self.assertEqual(self.started, [9111])

if __name__ == "__main__":
    unittest.main()
