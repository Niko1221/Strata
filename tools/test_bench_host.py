"""Tests for the server address tools/strata_bench.py polls, and for the branch that starts its own server,
without starting one.

    python -m unittest tools.test_bench_host
"""
from __future__ import annotations

import collections
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_bench as SB  # noqa: E402

Run = collections.namedtuple("Run", "rc out err start polled")   # rc, stdout, stderr, the start_server mock, the address polled


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


class StartsItsOwnServer(unittest.TestCase):
    """#2872/#2864: 69fe4055 replaced `port = ...; base = f"http://127.0.0.1:{port}"` with a server_base() call and
    deleted the only assignment of the local `port`, which the "start a server ourselves" branch went on to read
    three times.  Every bench run without --url died with `NameError: name 'port' is not defined` before the model
    loaded, and the server_base() tests above never touched that branch because they call the helper directly.
    These drive main() into it, with the server it starts and the readiness poll stubbed out."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="strata-bench-test-")
        self.addCleanup(tmp.cleanup)
        self.cfg = Path(tmp.name) / "strata-iq3_s.json"
        self.cfg.write_text(json.dumps({"host": "0.0.0.0", "port": 8123}), encoding="utf-8")

    def start_branch(self, *argv, health=None) -> Run:
        """main() with nothing answering: it takes the start branch, so it has to reach start_server()."""
        out, err = io.StringIO(), io.StringIO()
        proc = mock.Mock()
        proc.poll.return_value = 0            # already gone again: main()'s stop_server() has nothing to clean up
        with mock.patch.object(SB, "is_strata", return_value=False) as is_strata, \
                mock.patch.object(SB, "try_json", return_value=health), \
                mock.patch.object(SB, "wait_ready", return_value=False), \
                mock.patch.object(SB, "start_server", return_value=proc) as start, \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = SB.main(["--config", str(self.cfg), *argv])
        return Run(rc, out.getvalue(), err.getvalue(), start, is_strata.call_args.args[0])

    def test_the_config_port_is_started_and_polled(self):
        r = self.start_branch()
        self.assertEqual(r.rc, 1)                     # the stubbed readiness poll never reports ready
        r.start.assert_called_once()
        self.assertEqual(r.start.call_args.args[1], 8123)
        # #2330's point: the config binds 0.0.0.0, which is not connectable as written, so the poll goes to
        # loopback - on the same port the server was given.
        self.assertEqual(r.polled, "http://127.0.0.1:8123")
        self.assertIn("on port 8123", r.out)

    def test_port_flag_overrides_the_config(self):
        r = self.start_branch("--port", "9000")
        self.assertEqual(r.start.call_args.args[1], 9000)
        self.assertEqual(r.polled, "http://127.0.0.1:9000")
        self.assertIn("on port 9000", r.out)

    def test_a_foreign_service_on_the_port_is_named_and_not_started_over(self):
        r = self.start_branch(health={"service": "something-else"})
        self.assertEqual(r.rc, 2)
        self.assertIn("Port 8123 is used by something that is not Strata", r.err)
        r.start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
