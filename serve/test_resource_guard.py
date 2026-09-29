"""Pressure tests use synthetic readings, never allocate enough to exhaust the host."""
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

import psutil

from serve.resource_guard import Limits, MemoryProbe, supervise


class ResourceGuardTests(unittest.TestCase):
    def test_limits_and_missing_measurements(self):
        limits = Limits(100, 10)
        self.assertIsNone(limits.violation({"RAM": 100, "VRAM": 10}))
        for sample in ({"RAM": 99, "VRAM": 11}, {"RAM": 101},
                       {"RAM": float("nan"), "VRAM": 20}):
            self.assertIsNotNone(limits.violation(sample))
        self.assertIsNone(Limits(100).violation({"RAM": 101}))
        for values in ((0, 0), (-1, 2), (1, 2, 0), (float("inf"), 2)):
            with self.assertRaises(ValueError):
                Limits(*values)

    def test_denied_admission_does_not_start_command(self):
        # A nonexistent executable would raise if Popen were reached.
        for probe in (lambda: {"RAM": 1}, lambda: {}):
            self.assertEqual(2, supervise(["strata-no-such-command"], Limits(2), probe, report=lambda _: None))

    def test_live_ram_probe_and_normal_exit_code(self):
        probe = MemoryProbe(Limits(1), 0)
        self.assertGreater(probe()["RAM"], 1)
        result = supervise([sys.executable, "-c", "raise SystemExit(7)"], Limits(1), probe)
        self.assertEqual(7, result)

    def test_pressure_and_telemetry_failure_stop_only_owned_descendants(self):
        for missing in (False, True):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as folder:
                marker = Path(folder) / "child.pid"
                command = [sys.executable, "-c",
                           "import pathlib,subprocess,sys,time; "
                           "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']); "
                           "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(120)", str(marker)]
                unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(120)"])
                deadline = time.monotonic() + 15
                messages = []

                def probe():
                    if marker.exists():
                        return {} if missing else {"RAM": 0}
                    if time.monotonic() > deadline:
                        raise RuntimeError("fixture did not start")
                    return {"RAM": 100}

                try:
                    self.assertEqual(3, supervise(command, Limits(1, interval=.05), probe, report=messages.append))
                    self.assertTrue(marker.exists(), messages)
                    self.assertIsNone(unrelated.poll())
                    pid = int(marker.read_text())
                    try:
                        child = psutil.Process(pid)
                        child.wait(timeout=5)
                    except psutil.NoSuchProcess:
                        pass
                    self.assertTrue(messages)
                finally:
                    unrelated.kill()
                    unrelated.wait()

    def test_frontend_exit_does_not_leave_child_running(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "child.pid"
            command = [sys.executable, "-c",
                       "import pathlib,subprocess,sys; "
                       "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']); "
                       "pathlib.Path(sys.argv[1]).write_text(str(p.pid))", str(marker)]
            self.assertEqual(0, supervise(command, Limits(1), lambda: {"RAM": 100}))
            try:
                psutil.Process(int(marker.read_text())).wait(timeout=5)
            except psutil.NoSuchProcess:
                pass

    def test_mock_server_stops_and_releases_listener(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        url = f"http://127.0.0.1:{port}/health"
        ready = []
        deadline = time.monotonic() + 15

        def probe():
            try:
                with urllib.request.urlopen(url, timeout=.2) as response:
                    ready.append(response.status)
                return {"RAM": 0}
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("mock server did not start")
                return {"RAM": 100}

        command = [sys.executable, "-m", "serve.server", "--engine", "mock", "--port", str(port)]
        self.assertEqual(3, supervise(command, Limits(1, interval=.05), probe, report=lambda _: None))
        self.assertEqual([200], ready)
        with self.assertRaises(OSError):
            urllib.request.urlopen(url, timeout=.2)


if __name__ == "__main__":
    unittest.main()
