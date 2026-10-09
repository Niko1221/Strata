"""A port held by another Strata server can be taken over (--replace, or y in a terminal); anything else is left alone.
    python -m unittest serve.test_port_replace     (real servers with the mock engine; POSIX: lsof, ps, pgrep)"""
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def up(port, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).read()
            return True
        except Exception:                                  # noqa: BLE001 - starting up: refused, reset or cut short
            time.sleep(0.2)
    return False


@unittest.skipUnless(os.name == "posix" and shutil.which("lsof") and shutil.which("pgrep"), "POSIX lsof/pgrep needed")
class PortReplace(unittest.TestCase):
    def server(self, port, *extra):
        p = subprocess.Popen([sys.executable, str(HERE / "server.py"), "--engine", "mock", "--port", str(port), *extra],
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.addCleanup(lambda: (p.poll() is None) and (p.kill(), p.wait()))
        return p

    def test_replace_stops_the_strata_server_on_the_port(self):
        port = free_port()
        old = self.server(port)
        self.assertTrue(up(port))
        new = self.server(port, "--replace")
        self.assertTrue(up(port, 60))
        self.assertEqual(old.wait(30) is not None, True)          # the old one ended
        self.assertIsNone(new.poll())                              # the new one runs

    def test_a_suspended_server_is_stopped_too(self):
        port = free_port()
        old = self.server(port)
        self.assertTrue(up(port))
        os.kill(old.pid, signal.SIGSTOP)                           # Ctrl+Z
        new = self.server(port, "--replace")
        self.assertTrue(up(port, 60))
        self.assertIsNotNone(old.wait(30))

    def test_without_replace_and_no_terminal_nothing_is_stopped(self):
        port = free_port()
        old = self.server(port)
        self.assertTrue(up(port))
        new = self.server(port)
        out = new.communicate(timeout=60)[0]
        self.assertNotEqual(new.returncode, 0)
        self.assertIn("--replace", out)
        self.assertIsNone(old.poll())                              # still running

    def test_another_program_on_the_port_is_never_stopped(self):
        port = free_port()
        other = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: (other.kill(), other.wait()))
        time.sleep(1)
        new = self.server(port, "--replace")
        out = new.communicate(timeout=60)[0]
        self.assertNotEqual(new.returncode, 0)
        self.assertIn("held by", out)
        self.assertIsNone(other.poll())                            # left alone


if __name__ == "__main__":
    unittest.main()
