"""Lane NET item 3: packaging/install.sh's fetch() under the network rule, against local servers: a TLS server that never answers its handshake
(given up after the 15 s connect limit and one retry, about 32 s, not 10 minutes), a refused port (at once), a working server, and a slow link that
keeps sending (never cut; real-time, STRATA_SLOW_NET_TESTS=1). curl and POSIX sh are needed; the Windows script is item 6."""

import os
import re
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from tests.terminal.test_netpolicy import BlackHole, Server, ok_body, refused_port, trickle

ROOT = Path(__file__).resolve().parents[2]
SH = ROOT / "packaging" / "install.sh"


def curl_fetch_line() -> str:
    m = re.search(r"^\s*fetch\(\) \{ curl [^\n]*\}", SH.read_text(encoding="utf-8"), re.M)
    assert m, "install.sh has no curl fetch()"
    return m.group(0).strip()


@unittest.skipIf(os.name == "nt" or not (shutil.which("sh") and shutil.which("curl")), "needs POSIX sh and curl")
class InstallFetch(unittest.TestCase):
    def fetch(self, url, timeout=120):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "got"
            t0 = time.monotonic()
            env = {k: v for k, v in os.environ.items() if k.lower() not in ("http_proxy", "https_proxy", "all_proxy")}
            r = subprocess.run(["sh", "-c", curl_fetch_line() + '\nfetch "$1" "$2"', "sh", url, str(out)], capture_output=True, text=True,
                               timeout=timeout, env=env)
            return r.returncode, (out.read_bytes() if out.exists() else b""), time.monotonic() - t0, r.stderr

    def test_the_limits_are_in_the_script(self):
        line = curl_fetch_line()
        for flag in ("--connect-timeout 15", "--speed-limit 1", "--speed-time 60", "--retry 1"):
            self.assertIn(flag, line)
        self.assertNotIn("--max-time", line)                 # no total cap: a slow download that works is not cut
        self.assertNotIn("--retry-connrefused", line)        # a refused connection fails at once

    def test_a_handshake_that_never_answers_is_given_up_in_about_half_a_minute(self):
        hole = BlackHole()
        self.addCleanup(hole.close)
        code, _got, took, err = self.fetch(f"https://127.0.0.1:{hole.port}/install-manifest.txt")
        self.assertNotEqual(code, 0)
        self.assertGreater(took, 25)
        self.assertLess(took, 45, err)                       # 15 + 2 + 15 s, not 10 minutes
        self.assertEqual(len(hole.held), 2)

    def test_a_refused_port_fails_at_once(self):
        code, _got, took, _err = self.fetch(f"http://127.0.0.1:{refused_port()}/install-manifest.txt")
        self.assertNotEqual(code, 0)
        self.assertLess(took, 3)

    def test_a_working_server(self):
        srv = Server(ok_body(b"linux-x86_64 0.1 a b 1\n"))
        self.addCleanup(srv.close)
        code, got, _took, err = self.fetch(srv.url("/install-manifest.txt"))
        self.assertEqual((code, got), (0, b"linux-x86_64 0.1 a b 1\n"), err)

    @unittest.skipUnless(os.environ.get("STRATA_SLOW_NET_TESTS") == "1", "real-time: STRATA_SLOW_NET_TESTS=1")
    def test_a_slow_link_that_keeps_sending_is_not_cut(self):
        srv = Server(trickle(150, 0.5))                      # 2 bytes a second for 75 s: past the 60 s window, above 1 byte/s
        self.addCleanup(srv.close)
        code, got, took, err = self.fetch(srv.url("/terminal.tar.gz"), timeout=200)
        self.assertEqual((code, len(got)), (0, 150), err)
        self.assertGreater(took, 70)


if __name__ == "__main__":
    unittest.main()
