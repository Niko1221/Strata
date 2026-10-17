"""Installed-package smoke test: no model, GPU, checkout or loader overrides."""
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

package = Path(sys.argv[1])
assert not os.environ.get("LD_LIBRARY_PATH")
# --help exits before HIP initialization, but the ELF loader resolves ROCm libs.
commands = ["strata-iq-pack", "strata-mtp-pack", "strata-mtp-rt", "strata-tokenizer"]
if "--server-only" not in sys.argv:
    commands += ["strata", "strata-device"]
for command in commands:
    subprocess.run([package / "bin" / command, "--help"], check=True, stdout=subprocess.DEVNULL)

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
proc = subprocess.Popen([
    package / "bin/strata-server", "--engine", "mock",
    "--host", "127.0.0.1", "--port", str(port),
], cwd=os.environ["TMPDIR"])
try:
    for _ in range(100):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/health")
            response = conn.getresponse()
            response.read()
            conn.close()
            if response.status == 200:
                break
        except OSError:
            pass
        time.sleep(0.1)
    else:
        raise RuntimeError("mock server did not become healthy")

    for path, expected in (("/", b"<html"), ("/web/app.js", b""), ("/v1/models", b"data")):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", path)
        response = conn.getresponse()
        body = response.read()
        assert response.status == 200, (path, response.status, body)
        assert body and expected in body, path
        if path == "/v1/models":
            assert json.loads(body)["data"]
        conn.close()
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
