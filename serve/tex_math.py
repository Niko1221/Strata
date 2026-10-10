"""Optional full-TeX adapter transport. No install or container startup in the server."""
import json
import os
import subprocess
import threading

_lock = threading.Lock()


def render(req):
    source, display = req.get("source"), req.get("display")
    if not isinstance(source, str) or not isinstance(display, bool):
        return {"ok": False, "reason": "invalid"}
    try:
        source_bytes = source.encode("utf-8")
    except UnicodeEncodeError:
        return {"ok": False, "reason": "invalid"}
    if len(source_bytes) > 16384:
        return {"ok": False, "reason": "limit"}
    container = os.environ.get("STRATA_TEX_CONTAINER")
    if not container:
        return {"ok": False, "reason": "unavailable"}
    # No request values enter the command line, container name, or executable path.
    if not _lock.acquire(blocking=False):
        return {"ok": False, "reason": "limit"}
    try:
        proc = subprocess.run([os.environ.get("STRATA_TEX_DOCKER", "docker"), "exec", "--interactive", container,
                               "python3", "/opt/strata-math/worker.py"],
                              input=json.dumps({"source": source, "display": display}).encode(),
                              capture_output=True, timeout=12)
        if proc.returncode or len(proc.stdout) > 3 * 1024 * 1024:
            return {"ok": False, "reason": "unavailable"}
        result = json.loads(proc.stdout)
        if not isinstance(result, dict) or not isinstance(result.get("ok"), bool):
            return {"ok": False, "reason": "unavailable"}
        return result
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "limit"}
    except (OSError, ValueError):
        return {"ok": False, "reason": "unavailable"}
    finally:
        _lock.release()
