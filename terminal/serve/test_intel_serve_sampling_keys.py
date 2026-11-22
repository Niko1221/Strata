"""The Intel serve wrapper must accept the keywords the shared server sends (regression since v0.2.18).

terminal/serve/server.py calls StrataEngine.sampling_keys(...) with hy4= and inkling= at both call
sites under generate() — the GEN/GENI head (the f-strings) and the conversation-cache key (keys0).
engine/sycl/serve/server_intel.py shadows the method with install_logprobs' wrapper, which before
this regression guard's fix took only `sampling`: since v0.2.18 (836ab9714, "serve: advertise
Hy4-Flash in-process vision and strip unsupported GEN keys") every /v1/chat/completions served
through the Intel source path answered

    {"error": {"type": "invalid_request_error",
        "message": "the request is malformed (TypeError: install_logprobs.<locals>.sampling_keys()
        got an unexpected keyword argument 'hy4')"}}

before the engine saw a line. The mock engine never enters generate(), which is why every
mock-based smoke run (packaging/smoke_terminal.py included) stayed green through three releases.

The check runs in a subprocess on purpose: server_intel.py imports serve.server with sys.path
entries of its own, and unittest discovery runs every test_*.py in one process, where another
test may already hold a different `serve` module. A child interpreter is immune to that, needs
no GPU, and leaves nothing patched behind for the tests that follow.
"""
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]        # the checkout root (…/terminal/serve/<this file>)
WRAPPER = REPO / "engine" / "sycl" / "serve" / "server_intel.py"

# Exactly the shared server's call sites (terminal/serve/server.py): the GEN/GENI head and the
# conversation-cache key both call sampling_keys(sampling, hy4=getattr(self,'hy4_engine',False),
# inkling=getattr(self,'inkling_engine',False)). The wrapper must take the keywords — and forward
# them, so a config that really serves strata-hy4 through this wrapper keeps the shared function's
# hy4 key stripping — and its own logprobs suffix must survive them.
CHECK = """
import importlib.util
spec = importlib.util.spec_from_file_location("server_intel_under_test", {wrapper!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.install_logprobs()
keys = mod.S.StrataEngine.sampling_keys
assert "logprobs" not in keys({{}}, hy4=False, inkling=False)
k = keys({{"logprobs": True, "top_logprobs": 5}}, hy4=False, inkling=False)
assert k.endswith(" logprobs=5"), k
k = keys({{"logprobs": 3}}, hy4=False, inkling=False)
assert k.endswith(" logprobs=3"), k
assert "logprobs" not in keys({{}}, hy4=True, inkling=True)
print("ok")
"""


class SamplingKeysTakesTheSharedCallSitesKeywords(unittest.TestCase):

    def test_wrapper_accepts_and_forwards_the_keywords(self):
        proc = subprocess.run(                                    # noqa: S603 - our own interpreter, our own source
            [sys.executable, "-c", CHECK.format(wrapper=str(WRAPPER))],
            capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            self.fail(f"the wrapper rejected the shared call sites' keywords:\n{proc.stderr}")


if __name__ == "__main__":
    unittest.main()
