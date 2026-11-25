"""Regression test for PR #6610: engine/sycl/serve/server_intel.py install_logprobs() must forward
`hy4=` / `inkling=` keyword arguments to StrataEngine.sampling_keys instead of raising TypeError."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[1]
REPO = TERMINAL.parent
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

import serve.server as S  # noqa: E402


def _load_server_intel():
    path = REPO / "engine" / "sycl" / "serve" / "server_intel.py"
    spec = importlib.util.spec_from_file_location("strata_test_server_intel", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


class TestIntelServeSamplingKeys(unittest.TestCase):
    def setUp(self):
        self._orig_pump = S.StrataEngine.__dict__["_pump"]
        self._orig_keys = S.StrataEngine.__dict__["sampling_keys"]
        self._orig_generate = S.StrataEngine.__dict__["generate"]
        self._orig_chunks = S.openai_chunks
        self._orig_collect = S.openai_collect
        self.intel = _load_server_intel()
        self.intel.install_logprobs()

    def tearDown(self):
        S.StrataEngine._pump = self._orig_pump
        S.StrataEngine.sampling_keys = self._orig_keys
        S.StrataEngine.generate = self._orig_generate
        S.openai_chunks = self._orig_chunks
        S.openai_collect = self._orig_collect

    def test_sampling_keys_forwards_hy4_and_inkling_kwargs(self):
        base = {"temperature": 0.7, "top_p": 0.9, "seed": 42}
        # Shared serve/server.py calls sampling_keys(sampling, hy4=..., inkling=...) in StrataEngine.generate()
        plain = S.StrataEngine.sampling_keys(base, hy4=False, inkling=False)
        self.assertIn("temperature=0.7", plain)
        self.assertIn("seed=42", plain)
        self.assertNotIn("logprobs=", plain)

        with_lp = S.StrataEngine.sampling_keys({**base, "logprobs": True, "top_logprobs": 5}, hy4=False, inkling=False)
        self.assertIn("logprobs=5", with_lp)

        hy4_out = S.StrataEngine.sampling_keys({**base, "strata_checkpoint": False}, hy4=True, inkling=False)
        self.assertIn("temperature=0.7", hy4_out)
        self.assertNotIn("ckpt=", hy4_out)

        ink_out = S.StrataEngine.sampling_keys({**base, "strata_checkpoint": False}, hy4=False, inkling=True)
        self.assertIn("temperature=0.7", ink_out)
        self.assertNotIn("ckpt=", ink_out)

        # None sampling dict is tolerated cleanly, and calling via an instance still works (staticmethod preserved)
        self.assertEqual(S.StrataEngine.sampling_keys(None, hy4=False, inkling=False), "")
        inst = object.__new__(S.StrataEngine)
        self.assertIn("temperature=0.7", inst.sampling_keys(base, hy4=False, inkling=False))


if __name__ == "__main__":
    unittest.main()
