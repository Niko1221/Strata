"""#7044: engine_args adds the engine's own --conversation-cache-mib on a PC with a lot of RAM, and only then (strata_control/conversation_cache_size.py).

    python -m pytest serve/test_conversation_cache_budget.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import serve.server as server  # noqa: E402
from strata_control import conversation_cache_size as C  # noqa: E402

GIB = 1024.0


def flag_value(args):
    return args[args.index("--conversation-cache-mib") + 1] if "--conversation-cache-mib" in args else None


class EngineArgsBudget(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.model = Path(d.name) / "m.gguf"
        self.model.write_bytes(b"x" * 1024)
        self.cfg = {"args": ["--native", str(self.model), "--max-context", "8192"]}
        self.big = mock.patch.object(C, "host_ram_mib", return_value=(256 * GIB, 240 * GIB))
        self.big.start()
        self.addCleanup(self.big.stop)

    def args(self, cfg=None, has=True, announce=False):
        with mock.patch.object(server, "engine_has_text", return_value=has):
            return server.engine_args(cfg or self.cfg, "strata", announce=announce)

    def test_a_big_pc_with_an_engine_that_has_the_option(self):
        out = self.args()
        self.assertGreater(int(flag_value(out)), 8192)

    def test_the_other_arguments_are_what_they_were(self):
        with_flag = self.args()
        with mock.patch.dict("os.environ", {"STRATA_NO_AUTO_CONVERSATION_CACHE": "1"}):
            plain = self.args()
        self.assertNotIn("--conversation-cache-mib", plain)
        i = with_flag.index("--conversation-cache-mib")
        self.assertEqual(with_flag[:i] + with_flag[i + 2:], plain)

    def test_an_engine_without_the_option_or_an_unreadable_one_gets_nothing(self):
        self.assertNotIn("--conversation-cache-mib", self.args(has=False))
        self.assertNotIn("--conversation-cache-mib", self.args(has=None))

    def test_a_small_pc_gets_nothing(self):
        with mock.patch.object(C, "host_ram_mib", return_value=(16 * GIB, 12 * GIB)):
            self.assertNotIn("--conversation-cache-mib", self.args())

    def test_the_persons_own_value_stays(self):
        cfg = {"args": [*self.cfg["args"], "--conversation-cache-mib", "24576"]}
        out = self.args(cfg)
        self.assertEqual(out.count("--conversation-cache-mib"), 1)
        self.assertEqual(flag_value(out), "24576")
        cfg = {"args": [*self.cfg["args"]], "user_args": ["--conversation-cache-mib", "0"]}
        self.assertEqual(flag_value(self.args(cfg)), "0")

    def test_the_min_tokens_default_is_not_pulled_in_by_it(self):
        # #6463 adds --conversation-cache-min-tokens when the person's args carry a cache size; the size added here must not
        out = self.args()
        self.assertNotIn("--conversation-cache-min-tokens", out)

    def test_a_line_says_what_was_set_when_announced(self):
        said = []
        with mock.patch("builtins.print", side_effect=lambda *a, **k: said.append(" ".join(map(str, a)))):
            self.args(announce=True)
        line = [s for s in said if "conversation cache" in s]
        self.assertEqual(len(line), 1)
        self.assertTrue(line[0].startswith("[strata] conversation cache: "))

    def test_hy4_and_inkling_never_get_it(self):
        for cfg in ({"args": ["--model", "x"], "arch": "hy4flash"}, {"args": ["--model", "x"], "arch": "inkling2"}):
            with mock.patch.object(server.hy4_kind, "is_hy4", return_value=cfg["arch"] == "hy4flash"), \
                    mock.patch.object(server.inkling_kind, "is_inkling", return_value=cfg["arch"] == "inkling2"), \
                    mock.patch.object(server.hy4_kind, "engine_args", return_value=["--a"]), \
                    mock.patch.object(server.inkling_kind, "engine_args", return_value=["--a"]):
                self.assertEqual(server.engine_args(cfg, "strata", announce=False), ["--a"])


if __name__ == "__main__":
    unittest.main()
