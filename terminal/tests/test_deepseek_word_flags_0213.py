"""Round 0213 TUICFG, step 7: `--set mmap-experts=off` reaches strata-dsv41 as a word (#5027; the approach of PR #5028, written here), through the one
precedence (arg_layers) and into a second launch."""
from __future__ import annotations

import json
import unittest

from strata_cli.launch import settings as S
from strata_cli.launch.runner import derive_config
from strata_control import arg_layers, tune_knobs

DS = {"exe": "/e/strata-dsv41", "arch": "deepseek41", "port": 8095, "model_name": "deepseek-v4.1-flash", "env": {},
      "args": ["--pack", "p", "--max-context", "262144", "--cpu-threads", "48"]}
QWEN = {"exe": "/e/strata", "port": 8095, "model_name": "qwen3.8-flash-next-iq3_s", "env": {}, "args": ["--pack", "p", "--max-context", "65536"]}


def launch(base, extra=None, prev=None):
    cfg = derive_config(json.loads(json.dumps(base)), S.LaunchSettings(extra=dict(extra or {})), bool_flags=tune_knobs.BOOL_FLAGS, prev=prev)
    arg_layers.stamp(cfg)
    return json.loads(json.dumps(cfg))


class MmapExperts(unittest.TestCase):
    def test_deepseek_gets_the_word_on_every_launch(self):
        for word in ("off", "on", "auto"):
            first = launch(DS, {"--mmap-experts": word})
            for cfg in (first, launch(DS, {"--mmap-experts": word}, prev=first)):
                args = cfg["args"]
                self.assertEqual(args[args.index("--mmap-experts") + 1], word, word)
                self.assertEqual(args.count("--mmap-experts"), 1)
                self.assertEqual(args[args.index("--cpu-threads") + 1], "48")
                self.assertEqual(cfg["user_args"][cfg["user_args"].index("--mmap-experts") + 1], word, "kept by the server's start as the person's value")

    def test_a_saved_word_is_replaced_not_doubled(self):
        base = {**DS, "args": [*DS["args"], "--mmap-experts", "on"]}
        args = launch(base, {"--mmap-experts": "off"})["args"]
        self.assertEqual(args[-2:], ["--mmap-experts", "off"])
        self.assertEqual(args.count("--mmap-experts"), 1)
        self.assertNotIn("on", args)

    def test_leaving_it_out_next_time_gives_the_engines_default_back(self):
        first = launch(DS, {"--mmap-experts": "off"})
        second = launch(DS, prev=first)
        self.assertNotIn("--mmap-experts", second["args"])
        self.assertNotIn("--mmap-experts", second.get("user_args") or [])

    def test_qwen_keeps_the_bare_switch(self):
        on = launch(QWEN, {"--mmap-experts": "on"})["args"]
        self.assertEqual(on[-1], "--mmap-experts")
        base = {**QWEN, "args": [*QWEN["args"], "--mmap-experts"]}
        off = launch(base, {"--mmap-experts": "off"})
        self.assertNotIn("--mmap-experts", off["args"])
        self.assertNotIn("--mmap-experts", off.get("user_args") or [])


if __name__ == "__main__":
    unittest.main()
