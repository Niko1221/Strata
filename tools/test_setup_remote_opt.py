"""0.1.39b (#578): setup recommends --remote-expert-opt for a config on two or more GPUs, never for one, and keeps it
out when asked (setup's --no-remote-expert-opt, or "remote_expert_opt": false in the config).

    python -m unittest tools.test_setup_remote_opt
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_golden import PROFILES, install  # noqa: E402

FLAG = "--remote-expert-opt"


class Rule(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(setup, "ok", lambda *a: None)
        p.start()
        self.addCleanup(p.stop)

    def test_two_gpus_get_it_once(self):
        cfg = {"args": ["--spec", "4"], "gpu": [0, 1]}
        setup.recommend_remote_expert_opt(cfg)
        setup.recommend_remote_expert_opt(cfg)
        self.assertEqual(cfg["args"], ["--spec", "4", FLAG])

    def test_one_gpu_is_untouched(self):
        for gpu in (None, 0, [0]):
            cfg = {"args": ["--spec", "4"]} if gpu is None else {"args": ["--spec", "4"], "gpu": gpu}
            setup.recommend_remote_expert_opt(cfg)
            self.assertEqual(cfg["args"], ["--spec", "4"], gpu)

    def test_overrides(self):
        cfg = {"args": ["--spec", "4", FLAG], "gpu": [0, 1]}
        setup.recommend_remote_expert_opt(cfg, off=True)
        self.assertEqual(cfg["args"], ["--spec", "4"])
        cfg = {"args": ["--spec", "4"], "gpu": [0, 1, 2], "remote_expert_opt": False}
        setup.recommend_remote_expert_opt(cfg)
        self.assertEqual(cfg["args"], ["--spec", "4"])

    def test_the_users_key_is_kept_on_a_rerun(self):
        self.assertNotIn("remote_expert_opt", setup.SETUP_KEYS)   # #629: carried over, so the opt-out survives


class Install(unittest.TestCase):
    def test_install_on_two_gpus(self):
        ram, found = PROFILES["47GB-2x16GB"]
        args = ["--family", "qwen", "--no-start", "--model", "Q2_0"]
        code, text, cfg, _ = install(ram, found, args)
        self.assertEqual(code, 0, text)
        self.assertIsInstance(cfg.get("gpu"), list)
        self.assertEqual(cfg["args"].count(FLAG), 1)
        self.assertIn("--no-remote-expert-opt leaves it out", text)
        code, text, off, _ = install(ram, found, args + ["--no-remote-expert-opt"])
        self.assertEqual(code, 0, text)
        self.assertNotIn(FLAG, off["args"])
        self.assertEqual([a for a in cfg["args"] if a != FLAG], off["args"])

    def test_install_on_one_gpu(self):
        ram, found = PROFILES["96GB-1x16GB"]
        code, text, cfg, _ = install(ram, found, ["--family", "qwen", "--no-start", "--model", "Q2_0"])
        self.assertEqual(code, 0, text)
        self.assertNotIn(FLAG, cfg["args"])


if __name__ == "__main__":
    unittest.main()
