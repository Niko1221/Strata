"""#7035: Inkling-2-Mini takes its thinking level in the same order as every other model: the request's own level, then the level set in
the Chat settings shared with other apps, then the config's "reasoning_effort"; only when none of them is set does the Inkling default
(low, STRATA_INKLING_EFFORT) apply.  Before, effort_default() returned None for Inkling, so the config and the shared level were skipped.
The start-up line names only a level that is used.

    python -m pytest serve/test_inkling_config_effort.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve import inkling_dialect  # noqa: E402
from serve.frontend import ChatTemplate, anthropic_to_messages, openai_to_messages  # noqa: E402
from serve.server import Service, thinking_level_line  # noqa: E402
from serve.test_inkling_reasoning_budget import TEMPLATE, InkTokenizer, ThinkingEngine  # noqa: E402


class InklingConfigEffort(unittest.TestCase):
    def svc(self, config=None, shared=None):
        tok = InkTokenizer()
        svc = Service(ThinkingEngine(tok, "x"), tok, ChatTemplate(TEMPLATE))
        self.assertTrue(svc.inkling)
        svc.set_family("inkling2")                  # as _main does: the frontend reads the Inkling effort words
        self.addCleanup(svc.set_family, None)
        svc.config_effort = config
        svc.shared = shared or {}
        return svc

    def level_openai(self, svc, **extra):
        req = svc.with_shared({"messages": [{"role": "user", "content": "hi"}], **extra}, "openai")
        _, _, kw = openai_to_messages(req)
        return inkling_dialect.effort_of(kw)[0]

    def level_anthropic(self, svc, **extra):
        req = svc.with_shared({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 50, **extra}, "anthropic")
        _, _, kw = anthropic_to_messages(req, True)
        return inkling_dialect.effort_of(kw)[0]

    def test_the_effort_default_is_not_none_for_inkling(self):
        self.assertEqual(self.svc("none").effort_default(), "none")
        self.assertEqual(self.svc("high", {"reasoning_effort": "low"}).effort_default(), "low")      # shared before the config
        self.assertIsNone(self.svc().effort_default())

    def test_the_config_level_applies_to_a_request_that_names_none(self):
        for cfg in ("none", "low", "medium", "high"):
            self.assertEqual(self.level_openai(self.svc(cfg)), cfg, cfg)
        # (the config's "none" was the report: the request thought at low anyway)
        self.assertEqual(self.level_openai(self.svc("none")), "none")

    def test_the_order_is_request_then_shared_then_config(self):
        svc = self.svc("none", {"reasoning_effort": "medium"})
        self.assertEqual(self.level_openai(svc), "medium")                                    # shared beats the config
        self.assertEqual(self.level_openai(svc, reasoning_effort="high"), "high")             # the request beats both
        self.assertEqual(self.level_openai(svc, reasoning_effort="none"), "none")
        self.assertEqual(self.level_openai(svc, chat_template_kwargs={"enable_thinking": False}), "none")
        self.assertEqual(self.level_openai(self.svc("high"), reasoning_effort="low"), "low")

    def test_nothing_set_anywhere_keeps_the_inkling_default(self):
        # 0.2.21 made "low" the level of a request that names none (owner decision 2026-11-23): unchanged when nothing is set
        svc = self.svc()
        self.assertEqual(self.level_openai(svc), inkling_dialect.DEFAULT_EFFORT)
        self.assertEqual(inkling_dialect.DEFAULT_EFFORT, "low")
        self.assertEqual(self.level_anthropic(svc), "low")

    def test_the_anthropic_api_reads_it_too(self):
        self.assertEqual(self.level_anthropic(self.svc("high")), "high")
        self.assertEqual(self.level_anthropic(self.svc("none")), "none")                      # the config's "none" -> thinking disabled
        self.assertEqual(self.level_anthropic(self.svc("none", {"reasoning_effort": "medium"})), "medium")
        self.assertEqual(self.level_anthropic(self.svc("none"), thinking={"type": "enabled", "budget_tokens": 9000}), "high")

    def test_the_start_up_line_says_only_what_is_used(self):
        line = thinking_level_line("none", True)
        self.assertIn("thinking level: none", line)
        self.assertIn("reasoning_effort in the config", line)
        self.assertIn("thinking level: low", thinking_level_line(None, True))        # nothing in the config: the default, named as such
        self.assertIn("Inkling default", thinking_level_line(None, True))
        self.assertIsNone(thinking_level_line(None, False))                          # another model, nothing set: no line
        self.assertIn("thinking level: high", thinking_level_line("high", False))


if __name__ == "__main__":
    unittest.main()
