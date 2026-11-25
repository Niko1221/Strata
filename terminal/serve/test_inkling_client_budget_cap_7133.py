"""#7133: a thinking budget a client asks for is capped at the default budget of the level it maps to (Inkling-2-Mini only).

Claude Code's default is `thinking: {type: enabled, budget_tokens: 31999}`.  That is level high and, since 0.2.21, a budget of 31,999 tokens with max_new 32,000: minutes of
hidden thinking a step at 25 to 60 tokens a second.  The level's own default is 8,192 (high).  Now a number a client sends (Anthropic budget_tokens, `reasoning_budget_tokens`
and its aliases) is lowered to that default and never raised by it; a smaller number stays; the run config's `inkling_budget_cap: false` (or STRATA_INKLING_BUDGET_CAP=0) keeps
what the client asks for; a request's own 0 still means no budget; the config's own budget and other families are untouched.

    python -m pytest serve/test_inkling_client_budget_cap_7133.py
"""
from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve import inkling_dialect  # noqa: E402
from serve.frontend import ChatTemplate, anthropic_to_messages, openai_to_messages  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402
from serve.test_inkling_reasoning_budget import TEMPLATE, BudgetCase, InkTokenizer, ThinkingEngine  # noqa: E402


class InklingCase(BudgetCase):
    def start(self, thought=None):
        super().start(thought)
        self.svc.set_family("inkling2")
        self.addCleanup(self.svc.set_family, None)

    def budget(self, **req):
        return self.svc.reasoning_budget(req)


class Cap(InklingCase):

    def test_a_big_client_budget_is_the_levels_default(self):
        self.start()
        for level, want in (("low", 2048), ("medium", 4096), ("high", 8192), ("xhigh", 16384), ("max", 16384)):
            self.assertEqual(self.budget(reasoning_budget_tokens=31999, strata_effort=level), want, level)

    def test_a_smaller_client_budget_stays(self):
        self.start()
        self.assertEqual(self.budget(reasoning_budget_tokens=1024, strata_effort="low"), 1024)
        self.assertEqual(self.budget(reasoning_budget_tokens=3000, strata_effort="medium"), 3000)
        self.assertEqual(self.budget(reasoning_budget_tokens=8192, strata_effort="high"), 8192)
        self.assertEqual(self.budget(reasoning_budget_tokens=48), 48)

    def test_every_name_a_client_can_send_is_capped(self):
        self.start()
        for name in ("reasoning_budget_tokens", "thinking_token_budget", "thinking_budget", "thinking_budget_tokens"):
            self.assertEqual(self.budget(**{name: 20000, "strata_effort": "high"}), 8192, name)

    def test_a_request_zero_still_means_no_budget(self):
        self.start()
        self.assertIsNone(self.budget(reasoning_budget_tokens=0, strata_effort="high"))

    def test_the_opt_out_keeps_the_full_budget(self):
        self.start()
        self.svc.inkling_budget_cap = False
        self.assertEqual(self.budget(reasoning_budget_tokens=31999, strata_effort="high"), 31999)

    def test_the_environment_opt_out(self):
        for text, want in (("0", False), ("false", False), ("off", False), ("1", True), ("", True)):
            with mock.patch.dict("os.environ", {"STRATA_INKLING_BUDGET_CAP": text}):
                self.start()
                self.assertEqual(self.svc.inkling_budget_cap, want, text)

    def test_the_one_number_setting_is_the_cap_and_zero_lifts_it(self):
        self.start()
        self.svc.inkling_thinking_budget = 777                       # STRATA_INKLING_THINKING_BUDGET: one default for every level
        self.assertEqual(self.budget(reasoning_budget_tokens=31999, strata_effort="high"), 777)
        self.assertEqual(self.budget(reasoning_budget_tokens=300, strata_effort="high"), 300)
        self.svc.inkling_thinking_budget = 0                         # "no default": nothing to cap at
        self.assertEqual(self.budget(reasoning_budget_tokens=31999, strata_effort="high"), 31999)

    def test_the_configs_own_budget_is_not_a_client_budget(self):
        self.start()
        self.svc.reasoning_budget_tokens = 20000                     # the run config's reasoning_budget_tokens
        self.svc.reasoning_budget_configured = True
        self.assertEqual(self.budget(strata_effort="high"), 20000)

    def test_other_families_are_untouched(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve" / "chat_template.jinja"))
        self.assertFalse(svc.inkling)
        self.assertEqual(svc.reasoning_budget({"reasoning_budget_tokens": 31999, "strata_effort": "high"}), 31999)

    def test_the_line_is_said_once_and_only_by_a_real_request(self):
        self.start()
        out = io.StringIO()
        with redirect_stdout(out):
            self.svc.reasoning_budget({"reasoning_budget_tokens": 31999, "strata_effort": "high"})      # validation of a request says nothing
        self.assertEqual(out.getvalue(), "")
        with redirect_stdout(out):
            self.svc.reasoning_budget({"reasoning_budget_tokens": 31999, "strata_effort": "high"}, announce=True)
            self.svc.reasoning_budget({"reasoning_budget_tokens": 31999, "strata_effort": "high"}, announce=True)
        self.assertEqual(out.getvalue().count("[strata] a client asked for a thinking budget of 31999"), 1)
        self.assertIn("at most 8192", out.getvalue())
        self.assertIn("STRATA_INKLING_BUDGET_CAP=0", out.getvalue())


class ClaudeCode(InklingCase):
    """The request Claude Code sends by default, through the frontend and the server's own level step."""

    CC = {"model": "claude-sonnet", "max_tokens": 32000, "stream": True, "messages": [{"role": "user", "content": "hi"}],
          "thinking": {"type": "enabled", "budget_tokens": 31999}}

    def request(self, **extra):
        self.start()
        req = self.svc.with_shared({**self.CC, **extra}, "anthropic")
        _, _, kw = anthropic_to_messages(req, True)
        req["strata_effort"] = inkling_dialect.effort_of(kw)[0]       # what Service.prepare does
        return req

    def test_default_claude_code_is_high_with_8192(self):
        req = self.request()
        self.assertEqual(req["strata_effort"], "high")
        self.assertEqual(req["reasoning_budget_tokens"], 31999)       # what the client asked
        self.assertEqual(self.svc.reasoning_budget(req), 8192)        # what the server allows

    def test_claude_codes_side_calls_keep_theirs(self):
        req = self.request(thinking={"type": "enabled", "budget_tokens": 1024})
        self.assertEqual(self.svc.reasoning_budget(req), 1024)

    def test_openai_max_budget_names(self):
        self.start()
        req = self.svc.with_shared({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "medium", "thinking_budget_tokens": 30000}, "openai")
        _, _, kw = openai_to_messages(req)
        req["strata_effort"] = inkling_dialect.effort_of(kw)[0]
        self.assertEqual(self.svc.reasoning_budget(req), 4096)


class RunsOnTheEngine(BudgetCase):
    """Through Service.run with a scripted model that thinks 12,000 tokens: the cut is at the level's default, not at 31,999."""

    LONG = "Let me think step by step about two plus two. " * 270            # 12,150 bytes

    def run_cc(self, cap=True):
        self.start(self.LONG)
        self.svc.inkling_budget_cap = cap
        return self.run_request(max_new=32000, reasoning_budget_tokens=31999, strata_effort="high")

    def test_capped(self):
        content, _, done = self.run_cc(True)
        self.assertEqual(content, "The answer is 4.")
        self.assertEqual(len(self.engine.passes), 2)                    # cut, wrap-up, answer
        self.assertLess(self.engine.passes[0]["sent"], 8192 + 80)
        self.assertGreaterEqual(self.engine.passes[0]["sent"], 8192)

    def test_opt_out_thinks_it_all(self):
        content, _, done = self.run_cc(False)
        self.assertEqual(content, "The answer is 4.")
        self.assertEqual(len(self.engine.passes), 1)                    # the whole 12,150 bytes, then the answer


if __name__ == "__main__":
    unittest.main()
