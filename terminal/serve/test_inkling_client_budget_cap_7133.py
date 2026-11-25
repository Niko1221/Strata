"""#7133: Claude Code's DEFAULT thinking budget is held to the default of its level on Inkling-2-Mini; every other budget goes through as the client sent it.

Claude Code's default is `thinking: {type: enabled, budget_tokens: 31999}` on /v1/messages.  That is level high and, since 0.2.21, a budget of 31,999 tokens with max_new
32,000: minutes of hidden thinking a step at 25 to 60 tokens a second.  The level's own default is 8,192 (high).  The frontend marks exactly that request
(`strata_default_thinking_budget`); only a marked request is lowered to the default of the level it runs at.  Any other number (20,000, 32,000, a `reasoning_budget_tokens`
or an alias sent by a client, a budget with the OpenAI route's `reasoning_effort` and `max_completion_tokens`) is used as it is.  The run config's
`inkling_budget_cap: false` (or STRATA_INKLING_BUDGET_CAP=0) keeps the full 31,999; a request's own 0 still means no budget; the config's own budget and other families are untouched.

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
from serve.frontend import CLIENT_DEFAULT_THINKING_BUDGET, ChatTemplate, anthropic_to_messages, openai_to_messages  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402
from serve.test_inkling_reasoning_budget import BudgetCase  # noqa: E402

MARK = "strata_default_thinking_budget"


class InklingCase(BudgetCase):
    def start(self, thought=None):
        super().start(thought)
        self.svc.set_family("inkling2")
        self.addCleanup(self.svc.set_family, None)

    def budget(self, **req):
        return self.svc.reasoning_budget(req)


class Cap(InklingCase):
    def test_the_marked_default_is_the_levels_default(self):
        self.start()
        self.assertEqual(CLIENT_DEFAULT_THINKING_BUDGET, 31999)
        for level, want in (("low", 2048), ("medium", 4096), ("high", 8192), ("xhigh", 16384), ("max", 16384)):
            self.assertEqual(self.budget(reasoning_budget_tokens=31999, strata_effort=level, **{MARK: True}), want, level)

    def test_a_marked_default_below_the_cap_stays(self):
        self.start()
        self.assertEqual(self.budget(reasoning_budget_tokens=1024, strata_effort="high", **{MARK: True}), 1024)

    def test_every_other_budget_goes_through(self):
        self.start()
        for level in ("low", "medium", "high", "max"):
            for n in (20000, 31999, 32000, 5000, 100000):                  # (31,999 not marked: sent as a reasoning_budget_tokens by somebody)
                self.assertEqual(self.budget(reasoning_budget_tokens=n, strata_effort=level), n, (level, n))

    def test_the_aliases_go_through(self):
        self.start()
        for name in ("reasoning_budget_tokens", "thinking_token_budget", "thinking_budget", "thinking_budget_tokens"):
            self.assertEqual(self.budget(**{name: 31999, "strata_effort": "high"}), 31999, name)

    def test_a_request_zero_still_means_no_budget(self):
        self.start()
        self.assertIsNone(self.budget(reasoning_budget_tokens=0, strata_effort="high"))
        self.assertIsNone(self.budget(reasoning_budget_tokens=0, strata_effort="high", **{MARK: True}))

    def test_the_opt_out_keeps_the_full_budget(self):
        self.start()
        self.svc.inkling_budget_cap = False
        self.assertEqual(self.budget(reasoning_budget_tokens=31999, strata_effort="high", **{MARK: True}), 31999)

    def test_the_environment_opt_out(self):
        for text, want in (("0", False), ("false", False), ("off", False), ("1", True), ("", True)):
            with mock.patch.dict("os.environ", {"STRATA_INKLING_BUDGET_CAP": text}):
                self.start()
                self.assertEqual(self.svc.inkling_budget_cap, want, text)

    def test_the_one_number_setting_is_the_cap_and_zero_lifts_it(self):
        self.start()
        self.svc.inkling_thinking_budget = 777                       # STRATA_INKLING_THINKING_BUDGET: one default for every level
        self.assertEqual(self.budget(reasoning_budget_tokens=31999, strata_effort="high", **{MARK: True}), 777)
        self.assertEqual(self.budget(reasoning_budget_tokens=31999, strata_effort="high"), 31999)
        self.svc.inkling_thinking_budget = 0                         # "no default": nothing to cap at
        self.assertEqual(self.budget(reasoning_budget_tokens=31999, strata_effort="high", **{MARK: True}), 31999)

    def test_the_configs_own_budget_is_not_a_client_budget(self):
        self.start()
        self.svc.reasoning_budget_tokens = 20000                     # the run config's reasoning_budget_tokens
        self.svc.reasoning_budget_configured = True
        self.assertEqual(self.budget(strata_effort="high", **{MARK: True}), 20000)

    def test_other_families_are_untouched(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve" / "chat_template.jinja"))
        self.assertFalse(svc.inkling)
        self.assertEqual(svc.reasoning_budget({"reasoning_budget_tokens": 31999, "strata_effort": "high", MARK: True}), 31999)

    def test_the_line_is_said_once_and_only_by_a_real_request(self):
        self.start()
        req = {"reasoning_budget_tokens": 31999, "strata_effort": "high", MARK: True}
        out = io.StringIO()
        with redirect_stdout(out):
            self.svc.reasoning_budget(req)                           # validation of a request says nothing
        self.assertEqual(out.getvalue(), "")
        with redirect_stdout(out):
            self.svc.reasoning_budget(req, announce=True)
            self.svc.reasoning_budget(req, announce=True)
        self.assertEqual(out.getvalue().count("[strata] the client's default thinking budget of 31999"), 1)
        self.assertIn("held to 8192", out.getvalue())
        self.assertIn("STRATA_INKLING_BUDGET_CAP=0", out.getvalue())
        out = io.StringIO()
        self.svc.inkling_budget_cap_said = False
        with redirect_stdout(out):
            self.svc.reasoning_budget({"reasoning_budget_tokens": 20000, "strata_effort": "high"}, announce=True)
        self.assertEqual(out.getvalue(), "")                         # a budget somebody chose: nothing lowered, nothing said


class Requests(InklingCase):
    """What the frontends make of the requests that matter, then what the server allows."""

    BASE = {"model": "claude-sonnet", "max_tokens": 32000, "stream": True, "messages": [{"role": "user", "content": "hi"}]}

    def anthropic(self, **extra):
        self.start()
        req = self.svc.with_shared({**self.BASE, **extra}, "anthropic")
        _, _, kw = anthropic_to_messages(req, True)
        req["strata_effort"] = inkling_dialect.effort_of(kw)[0]       # what Service.prepare does
        return req

    def test_claude_codes_default_is_marked_high_and_held_to_8192(self):
        req = self.anthropic(thinking={"type": "enabled", "budget_tokens": 31999})
        self.assertEqual(req["strata_effort"], "high")
        self.assertEqual(req["reasoning_budget_tokens"], 31999)       # what the client asked
        self.assertIs(req.get(MARK), True)
        self.assertEqual(self.svc.reasoning_budget(req), 8192)        # what the server allows

    def test_the_opt_out_with_the_default_request(self):
        req = self.anthropic(thinking={"type": "enabled", "budget_tokens": 31999})
        self.svc.inkling_budget_cap = False
        self.assertEqual(self.svc.reasoning_budget(req), 31999)

    def test_other_messages_budgets_are_not_marked_and_not_capped(self):
        for n in (20000, 32000, 31998, 8000, 1024):
            req = self.anthropic(thinking={"type": "enabled", "budget_tokens": n})
            self.assertNotIn(MARK, req, n)
            self.assertEqual(self.svc.reasoning_budget(req), n, n)

    def test_a_reasoning_budget_tokens_in_the_request_wins_over_thinking_and_is_not_marked(self):
        req = self.anthropic(thinking={"type": "enabled", "budget_tokens": 31999}, reasoning_budget_tokens=20000)
        self.assertNotIn(MARK, req)
        self.assertEqual(self.svc.reasoning_budget(req), 20000)

    def test_openai_reasoning_effort_high_with_max_completion_tokens(self):
        self.start()
        req = self.svc.with_shared({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "high", "max_completion_tokens": 32000}, "openai")
        _, _, kw = openai_to_messages(req)
        req["strata_effort"] = inkling_dialect.effort_of(kw)[0]
        self.assertEqual(req["strata_effort"], "high")
        self.assertNotIn(MARK, req)
        self.assertEqual(self.svc.reasoning_budget(req), 8192)        # the level's own default (the request names no number); max_completion_tokens is no budget

    def test_openai_explicit_budget_with_effort_high_goes_through(self):
        self.start()
        req = self.svc.with_shared({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "high", "max_completion_tokens": 32000,
                                    "reasoning_budget_tokens": 20000}, "openai")
        _, _, kw = openai_to_messages(req)
        req["strata_effort"] = inkling_dialect.effort_of(kw)[0]
        self.assertEqual(self.svc.reasoning_budget(req), 20000)

    def test_a_client_cannot_make_a_budget_the_default_by_sending_31999_on_the_chat_route(self):
        self.start()
        req = self.svc.with_shared({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "high", "reasoning_budget_tokens": 31999}, "openai")
        openai_to_messages(req)
        req["strata_effort"] = "high"
        self.assertEqual(self.svc.reasoning_budget(req), 31999)


class RunsOnTheEngine(BudgetCase):
    """Through Service.run with a scripted model that thinks 12,150 tokens: the cut is at the level's default only for the marked default."""

    LONG = "Let me think step by step about two plus two. " * 270            # 12,150 bytes

    def run_cc(self, cap=True, marked=True, budget=31999):
        self.start(self.LONG)
        self.svc.inkling_budget_cap = cap
        extra = {MARK: True} if marked else {}
        return self.run_request(max_new=32000, reasoning_budget_tokens=budget, strata_effort="high", **extra)

    def test_the_default_is_cut(self):
        content, _, done = self.run_cc()
        self.assertEqual(content, "The answer is 4.")
        self.assertEqual(len(self.engine.passes), 2)                    # cut, wrap-up, answer
        self.assertLess(self.engine.passes[0]["sent"], 8192 + 80)
        self.assertGreaterEqual(self.engine.passes[0]["sent"], 8192)

    def test_the_opt_out_thinks_it_all(self):
        content, _, done = self.run_cc(cap=False)
        self.assertEqual(content, "The answer is 4.")
        self.assertEqual(len(self.engine.passes), 1)                    # the whole 12,150 bytes, then the answer

    def test_another_budget_thinks_it_all(self):
        content, _, done = self.run_cc(marked=False, budget=20000)
        self.assertEqual(content, "The answer is 4.")
        self.assertEqual(len(self.engine.passes), 1)

    def test_another_budget_is_still_cut_at_itself(self):
        content, _, done = self.run_cc(marked=False, budget=3000)
        self.assertEqual(len(self.engine.passes), 2)
        self.assertLess(self.engine.passes[0]["sent"], 3000 + 80)


if __name__ == "__main__":
    unittest.main()
