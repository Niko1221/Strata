"""Tests for the thinking-budget rule (Service.apply_thinking_policy).

Run: .venv/bin/python -m pytest serve/test_thinking_policy.py -q

The contract: nothing at all when off; a client's own choice is never overridden; only a prompt that asks for a
short DIRECT answer loses its thinking; a code prompt never does; a classifier failure leaves the request alone.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve import server as S  # noqa: E402

DIRECT = "In the memo below, reply with the invoice number only:\n\nMemo: the invoice is INV-4471, dated Tuesday."
REASON = "Explain why the sky is blue, in one paragraph."
CODE = "Why does this panic?\n\nTraceback (most recent call last):\n  File \"main.py\", line 12\nKeyError: 'retries'"


def service(mode="off"):
    svc = S.Service.__new__(S.Service)
    svc.thinking_policy = mode
    svc.thinking_stats = {"direct": 0, "reason": 0, "applied": 0, "shadowed": 0, "last": None}
    return svc


def messages(text):
    return [{"role": "user", "content": text}]


def run(svc, text, kwargs=None):
    kw = dict(kwargs or {})
    S.Service.apply_thinking_policy(svc, messages(text), kw)
    return kw


def test_off_changes_nothing():
    svc = service("off")
    assert run(svc, DIRECT) == {}
    assert svc.thinking_stats["direct"] == 0


def test_shadow_counts_without_touching_the_request():
    svc = service("shadow")
    assert run(svc, DIRECT) == {}
    assert svc.thinking_stats["direct"] == 1 and svc.thinking_stats["shadowed"] == 1


def test_on_turns_thinking_off_for_a_direct_request():
    svc = service("on")
    kw = run(svc, DIRECT)
    assert kw == {"enable_thinking": False}
    assert svc.thinking_stats["applied"] == 1


def test_a_reasoning_request_keeps_its_thinking():
    svc = service("on")
    assert run(svc, REASON) == {}
    assert svc.thinking_stats["reason"] == 1 and svc.thinking_stats["applied"] == 0


def test_a_code_prompt_keeps_its_thinking():
    svc = service("on")
    assert run(svc, CODE) == {}, S.prompt_kind.wants_direct_answer(CODE)


def test_a_clients_own_choice_is_never_overridden():
    svc = service("on")
    assert run(svc, DIRECT, {"reasoning_effort": "high"}) == {"reasoning_effort": "high"}
    assert run(svc, DIRECT, {"enable_thinking": False}) == {"enable_thinking": False}
    assert svc.thinking_stats["applied"] == 0


def test_dropping_a_reasoning_effort_when_the_rule_fires():
    """A request that carried some default effort still gets thinking off - but only if the effort was not the
    client's (that case returns above).  Here the default came from the config, so the rule owns it."""
    svc = service("on")
    kw = {"reasoning_effort": "medium"}
    S.Service.apply_thinking_policy(svc, messages(DIRECT), kw)
    assert kw == {"reasoning_effort": "medium"}          # conservative: an effort present is treated as the client's


def test_a_classifier_failure_leaves_the_request_alone():
    svc = service("on")

    class Boom:
        def __iter__(self):
            raise RuntimeError("no")

    messages_bad = Boom()
    kw: dict = {}
    S.Service.apply_thinking_policy(svc, messages_bad, kw)
    assert kw == {}


def test_report_shape():
    svc = service("shadow")
    run(svc, DIRECT)
    rep = S.Service.thinking_report(svc)
    assert rep["mode"] == "shadow" and rep["direct"] == 1 and "last" not in rep


# ---- the compute guard: the class of prompts that must NEVER lose their thinking -------------------------------
# Added after a measured accuracy campaign: without it, "Is 91 divisible by 7?" answered "No" (it is 13x7) and
# "Convert 2048 MiB to GiB" answered 1.9999999999999998. These are the prompts that must keep the scratchpad.
COMPUTE = [
    "Is 91 divisible by 7? Reply yes or no.",
    "Convert 2048 MiB to GiB. Reply with the number only.",
    "What percent of 250 is 40? Reply with the number only.",
    "What is the average of 4, 9 and 17? Reply with the number only.",
    "Which is larger, 0.8 or 0.75? Reply with the larger number only.",
    "Reply with only the largest value: 3, 41, 12, 9",
    "Return only one CSV line with these fields in order: host=buildbox port=8095 gpu=B70",
    "Add these and reply with only the number: 17 + 25 + 8",
]


def test_compute_questions_are_never_suppressed():
    svc = service("on")
    for text in COMPUTE:
        kw = run(svc, text)
        assert "enable_thinking" not in kw, f"suppressed a compute question: {text!r} -> {kw}"


def test_a_date_with_dashes_is_still_a_direct_answer():
    """The guard deliberately does not treat a bare '-' as arithmetic, or every date would keep its thinking."""
    svc = service("on")
    kw = run(svc, "Reply with only the date.\n\nPublished on 2026-11-04.")
    assert kw.get("enable_thinking") is False


def test_a_server_filled_effort_does_not_disable_the_rule():
    """The bug this guards: with_shared() writes reasoning_effort into the request before the policy runs, and the
    policy read it as the client's own choice, so it silently never fired. `_effort_from_shared` marks the server's
    default."""
    svc = service("on")
    kw = {}
    S.Service.apply_thinking_policy(svc, messages(DIRECT), kw, req={"reasoning_effort": "medium",
                                                                  "_effort_from_shared": True})
    assert kw.get("enable_thinking") is False
    assert "reasoning_effort" not in kw


def test_a_client_effort_in_the_request_is_respected():
    svc = service("on")
    kw = {}
    S.Service.apply_thinking_policy(svc, messages(DIRECT), kw, req={"reasoning_effort": "high"})
    assert "enable_thinking" not in kw
