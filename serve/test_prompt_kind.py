"""Tests for serve/prompt_kind.py - the label a per-request retune keys on.

Run: .venv/bin/python -m pytest serve/test_prompt_kind.py -q
The cases are realistic prompts, not strings built to satisfy the regexes: a classifier that only passes its
own examples would move the draft floor on the wrong requests, which is worse than not moving it at all.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve import prompt_kind as PK  # noqa: E402

CODE_PROMPTS = [
    "Write a Python function that reverses a singly linked list in place.",
    "fix the failing test in src/parser.py",
    "Refactor this class so the retry logic lives in one method:\n\n"
    "```python\nclass Client:\n    def get(self, url):\n        for i in range(3):\n            try:\n"
    "                return self._do(url)\n            except TimeoutError:\n                continue\n```",
    "Why does this panic?\n\nTraceback (most recent call last):\n  File \"main.py\", line 12, in <module>\n"
    "    run(cfg[\"retries\"])\nKeyError: 'retries'",
    "Add a --dry-run flag to the deploy script and a test for it.",
    "diff --git a/serve/server.py b/serve/server.py\n--- a/serve/server.py\n+++ b/serve/server.py\n@@ -1,3 +1,4 @@\n"
    " x = 1\n+y = 2",
    "SELECT customer_id, SUM(total) FROM orders GROUP BY customer_id HAVING SUM(total) > 1000;"
    " -- make this faster for 40M rows",
    "Implement a rate limiter middleware for our express app, token bucket, 100 req/min per key.",
    "the build fails with `cannot find -lmkl_sequential` when I run ninja -C build-sycl-aot; what do I change in CMakeLists?",
    "Convert this bash snippet to Python:\n\nfor f in *.gguf; do echo \"$f\"; done",
]

PROSE_PROMPTS = [
    "Explain why the sky is blue.",
    "Summarize the plot of Hamlet in three sentences.",
    "What are the main trade-offs between consistency and availability in distributed systems, and why does the "
    "choice depend on the workload? Please answer in a paragraph or two.",
    "Write a short email to a customer explaining that their refund will take five business days.",
    "Compare the tone of the two reviews and say which one is more persuasive.",
    "Plan a five-day trip to Lisbon for someone who likes food markets and walking, with one day trip.",
    "Why do some people prefer analog photography when digital is cheaper and more convenient? Consider the "
    "craft, the workflow, and the look of the images, and explain how those factors interact.",
    "Brainstorm names for a small coffee roastery that sells only single-origin beans.",
]


def test_code_prompts_are_code():
    wrong = [(p[:40], str(PK.classify(p))) for p in CODE_PROMPTS if PK.classify(p).kind != PK.CODE]
    assert not wrong, wrong


def test_prose_prompts_are_prose():
    wrong = [(p[:40], str(PK.classify(p))) for p in PROSE_PROMPTS if PK.classify(p).kind != PK.PROSE]
    assert not wrong, wrong


def test_empty_is_prose():
    for text in ("", "   \n\n ", None):
        assert PK.classify(text or "").kind == PK.PROSE


def test_fence_alone_is_code():
    text = "What does this do?\n```\nSELECT 1;\n```"
    assert PK.classify(text).kind == PK.CODE


def test_message_list_uses_the_last_user_turn():
    msgs = [{"role": "user", "content": "Write a Rust function that dedups a Vec<u32>."},
            {"role": "assistant", "content": "Sure - here it is."},
            {"role": "user", "content": "Now write a test for it."}]
    assert PK.classify_messages(msgs).kind == PK.CODE


def test_continued_question_keeps_the_kind_of_what_precedes_it():
    code = "def dedup(xs):\n    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out"
    msgs = [{"role": "user", "content": "Write a Python dedup function.\n" + code},
            {"role": "assistant", "content": "Here it is."},
            {"role": "user", "content": "Does it keep the original order? Explain your reasoning briefly."}]
    assert PK.classify_messages(msgs).kind == PK.CODE, PK.classify_messages(msgs)


def test_no_user_turn_is_prose():
    assert PK.classify_messages([{"role": "system", "content": "be brief"}]).kind == PK.PROSE
    assert PK.classify_messages([]).kind == PK.PROSE


def test_image_parts_carry_no_text():
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]
    assert PK.classify_messages(msgs).kind == PK.PROSE


def test_reasons_are_reported_and_signed():
    lab = PK.classify("fix the bug in main.go")
    assert lab.kind == PK.CODE
    assert lab.reasons and lab.score > 0
    assert str(lab).startswith("code")
    lab = PK.classify("Describe the smell of rain.")
    assert lab.kind == PK.PROSE and lab.score < PK.THRESHOLD


def test_prose_about_code_is_still_prose_when_nothing_is_asked():
    text = ("I have been reading about how compilers allocate registers, and I find the graph colouring "
            "approach beautiful, though I do not fully understand why spilling is unavoidable in some cases "
            "and what the trade-offs are for a large function with many live values.")
    assert PK.classify(text).kind == PK.PROSE, PK.classify(text)

class MeasuredFailures(unittest.TestCase):
    """The three prompts the accuracy campaign caught the suppressed arm answering WRONG (2026-11-04).

    Each asks so directly that the wording alone reads "short answer", and each needs the scratchpad to be right:
    the gate must keep their thinking. They are here so it cannot be relaxed back.
    """

    def test_divisibility_keeps_thinking(self):
        self.assertEqual(PK.wants_direct_answer("Is 91 divisible by 7? Reply yes or no.").kind, "reason")

    def test_unit_conversion_keeps_thinking(self):
        self.assertEqual(PK.wants_direct_answer(
            "Convert 2048 MiB to GiB. Reply with the number only.").kind, "reason")

    def test_field_order_keeps_thinking(self):
        self.assertEqual(PK.wants_direct_answer(
            "Return only one CSV line with these fields in order: host=buildbox port=8095 gpu=B70").kind, "reason")

    def test_plain_extraction_is_still_suppressed(self):
        for text in ("Reply with only the invoice number.\n\nMemo: INV-4471 was issued on Tuesday.",
                     "Reply with only the port.\n\nconfig: host=192.0.2.4 port=8095 retries=3",
                     "Reply with only the email address.\n\nContact: Sam Rivera, someone@example.com."):
            self.assertEqual(PK.wants_direct_answer(text).kind, "direct", text[:40])


if __name__ == "__main__":
    unittest.main()


def test_classify_does_not_raise_on_junk():
    """The docstring promises it never raises; a caller passing None or a number must get prose, not a TypeError."""
    for junk in (None, 123, b"bytes", [], {}):
        got = PK.classify(junk)
        assert got.kind == PK.PROSE, junk
        assert got.score == 0.0, junk


def test_classify_messages_agrees_with_classify_of_messages_text():
    """The two public entry points share one reader; this is what proves it."""
    cases = [
        [{"role": "user", "content": "def f(x):\n    return x + 1\nWrite a unit test for this."}],
        [{"role": "user", "content": "Explain what this does"}],
        [{"role": "user", "content": "def parse(s):\n    return s.split(',')"},
         {"role": "assistant", "content": "It splits a line."},
         {"role": "user", "content": "Now make it handle quotes"}],
        [{"role": "system", "content": "you are terse"}],
        [],
    ]
    for messages in cases:
        a = PK.classify_messages(messages)
        b = PK.classify(PK.messages_text(messages))
        assert a.kind == b.kind, messages
        assert a.score == b.score, messages


def test_messages_without_a_user_turn_say_why():
    got = PK.classify_messages([{"role": "system", "content": "hi"}])
    assert got.kind == PK.PROSE
    assert "no user turn" in got.reasons
    assert PK.messages_text([{"role": "system", "content": "hi"}]) == ""
