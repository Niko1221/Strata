#!/usr/bin/env python3
"""The #728 loop guard's two extensions, without a GPU or a model:

  python3 tools/loop_guard_test.py

A: focused_recovery_prompt splices the medium effort sentence to low as well as the xhigh one, and the xhigh
   splice is byte-identical to the old single-sentence algorithm.
B: reasoning_sentence_run counts the run of identical sentences at the end of the reasoning: it fires on the
   production shape (one ~20-word sentence repeated) while the coverage look is still under its line, and stays
   silent on short code pieces, numbered enumerations and interleaved text.

Run from the repository root.
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serve.server import (EFFORT_TEXT, HIGH_EFFORT, LOW_EFFORT, IM_END, LOOP_SENTENCE_RUN,
                          focused_recovery_prompt, reasoning_repeat_coverage, reasoning_sentence_run)

SPECIAL = 0xE000


class FakeTok:
    """One id per character; IM_END is the single special token, as the real tokenizer's special ids are."""

    def encode(self, s, parse_special=False):
        if s == IM_END:
            return [SPECIAL]
        ids, parts = [], s.split(IM_END)
        for k, piece in enumerate(parts):
            ids.extend(ord(c) for c in piece)
            if k < len(parts) - 1:
                ids.append(SPECIAL)
        return ids

    def decode(self, ids):
        return "".join(IM_END if i == SPECIAL else chr(i) for i in ids)


def contains(hay, needle):
    """needle (a list of ids) appears in hay (a list of ids) as a run."""
    n = len(needle)
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def check(name, got, want):
    assert got == want, f"{name}: got {got!r}, want {want!r}"
    print(f"  ok  {name}")


tok = FakeTok()
GEN = [ord(c) for c in "thought so far"]


def prompt_with(sentence):
    body = "<|im_start|>system\n" + sentence + " Follow the task.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"
    return tok.encode(body)


# A: the splice.

old_xhigh = prompt_with(HIGH_EFFORT)
spliced = focused_recovery_prompt(tok, old_xhigh, GEN)
low_ids = tok.encode(LOW_EFFORT)
high_ids = tok.encode(HIGH_EFFORT)
check("xhigh: the low sentence is in the spliced prompt", contains(spliced, low_ids), True)
check("xhigh: the spliced head no longer carries the xhigh sentence",
      contains(spliced[:spliced.index(SPECIAL)], high_ids), False)
# byte-identical to the old algorithm: the same ids with the one xhigh occurrence replaced, generated appended
i = old_xhigh.index(high_ids[0])
check("xhigh: byte-identical to the old single-sentence splice",
      spliced, old_xhigh[:i] + low_ids + old_xhigh[i + len(high_ids):] + GEN)

medium_ids = tok.encode(EFFORT_TEXT["medium"])
spliced_m = focused_recovery_prompt(tok, prompt_with(EFFORT_TEXT["medium"]), GEN)
check("medium: now splices to low (the old code returned None)",
      spliced_m is not None and contains(spliced_m, low_ids)
      and not contains(spliced_m[:spliced_m.index(SPECIAL)], medium_ids), True)

check("low: splicing low to itself is a no-op, None", focused_recovery_prompt(tok, prompt_with(EFFORT_TEXT["low"]), GEN), None)
check("no effort sentence: None", focused_recovery_prompt(tok, prompt_with("Just a plain system instruction."), GEN), None)

# the xhigh sentence in the user turn, not the system one: the head stops at the first im_end, so None.
tricky = ("<|im_start|>system\nplain.<|im_end|>\n<|im_start|>user\n" + HIGH_EFFORT + "<|im_end|>\n")
check("user text never counts: None", focused_recovery_prompt(tok, tok.encode(tricky), GEN), None)

# B: the sentence run.

SENT = ("let me check the current state of the port and see whether the model is already running "
        "since the user says the line switched")
check("the production shape: 15 in a row", reasoning_sentence_run((SENT + ". ") * 15), 15)
check("4 in a row stays under the line", reasoning_sentence_run((SENT + ". ") * 4) >= LOOP_SENTENCE_RUN, False)
check("15 in a row crosses it", reasoning_sentence_run((SENT + ". ") * 15) >= LOOP_SENTENCE_RUN, True)
check("interleaved text is not a run", reasoning_sentence_run((SENT + ". different work here. ") * 8), 0)
check("case and spacing variants still count as one", reasoning_sentence_run((SENT.upper() + ".\n") * 6), 6)
check("a short code tail stops the count", reasoning_sentence_run((SENT + ". ") * 9 + " }"), 0)
check("numbered enumerations differ after normalizing",
      reasoning_sentence_run("".join(f"Step {k}: now we verify the previous result carefully. " for k in range(1, 9))), 1)
check("a short English sentence is under the length gate", reasoning_sentence_run(("do it again now. ") * 12), 0)

CJK = "让我检查一下当前端口上服务的模型是否已经加载完成然后再决定下一步要做什么事情比较合适"
check("a Chinese sentence run counts by characters", reasoning_sentence_run((CJK + "。") * 6), 6)

# the coverage look still works (regression): a dense passage loop crosses its line.
passage = "the model should now answer with the final result and stop thinking about it"
dense = " ".join(f"filler word number {k}" for k in range(1500)) + ". " + (passage + ". ") * 60
check("coverage regression: a dense loop still fires", reasoning_repeat_coverage(dense) >= 0.25, True)

print("loop_guard_test: all ok")
