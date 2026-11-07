"""Tests for serve/conv_router.py — the per-request cache decision.

The router spends 120 ms to save 5 s, so its failure modes matter: a wrong "off" costs a full re-prefill on the next
repeat, a wrong "on" wastes 120 ms.  These tests pin the bias (keep the cache), the hysteresis (a RUN of single-shot
requests, not one), and the immediate switch back.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve.conv_router import BATCH, CHAT, Router  # noqa: E402


def one(text="Summarise this document."):
    return [{"role": "user", "content": text}]


def chat(turns=2, first="Summarise this document."):
    msgs = [{"role": "user", "content": first}]
    for i in range(turns - 1):
        msgs.append({"role": "assistant", "content": "ok"})
        msgs.append({"role": "user", "content": f"and then? {i}"})
    return msgs


def test_one_single_request_keeps_the_cache():
    r = Router(batch_after=4)
    d = r.decide(one("a"))
    assert d.mode == CHAT and not d.changed, "one single-shot request must never switch the cache off"


def test_a_run_of_single_requests_switches_off():
    r = Router(batch_after=4)
    modes = [r.decide(one(f"prompt {i}")).mode for i in range(4)]
    assert modes == [CHAT, CHAT, CHAT, BATCH], modes
    last = r.decide(one("prompt 5"))
    assert last.mode == BATCH and not last.changed, "already in batch: no repeated switching"


def test_a_conversational_request_switches_back_immediately():
    r = Router(batch_after=3)
    for i in range(3):
        r.decide(one(f"p{i}"))
    assert r.mode == BATCH
    d = r.decide(chat(2))
    assert d.mode == CHAT and d.changed and d.wanted_cache == r.cache_on
    assert r.consecutive_single == 0


def test_the_counter_resets_on_a_conversational_request():
    r = Router(batch_after=3)
    r.decide(one("a"))
    r.decide(one("b"))
    r.decide(chat(2))               # resets
    r.decide(one("c"))
    r.decide(one("d"))
    assert r.mode == CHAT, "two singles after a conversation must not trip a threshold of three"


def test_tools_alone_mean_chat():
    r = Router(batch_after=2)
    d = r.decide(one("look this up"), tools=[{"type": "function", "function": {"name": "f"}}])
    assert d.mode == CHAT and "tools" in d.reason


def test_a_repeat_prefix_means_chat_even_when_single_turn():
    r = Router(batch_after=2)
    r.decide(one("the same document"))
    r.decide(one("another document"))
    assert r.mode == BATCH, "two singles trip the threshold"
    d = r.decide(one("the same document"))     # same first turn as request 1
    assert d.mode == CHAT and "repeat" in d.reason and d.changed


def test_prefix_key_ignores_whitespace_and_tracks_content():
    a = Router.prefix_key(one("hello   world\n"))
    b = Router.prefix_key(one("hello world"))
    c = Router.prefix_key(one("hello worlds"))
    assert a == b and a != c


def test_prefix_key_survives_a_content_list_and_none():
    assert Router.prefix_key([{"role": "user", "content": [{"type": "text", "text": "hi there"}]}])
    assert Router.prefix_key([{"role": "system", "content": "x"}]) == ""
    assert Router.prefix_key(None) == ""


def test_a_refused_switch_rolls_back():
    r = Router(batch_after=2)
    r.decide(one("a"))
    d = r.decide(one("b"))
    assert d.mode == BATCH
    r.revert()                                  # the engine refused the tune
    assert r.mode == CHAT
    d2 = r.decide(one("c"))
    assert d2.mode == CHAT, "after a rollback the next request must not think it is already in batch"


def test_report_counts_every_request():
    r = Router(batch_after=2)
    r.decide(one("a"))
    r.decide(one("b"))
    r.decide(chat(2))
    rep = r.report()
    assert rep["requests"] == 3 and rep["switches_off"] == 1 and rep["switches_on"] == 1
    assert rep["chat"] + rep["batch"] == 3


def test_the_prefix_history_is_bounded():
    r = Router(repeat_window=8)
    for i in range(50):
        r.decide(one(f"prompt {i}"))
    assert len(r.seen) <= 8
