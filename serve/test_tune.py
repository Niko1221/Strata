"""Tests for the in-place retune: the TUNE line, its reply, and what the cache router does when it cannot get it.

The engine's side is a line protocol over the serve pipe (handle_tune in src/program/generate.cpp): `TUNE k=v ...`
between requests, answered `TUNED k=v ...` with a `refused:` tail for keys it will not take.  These tests pin the
client's half - the key spelling, the value types, the three ways the engine can answer - and the router's
behaviour when the FIFO is busy, which must be a skipped switch rather than a held request.
"""
from __future__ import annotations

import io
import queue
import unittest

from serve.server import EngineDied, ModelBusy, Service, StrataEngine
from serve.conv_router import CHAT, BATCH


class FakeProc:
    def __init__(self):
        self.stdin = io.StringIO()

    def written(self) -> str:
        return self.stdin.getvalue()


class FakeEngine:
    """Just enough of StrataEngine: alive(), proc.stdin, the line queue."""

    batch = 0
    ctl = None

    def __init__(self, reply: str | None = "TUNED prompt_cache=0"):
        self.proc = FakeProc()
        self.lines: queue.Queue = queue.Queue()
        if reply is not None:
            self.lines.put(reply)

    def alive(self) -> bool:
        return True


tune = StrataEngine.tune            # unbound: the stub supplies self


class TuneLine(unittest.TestCase):
    def test_a_reply_is_parsed_into_applied_values(self):
        eng = FakeEngine("TUNED prompt_cache=0 spec_min_p=0.4 short_read=512")
        out = tune(eng, {"prompt_cache": 0, "spec_min_p": 0.4, "short_read": 512})
        self.assertEqual(out["applied"], {"prompt_cache": 0, "spec_min_p": 0.4, "short_read": 512})
        self.assertEqual(out["refused"], [])

    def test_the_line_spells_keys_the_engine_way(self):
        eng = FakeEngine()
        tune(eng, {"spec-min-p": 0.3, "--short-read": 256, "keep": True})
        self.assertEqual(eng.proc.written(), "TUNE spec_min_p=0.3 short_read=256 keep=1\n")

    def test_refused_keys_are_reported_not_raised(self):
        eng = FakeEngine("TUNED adapt_every=8 refused: spec=4(unknown key, or a value it cannot take)")
        out = tune(eng, {"adapt_every": 8, "spec": 4})
        self.assertEqual(out["applied"], {"adapt_every": 8})
        self.assertEqual(out["refused"], ["spec=4"])

    def test_an_error_answer_is_a_value_error(self):
        eng = FakeEngine("ERR the engine is not serving")
        with self.assertRaises(ValueError):
            tune(eng, {"prompt_cache": 0})

    def test_an_engine_that_never_answers_is_reported_as_died(self):
        eng = FakeEngine(reply=None)
        with self.assertRaises(EngineDied):
            tune(eng, {"prompt_cache": 0}, timeout=0.2)

    def test_no_settings_is_refused_before_touching_the_pipe(self):
        eng = FakeEngine()
        with self.assertRaises(ValueError):
            tune(eng, {})
        self.assertEqual(eng.proc.written(), "")


class Stub:
    """The pieces of Service that conv_route_decide touches."""
    def __init__(self, tune_raises=None):
        self.conv_route = "on"
        from serve.conv_router import Router
        self.conv_router = Router(cache_on=6, batch_after=2)
        self.conv_route_stats = {"switches": 0, "refused": 0, "last": ""}
        self.tune_raises = tune_raises
        self.tuned = []

    def tune(self, settings, wait_s=5.0):        # noqa: ARG002 - the signature is what matters here
        if self.tune_raises is not None:
            raise self.tune_raises
        self.tuned.append(settings)
        return {"applied": dict(settings), "refused": []}


decide = Service.conv_route_decide


def run_of(svc, n, text="Summarise this note", unique=True):
    """`unique` gives every request a different text, which is what single-shot traffic looks like: repeating one
    prompt is a REPEAT (and the router keeps the cache on for it) - see test_a_repeat_brings_the_cache_back."""
    for i in range(n):
        body = f"{text} #{i}" if unique else text
        decide(svc, {"messages": [{"role": "user", "content": body}]})


class CacheRouter(unittest.TestCase):
    def test_a_single_shot_run_switches_the_cache_off_once(self):
        svc = Stub()
        run_of(svc, 4)
        self.assertEqual(svc.tuned, [{"prompt_cache": 0}])      # batch_after=2: switched once, not per request
        self.assertEqual(svc.conv_route_stats["switches"], 1)

    def test_a_busy_engine_is_a_skipped_switch_not_a_held_request(self):
        svc = Stub(tune_raises=ModelBusy("busy"))
        run_of(svc, 4)
        self.assertEqual(svc.conv_route_stats["switches"], 0)
        self.assertGreaterEqual(svc.conv_route_stats["refused"], 1)
        self.assertEqual(svc.conv_router.mode, CHAT)            # and the router does not believe it switched

    def test_shadow_never_calls_the_engine(self):
        svc = Stub()
        svc.conv_route = "shadow"
        run_of(svc, 4)
        self.assertEqual(svc.tuned, [])
        self.assertEqual(svc.conv_route_stats["switches"], 0)

    def test_a_repeat_brings_the_cache_back(self):
        svc = Stub()
        run_of(svc, 3)                                          # a single-shot run: cache off
        self.assertEqual(svc.tuned[-1], {"prompt_cache": 0})
        decide(svc, {"messages": [{"role": "user", "content": "Summarise this note #0"}]})   # a prefix seen before
        self.assertEqual(svc.tuned[-1], {"prompt_cache": 6})
        self.assertEqual(svc.conv_router.mode, CHAT)

    def test_off_does_nothing(self):
        svc = Stub()
        svc.conv_route = "off"
        svc.conv_router = None
        run_of(svc, 4)
        self.assertEqual(svc.tuned, [])


if __name__ == "__main__":
    unittest.main()
