"""Local validation of PR #3971's thinking-budget drain fix on frontend 9780a8c2.

The native transport tests use the real solo generator; no GPU/model is required.
"""
import io
import json
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from serve import server
from serve import test_server as fixtures


class SoloDrain(unittest.TestCase):
    def engine(self, *lines):
        e = server.StrataEngine.__new__(server.StrataEngine)
        e.proc = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
        e.lines, e.can_stop, e.max_context, e.silence_s = queue.Queue(), True, 65536, 0
        for line in lines:
            e.lines.put(line)
        return e

    def cut(self, e, sampling=None):
        gen = e.generate([1], 20, sampling or {}, threading.Event())
        self.assertEqual(next(gen), 72)
        gen.close()
        self.assertIn("STOP\n", e.proc.stdin.getvalue())

    def test_stop_drain_keeps_only_tokens_in_its_acknowledged_done(self):
        e = self.engine("T 72", "T 105", "LP -0.2", "T 33", "DONE 3 1 0 3 cancel 0 0 0")
        self.cut(e)
        self.assertEqual(e.last.get("_thinking_overrun"), (105, 33))
        self.assertEqual(e.last["generated"], 3)  # keep native work accounting, do not count twice

    def test_logprob_lookahead_token_is_not_lost(self):
        # A native engine without LP support leaves the next T in the generator's carry slot.
        e = self.engine("T 72", "T 105", "T 33", "DONE 3 1 0 3 stop 0 0 0")
        self.cut(e, {"logprobs": True})
        self.assertEqual(e.last.get("_thinking_overrun"), (105, 33))

    def test_failed_or_missing_done_does_not_publish_partial_drain(self):
        for end in (None, "ERR failed", "DONE 3 1 0 3 nonfinite 0 0 0"):
            with self.subTest(end=end):
                e = self.engine("T 72", "T 105", end)
                self.cut(e)
                self.assertNotIn("_thinking_overrun", e.last)

    def test_natural_next_done_clears_previous_drain_metadata(self):
        e = self.engine("T 72", "T 105", "DONE 2 1 0 2 stop 0 0 0")
        self.cut(e)
        previous = e.last
        e.lines.put("T 89")
        e.lines.put("DONE 1 1 0 1 length 0 0 0")
        self.assertEqual(list(e.generate([2], 1, {}, threading.Event())), [89])
        self.assertNotIn("_thinking_overrun", e.last)
        self.assertEqual(previous.get("_thinking_overrun"), (105,))

    def test_malformed_token_or_replaced_process_prevents_carry(self):
        for replaced in (False, True):
            with self.subTest(replaced=replaced):
                e = self.engine("T 72", "T 105", "T 33" if replaced else "T invalid",
                                "DONE 3 1 0 3 stop 0 0 0")
                gen = e.generate([1], 20, {}, threading.Event())
                self.assertEqual(next(gen), 72)
                if replaced:
                    e.proc = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
                gen.close()
                self.assertNotIn("_thinking_overrun", e.last)

    def test_native_multi_drain_keeps_existing_behavior(self):
        e = self.engine()
        e.multi = e.batch = 2
        e.rid_lock, e.rid_n, e.rid_q = threading.Lock(), 0, {}
        def send(line):
            q = next(iter(e.rid_q.values()))
            for reply in ("T 72", "T 105", "DONE 2 1 0 2 stop 0 0 0"):
                q.put(reply)
        e._send = send
        gen = e.generate([1], 20, {}, threading.Event())
        self.assertEqual(next(gen), 72)
        gen.close()
        self.assertNotIn("_thinking_overrun", e.last)
        self.assertFalse(e.rid_q)

    def test_timeout_does_not_publish_partial_drain(self):
        e = self.engine("T 72", "T 105")
        e.silence_s = 0.01
        e._silent = lambda why: server.EngineSilent(why)
        gen = e.generate([1], 20, {}, threading.Event())
        self.assertEqual(next(gen), 72)
        with self.assertRaises(server.EngineSilent):
            gen.close()
        self.assertNotIn("_thinking_overrun", e.last)

    def test_external_cancellation_does_not_publish_carry(self):
        e = self.engine("T 72", "T 105", "DONE 2 1 0 2 cancel 0 0 0")
        cancel = threading.Event()
        gen = e.generate([1], 20, {}, cancel)
        self.assertEqual(next(gen), 72)
        cancel.set()
        gen.close()
        self.assertNotIn("_thinking_overrun", e.last)

    def test_batch_delegate_does_not_get_solo_drain_metadata(self):
        e = self.engine()
        e.batch = 1
        e.generate_batched = mock.Mock(return_value=iter((72, 105)))
        self.assertEqual(list(e.generate([1], 20, {}, threading.Event())), [72, 105])
        e.generate_batched.assert_called_once()
        self.assertNotIn("_thinking_overrun", e.last)


class InFlightThinkingEngine(fixtures.ThinkingEngine):
    """A STOP completes an in-flight native window; its last output is not committed."""
    IN_FLIGHT = 2
    CANCEL_AFTER_CUT = False

    def __init__(self, tok):
        super().__init__(tok)
        self.limits = []
        self.held_after_cut = None

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        self.limits.append(max_new)
        prompt = self.tok.decode(ids)
        done = prompt.endswith(("</think>\n\n", "<function=search>\n", "<tool_call>\n"))
        if prompt.endswith("<function=search>\n"):
            text = fixtures.CallingEngine.ARGS
        elif prompt.endswith("<tool_call>\n"):
            text = "<function=search>\n" + fixtures.CallingEngine.ARGS
        else:
            text = self.ANSWER if done else self.THOUGHT + "</think>\n\n" + self.ANSWER
        toks = (self.tok.encode(text, parse_special=True) + self.tok.encode("<|im_end|>", parse_special=True))[:max_new]
        sent = 0
        try:
            for t in toks:
                if cancel.is_set():
                    return
                sent += 1
                yield t
        finally:
            over = tuple(toks[sent:sent + self.IN_FLIGHT])
            self.last = {"generated": sent + len(over), "prompt_tokens": len(ids), "prompt_ms": 1.,
                         "decode_ms": 1., "finish": "stop" if done else "cancel", "reused": 0,
                         "_thinking_overrun": over}
            if not done:
                self.held_after_cut = list(ids) + toks[:max(0, sent + len(over) - 1)]
                if self.CANCEL_AFTER_CUT:
                    cancel.set()


class BudgetContinuation(unittest.TestCase):
    def setUp(self):
        self.h = fixtures.ThinkingBudget()
        self.h.setUp()
        self.e = InFlightThinkingEngine(self.h.tok)
        self.h.engine = self.h.svc.engine = self.e

    def tearDown(self):
        self.h.tearDown()

    def test_zero_one_and_multiple_in_flight_tokens_keep_the_live_prefix(self):
        for count in (0, 1, 2, 3):
            with self.subTest(count=count):
                self.e.IN_FLIGHT = count
                self.e.prompts.clear()
                code, b = self.h.openai(reasoning_budget_tokens=20)
                self.assertEqual(code, 200, b)
                first, second = self.e.prompts
                extra = self.h.tok.encode(server.REASONING_WRAP_UP, parse_special=True)
                self.assertEqual(second, first + self.h.tok.encode(self.e.THOUGHT[:20 + count]) + extra)
                self.assertEqual(second[:len(self.e.held_after_cut)], self.e.held_after_cut)
                self.assertEqual(b["usage"]["completion_tokens"], 20 + count + len(extra) + len(self.e.ANSWER) + 1)
                self.assertEqual(b["usage"]["prompt_tokens"], len(first))
                self.assertEqual(b["choices"][0]["message"]["content"], self.e.ANSWER)

    def test_unsafe_tag_stop_or_split_character_keeps_the_old_cut(self):
        for suffix, count in (("<x", 2), ("<|im_end|>", 1), ("€", 2)):
            with self.subTest(suffix=suffix):
                self.e.IN_FLIGHT = count
                self.e.THOUGHT = "x" * 20 + suffix + "y" * 30
                self.e.prompts.clear()
                code, b = self.h.openai(reasoning_budget_tokens=20)
                self.assertEqual(code, 200, b)
                first, second = self.e.prompts
                extra = self.h.tok.encode(server.REASONING_WRAP_UP, parse_special=True)
                self.assertEqual(second, first + self.h.tok.encode("x" * 20) + extra)

    def test_external_cancellation_at_budget_cut_does_not_continue(self):
        self.e.CANCEL_AFTER_CUT = True
        code, b = self.h.openai(reasoning_budget_tokens=20)
        self.assertEqual(code, 200, b)
        self.assertEqual(len(self.e.prompts), 1)
        self.assertEqual(b["usage"]["completion_tokens"], 20)
        self.assertFalse(b["choices"][0]["message"]["content"])

    def test_exact_one_answer_token_fit_and_no_room_for_carry(self):
        extra = self.h.tok.encode(server.REASONING_WRAP_UP, parse_special=True)
        for carry in (True, False):
            with self.subTest(carry=carry):
                self.e.prompts.clear()
                cap = 20 + len(extra) + 1 + (self.e.IN_FLIGHT if carry else 0)
                code, b = self.h.openai(reasoning_budget_tokens=20, max_tokens=cap)
                self.assertEqual(code, 200, b)
                self.assertEqual(b["usage"]["completion_tokens"], cap)
                self.assertEqual(b["choices"][0]["message"]["content"], self.e.ANSWER[:1])
                first, second = self.e.prompts
                k = self.e.IN_FLIGHT if carry else 0
                self.assertEqual(second, first + self.h.tok.encode(self.e.THOUGHT[:20 + k]) + extra)

    def test_65536_context_keeps_every_native_leg_within_the_original_cap(self):
        self.e.max_context = 65536
        code, b = self.h.openai(reasoning_budget_tokens=100, max_tokens=600,
                                messages=[{"role": "user", "content": "q" * 65000}])
        self.assertEqual(code, 200, b)
        self.assertEqual(len(self.e.prompts), 2)
        bounds = [len(ids) + cap + server.CTX_SLACK for ids, cap in zip(self.e.prompts, self.e.limits)]
        self.assertEqual(bounds[0], 65536)
        self.assertEqual(bounds, [bounds[0]] * len(bounds))
        self.assertLessEqual(b["usage"]["total_tokens"] + server.CTX_SLACK, 65536)

    def test_streamed_reasoning_and_forced_tool_after_the_wrap(self):
        code, raw = self.h.openai(reasoning_budget_tokens=20, stream=True,
                                  stream_options={"include_usage": True},
                                  tools=fixtures.ForcedToolChoice.TOOLS,
                                  tool_choice=fixtures.ForcedToolChoice.NAMED)
        self.assertEqual(code, 200, raw)
        chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
        deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
        thinking = "".join(d.get("reasoning_content", "") for d in deltas)
        self.assertEqual(thinking, self.e.THOUGHT[:22] + server.REASONING_WRAP_UP.split("</think>")[0])
        calls = [tc for d in deltas for tc in d.get("tool_calls", [])]
        self.assertEqual(next(tc["function"]["name"] for tc in calls if tc["function"].get("name")), "search")
        self.assertEqual(json.loads("".join(tc["function"].get("arguments", "") for tc in calls)), {"q": "2+2"})
        self.assertTrue(any(c.get("choices") and c["choices"][0].get("finish_reason") == "tool_calls" for c in chunks))
        usage = next(c["usage"] for c in reversed(chunks) if c.get("usage"))
        self.assertLessEqual(usage["completion_tokens"], 400)
        self.assertEqual(self.e.prompts[1][:len(self.e.held_after_cut)], self.e.held_after_cut)


if __name__ == "__main__":
    unittest.main()
