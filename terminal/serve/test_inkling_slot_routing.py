"""#6733 / #6781: where the server sends an Inkling-2-Mini request decides whether the engine can reuse the conversation's cache.

The real engine keeps ONE conversation per slot (the whole stream it holds, and the prompt-end checkpoint) and a request that does not
continue it wipes it.  serve/server.py's routing used to look only at the whole held stream (never the checkpoint) and sent every
request that arrived alone to slot 0 (GEN), so a short request between two turns of a chat (a title, a translation: LobeHub, Hermes)
wiped the chat even with `--parallel 2`.  tools/inkling_mock_engine.py --cache-model has the engine's reuse rules; these tests read
`usage.prompt_tokens_details.cached_tokens` of the replies over HTTP, as a client sees it.

    python -m pytest serve/test_inkling_slot_routing.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[1]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

from serve.test_inkling_serve_e2e import _MockServerHarness, _post_json  # noqa: E402

REPLY = ("<|message_model|><|content_text|>Sure, here it is.<|end_message|><|content_model_end_sampling|>")
BIG = "The quick brown fox jumps over the lazy dog. " * 70          # about 3,100 tokens: a long conversation (the toy tokenizer is per byte)


class Harness(_MockServerHarness):
    def __init__(self, tmp, *, parallel, extra=()):
        super().__init__(tmp, [REPLY], parallel=parallel)
        cfg = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        cfg["args"] = cfg["args"] + ["--cache-model", *extra]
        self.cfg_path.write_text(json.dumps(cfg), encoding="utf-8")


class Client:
    def __init__(self, base):
        self.base = base

    def chat(self, messages, **extra):
        r = _post_json(f"{self.base}/v1/chat/completions",
                       {"model": "inkling-2-mini-sq-20", "messages": messages, "max_tokens": 64, "temperature": 0, **extra}, timeout=60)
        u = r["usage"]
        return r["choices"][0]["message"].get("content") or "", u["prompt_tokens"], (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)


def conversation(tag):
    # (the test tokenizer's template has no system role: the long part is the first user message)
    return [{"role": "user", "content": f"[{tag}] " + BIG + " first question"}]


def next_turn(msgs, reply, question):
    return msgs + [{"role": "assistant", "content": reply}, {"role": "user", "content": question}]


class SlotRouting(unittest.TestCase):
    def run_session(self, parallel, between=None, extra=()):
        """turn 1, optionally a request in between, turn 2, turn 3 of one conversation: [(prompt_tokens, cached_tokens), ...]"""
        with tempfile.TemporaryDirectory() as td:
            with Harness(Path(td), parallel=parallel, extra=extra) as h:
                c = Client(f"http://127.0.0.1:{h.port}")
                got = []
                msgs = conversation("A")
                reply, p, k = c.chat(msgs)
                got.append((p, k))
                if between:
                    c.chat(between)
                msgs = next_turn(msgs, reply, "second question")
                reply, p, k = c.chat(msgs)
                got.append((p, k))
                msgs = next_turn(msgs, reply, "third question")
                _, p, k = c.chat(msgs)
                got.append((p, k))
                return got

    def assertReused(self, row, label):
        p, k = row
        self.assertGreaterEqual(k, p - 90, f"{label}: {k} of {p} prompt tokens reused")

    def assertNotReused(self, row, label):
        self.assertEqual(row[1], 0, f"{label}: {row[1]} of {row[0]} reused")

    def test_a_chat_reuses_its_prompt_one_slot_and_two(self):
        for parallel in (1, 2):
            with self.subTest(parallel=parallel):
                t1, t2, t3 = self.run_session(parallel)
                self.assertNotReused(t1, "turn 1")
                self.assertReused(t2, "turn 2")
                self.assertReused(t3, "turn 3")

    def test_a_short_request_between_two_turns_does_not_wipe_the_chat_with_two_slots(self):
        title = [{"role": "user", "content": "Name this conversation in three words."}]
        t1, t2, t3 = self.run_session(2, between=title)
        self.assertReused(t2, "turn 2 after a title request")
        self.assertReused(t3, "turn 3")

    def test_with_one_slot_the_short_request_still_replaces_the_chat(self):
        # the engine has one conversation per slot: with `parallel 1` there is nowhere else for the title (the limit this fix cannot lift)
        title = [{"role": "user", "content": "Name this conversation in three words."}]
        t1, t2, t3 = self.run_session(1, between=title)
        self.assertNotReused(t2, "turn 2 after a title request, one slot")
        self.assertReused(t3, "turn 3 (the chat is back in the slot)")

    def server_log_after(self, parallel, requests):
        with tempfile.TemporaryDirectory() as td:
            with Harness(Path(td), parallel=parallel) as h:
                c = Client(f"http://127.0.0.1:{h.port}")
                msgs = conversation("A")
                reply, _, _ = c.chat(msgs)
                for _ in range(requests):
                    c.chat([{"role": "user", "content": "Name this conversation in three words."}])
                    msgs = next_turn(msgs, reply, "again")
                    reply, _, _ = c.chat(msgs)
                h.proc.terminate()
                out = h.proc.stdout.read()
                h.proc.stdout.close()
        return [x for x in out.splitlines() if "a side request" in x]

    def test_one_slot_says_once_that_a_side_request_replaced_the_chat(self):
        lines = self.server_log_after(1, 3)           # three title requests, each replaced the chat
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("--parallel 2 keeps it", lines[0])
        self.assertIn("[strata]", lines[0])

    def test_two_slots_never_say_it(self):
        self.assertEqual(self.server_log_after(2, 2), [])

    def test_a_helper_request_that_needs_two_passes_stays_out_of_the_chats_slot(self):
        # With thinking on, a short helper request spends its thinking budget, the server ends the thinking and sends a second pass (the prompt
        # plus what it wrote plus the wrap-up).  That pass arrives while the first one's slot is still being released: it must wait for it,
        # not take the one free slot (the chat's) because it is the only one free.
        think = "<|message_model|><|content_thinking|>" + "x" * 300
        scripts = [REPLY, think, "<|message_model|><|content_text|>Title<|end_message|><|content_model_end_sampling|>", REPLY]
        with tempfile.TemporaryDirectory() as td:
            h = Harness(Path(td), parallel=2)
            h.tok = h.tok
            encoded = [h.tok.encode(s, parse_special=True) for s in scripts]
            h.script_path.write_text(json.dumps(encoded), encoding="utf-8")
            with h:
                c = Client(f"http://127.0.0.1:{h.port}")
                msgs = conversation("A")
                reply, p1, _ = c.chat(msgs)
                c.chat([{"role": "user", "content": "Name this conversation in three words."}], max_tokens=24)
                _, p2, k2 = c.chat(next_turn(msgs, reply, "second"))
                self.assertGreaterEqual(k2, p1 - 90, f"turn 2: {k2} of {p2} prompt tokens reused after a two-pass helper request")

    def wrapped_up_chat(self, parallel, extra=()):
        """turn 1 spends its thinking budget (a second pass: the prompt, what it wrote, the wrap-up); turn 2 is what a client sends back
        (the prompt and the reply, no thinking): (p1, turn 2's prompt tokens, its cached tokens, the engine's GEN/BGEN lines)."""
        think = "<|message_model|><|content_thinking|>" + "x" * 300
        scripts = [think, REPLY, REPLY]
        with tempfile.TemporaryDirectory() as td:
            h = Harness(Path(td), parallel=parallel, extra=extra)
            h.script_path.write_text(json.dumps([h.tok.encode(s, parse_special=True) for s in scripts]), encoding="utf-8")
            with h:
                c = Client(f"http://127.0.0.1:{h.port}")
                msgs = conversation("A")
                reply, p1, _ = c.chat(msgs, max_tokens=60)
                _, p2, k2 = c.chat(next_turn(msgs, reply, "second"), max_tokens=60)
                lines = h.record_path.read_text(encoding="utf-8").splitlines()
        return p1, p2, k2, lines

    def test_a_wrap_up_pass_does_not_replace_the_chats_checkpoint(self):
        # turn 2 can only resume from turn 1's prompt-end checkpoint (the cached stream holds the thinking the client never sees);
        # the wrap-up pass has a longer prompt and used to take that checkpoint over: nothing was reused after a wrap-up.
        for parallel in (1, 2):
            with self.subTest(parallel=parallel):
                p1, p2, k2, lines = self.wrapped_up_chat(parallel)
                self.assertEqual(len(lines), 3, lines)                    # turn 1's two passes, turn 2
                self.assertNotIn("ckpt=0", lines[0])                      # the first pass saves its checkpoint
                self.assertIn("ckpt=0", lines[1])                         # the later pass keeps it
                self.assertNotIn("ckpt=0", lines[2])
                self.assertGreaterEqual(k2, p1 - 90, f"turn 2: {k2} of {p2} prompt tokens reused after a wrap-up pass")

    def test_the_env_knob_turns_the_prompt_checkpoint_off_on_every_pass(self):
        # STRATA_INKLING_PROMPT_CKPT=0 (an exactness A/B against an engine that has no checkpoint): no pass saves one
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"STRATA_INKLING_PROMPT_CKPT": "0"}):
            p1, p2, k2, lines = self.wrapped_up_chat(1)
        self.assertEqual(len(lines), 3, lines)
        self.assertTrue(all("ckpt=0" in x for x in lines), lines)
        self.assertEqual(k2, 0)

    def test_an_engine_that_does_not_say_ckpt_key_is_never_sent_the_key(self):
        # (an older engine answers an unknown key with an error): the mock without --cache-model has no ckpt_key in INFO
        with tempfile.TemporaryDirectory() as td:
            think = "<|message_model|><|content_thinking|>" + "x" * 300
            h = Harness(Path(td), parallel=1)
            h.cfg_path.write_text(json.dumps({**json.loads(h.cfg_path.read_text(encoding="utf-8")),
                                              "args": [a for a in json.loads(h.cfg_path.read_text(encoding="utf-8"))["args"] if a != "--cache-model"]}),
                                  encoding="utf-8")
            h.script_path.write_text(json.dumps([h.tok.encode(s, parse_special=True) for s in (think, REPLY)]), encoding="utf-8")
            with h:
                Client(f"http://127.0.0.1:{h.port}").chat(conversation("A"), max_tokens=60)
                lines = h.record_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2, lines)
        self.assertTrue(all("ckpt=" not in x for x in lines), lines)

    def test_the_key_is_only_written_for_the_servers_own_flag(self):
        from serve.server import StrataEngine
        from serve.server_parts.base import KEEP_CKPT_KEY
        self.assertIn(" ckpt=0", StrataEngine.sampling_keys({KEEP_CKPT_KEY: True}, inkling=True))
        self.assertNotIn("ckpt", StrataEngine.sampling_keys({}, inkling=True))
        self.assertNotIn("ckpt", StrataEngine.sampling_keys({"strata_checkpoint": False}, inkling=True))   # (a client's one-shot flag is not this)

    def test_two_chats_each_keep_their_prompt_with_two_slots(self):
        with tempfile.TemporaryDirectory() as td:
            with Harness(Path(td), parallel=2) as h:
                c = Client(f"http://127.0.0.1:{h.port}")
                a, b = conversation("A"), conversation("B")
                ra, _, _ = c.chat(a)                    # slot 0
                title = [{"role": "user", "content": "short helper request"}]
                c.chat(title)                           # slot 1
                a = next_turn(a, ra, "second")
                _, p, k = c.chat(a)
                self.assertReused((p, k), "chat A, turn 2")
                rb, _, _ = c.chat(b)                    # a new long chat: slot 0 (or whichever it wipes)
                b = next_turn(b, rb, "second")
                _, p, k = c.chat(b)
                self.assertReused((p, k), "chat B, turn 2")

    def test_a_follow_up_found_by_the_checkpoint_goes_to_the_slot_that_holds_it(self):
        # the follow-up does not extend the whole held stream (the reply was not sent back token for token), only the checkpoint:
        # the server must still recognise the slot.  A different assistant text than the one the mock produced:
        with tempfile.TemporaryDirectory() as td:
            with Harness(Path(td), parallel=2) as h:
                c = Client(f"http://127.0.0.1:{h.port}")
                a = conversation("A")
                c.chat(a)
                c.chat([{"role": "user", "content": "helper"}])      # a request in between
                a2 = next_turn(a, "A reply the model never wrote", "second")
                _, p, k = c.chat(a2)
                self.assertReused((p, k), "turn 2 with a reply that is not the generated one")


class KeepChats(unittest.TestCase):
    """#7226: Claude Code's auto mode alternates the main conversation with a large safety-check request (a different prompt of about the same size),
    and a client may also send short title requests.  With `--parallel 2` each long conversation keeps its slot (STRATA_INKLING_KEEP_CHATS, on unless 0),
    with the DSpark drafter on or off; with the switch at 0 the 0.2.22 routing is back: every large request wipes slot 0."""

    def alternate(self, keep, draft, rounds=3):
        import os
        from unittest import mock
        env = {"STRATA_INKLING_KEEP_CHATS": keep} if keep is not None else {}
        rows = []
        with mock.patch.dict(os.environ, env):
            if keep is None:
                os.environ.pop("STRATA_INKLING_KEEP_CHATS", None)
            with tempfile.TemporaryDirectory() as td:
                with Harness(Path(td), parallel=2, extra=("--draft", draft)) as h:
                    c = Client(f"http://127.0.0.1:{h.port}")
                    main, check = conversation("MAIN"), conversation("CHECK")
                    for i in range(rounds):
                        rm, p, k = c.chat(main)
                        rows.append(("main", i, p, k))
                        main = next_turn(main, rm, f"step {i}")
                        rc, p, k = c.chat(check)
                        rows.append(("check", i, p, k))
                        check = next_turn(check, rc, f"verify {i}")
        return rows

    def test_two_large_conversations_keep_their_slots_by_default(self):
        for draft in ("off", "dspark"):
            with self.subTest(draft=draft):
                rows = self.alternate(None, draft)
                for who, i, p, k in rows:
                    if i == 0:
                        self.assertEqual(k, 0, f"{who} {i}")
                    else:
                        self.assertGreaterEqual(k, p - 90, f"{who} {i}: {k} of {p} reused, drafter {draft}")

    def test_the_switch_at_zero_brings_back_the_old_routing(self):
        for draft in ("off", "dspark"):
            for off in ("0", "off"):
                with self.subTest(draft=draft, switch=off):
                    rows = self.alternate(off, draft)
                    self.assertTrue(all(k == 0 for _, _, _, k in rows), rows)

    def test_one_slot_is_unchanged_by_the_switch(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("STRATA_INKLING_KEEP_CHATS", None)
            with tempfile.TemporaryDirectory() as td:
                with Harness(Path(td), parallel=1) as h:
                    c = Client(f"http://127.0.0.1:{h.port}")
                    main, check = conversation("MAIN"), conversation("CHECK")
                    rm, _, _ = c.chat(main)
                    c.chat(check)
                    _, p, k = c.chat(next_turn(main, rm, "again"))
                    self.assertEqual(k, 0, "one slot: the other conversation took it")

    def run_title(self, draft, keep=None):
        import os
        from unittest import mock
        env = {"STRATA_INKLING_KEEP_CHATS": keep} if keep is not None else {}
        with mock.patch.dict(os.environ, env):
            if keep is None:
                os.environ.pop("STRATA_INKLING_KEEP_CHATS", None)
            return SlotRouting().run_session(2, between=[{"role": "user", "content": "Name this conversation in three words."}],
                                             extra=("--draft", draft))

    def test_a_title_request_does_not_wipe_the_chat_with_the_drafter_on(self):
        t1, t2, t3 = self.run_title("dspark")
        self.assertGreaterEqual(t2[1], t2[0] - 90, "turn 2 after a title request, drafter on")
        self.assertGreaterEqual(t3[1], t3[0] - 90, "turn 3")

    def test_the_switch_at_zero_keeps_the_old_rule_for_the_title_with_the_drafter_on(self):
        t1, t2, t3 = self.run_title("dspark", keep="0")
        self.assertEqual(t2[1], 0, "0.2.22: the drafter's slot-0 preference wipes the chat")

    def test_a_wrap_up_pass_keeps_the_checkpoint_with_the_drafter_on_and_off(self):
        for draft in ("off", "dspark"):
            with self.subTest(draft=draft):
                p1, p2, k2, lines = SlotRouting().wrapped_up_chat(2, extra=("--draft", draft))
                self.assertGreaterEqual(k2, p1 - 90, f"turn 2: {k2} of {p2} reused after a wrap-up pass, drafter {draft}")


if __name__ == "__main__":
    unittest.main()
