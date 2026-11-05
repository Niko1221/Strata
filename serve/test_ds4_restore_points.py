"""STRATA_DS4_RESTORE_POINTS: where the server asks the DeepSeek engine to keep restore points (its GEN line's marks=,
tail= and shared=), computed from the chat as the template renders it - no GPU, no engine (a fake one).  From Project
Maya's restore points (#639, mw00 / needmorevram), on Strata's shared-start pin.

    python -m unittest serve.test_ds4_restore_points -v
"""
from __future__ import annotations

import os
import sys
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve.server import StrataEngine, deepseek_restore_points, restore_point_settings  # noqa: E402
from serve import frontend  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import Service, default_model_name, serve  # noqa: E402
from serve.test_dsv41 import TEMPLATE, ApiBase, DsEngine  # noqa: E402

ON, TAIL = "STRATA_DS4_RESTORE_POINTS", "STRATA_DS4_RESTORE_TAIL"
# one id a byte (the stand-in tokenizer): a system prompt of ~2,700 tokens
SYSTEM = "You are a careful assistant; answer from the documents you are given. " * 40
DOC = "".join(f"Line {i}: the quarterly figures for region {i % 7} rose by {i % 13} percent.\n" for i in range(400))


def lcp(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class Pure(unittest.TestCase):
    """deepseek_restore_points() and the settings on ids built by hand."""

    USER, SYS = 900, 901

    def ids(self, *parts):
        out = []
        for p in parts:
            out += [p] if p in (self.USER, self.SYS) else [7] * p
        return out

    def test_last_user_turn_the_shared_start_and_the_tail(self):
        ids = self.ids(2000, self.USER, 1500, 5, self.USER, 3000)
        pts = deepseek_restore_points(ids, self.USER, self.SYS, 512)
        self.assertEqual(pts["shared"], 2000)
        self.assertEqual(pts["marks"], [2000 + 1 + 1500 + 5])
        self.assertEqual(pts["tail"], len(ids) - 512)

    def test_short_starts_are_left_out(self):
        ids = self.ids(300, self.USER, 100, self.USER, 100)
        self.assertEqual(deepseek_restore_points(ids, self.USER, self.SYS, 512), {})
        self.assertEqual(deepseek_restore_points(self.ids(5000), None), {})   # (no user token: no DeepSeek prompt)

    def test_a_system_message_after_the_first_user_turn_is_a_boundary_too(self):
        ids = self.ids(1500, self.USER, 1500, self.SYS, 40)
        pts = deepseek_restore_points(ids, self.USER, self.SYS, 0)
        self.assertEqual(pts, {"shared": 1500, "marks": [1500 + 1 + 1500]})
        self.assertEqual(ids[pts["marks"][0]], self.SYS)

    def test_tail_zero_and_a_tail_inside_the_last_turn_only(self):
        ids = self.ids(1200, self.USER, 4000)
        self.assertNotIn("tail", deepseek_restore_points(ids, self.USER, self.SYS, 0))
        self.assertEqual(deepseek_restore_points(ids, self.USER, self.SYS, 300)["tail"], len(ids) - 300)
        short = self.ids(1200, self.USER, 100)   # (the last turn shorter than the tail: the boundary alone)
        self.assertNotIn("tail", deepseek_restore_points(short, self.USER, self.SYS, 512))

    def test_settings(self):
        self.assertEqual(restore_point_settings({}), (True, 512))   # (on by default: 4)
        self.assertEqual(restore_point_settings({ON: "0"}), (False, 512))
        self.assertEqual(restore_point_settings({ON: "4"}), (True, 512))
        self.assertEqual(restore_point_settings({ON: "yes", TAIL: "x"}), (True, 512))   # (not a number: the default)
        self.assertEqual(restore_point_settings({ON: "8", TAIL: "0"}), (True, 0))
        self.assertEqual(restore_point_settings({ON: "8", TAIL: "2000"}), (True, 2000))

    def test_the_engine_line(self):
        s = {"_restore": {"marks": [3000, 4000], "tail": 5000, "shared": 2000}}
        self.assertEqual(StrataEngine.restore_keys(s), " marks=3000,4000 tail=5000 shared=2000")
        self.assertEqual(StrataEngine.restore_keys({}), "")
        self.assertEqual(StrataEngine.restore_keys({"_restore": {"marks": [0, -1, True], "shared": 0}}), "")
        self.assertIn("marks=3000,4000", StrataEngine.sampling_keys({**s, "temperature": 0.7}).split())
        self.assertEqual(StrataEngine.sampling_keys({"temperature": 0.7}),
                         StrataEngine.sampling_keys({"temperature": 0.7, "_restore": {}}))


class Api(ApiBase):
    """The same through the server, with the template's own tokens."""

    def start(self, script):
        engine = DsEngine(self.tok, script, max_context=1 << 20)   # (the documents here are tens of thousands of tokens)
        svc = Service(engine, self.tok, ChatTemplate(TEMPLATE), model_name=default_model_name({"arch": "deepseek41"}))
        svc.model_arch = "deepseek41"
        svc.set_family("deepseek41")
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.addCleanup(frontend.use_family, None)
        self.engine, self.svc = engine, svc
        self.base = f"http://127.0.0.1:{httpd.server_address[1]}"
        return engine

    def chat_body(self, last="And how did region 3 do in the last quarter?", system=SYSTEM):
        return {"messages": [{"role": "system", "content": system},
                             {"role": "user", "content": "Here are the figures.\n" + DOC},
                             {"role": "assistant", "content": "I have read them."},
                             {"role": "user", "content": last}], "max_tokens": 40}

    def run_request(self, body, env=None):
        env = env or {}
        with unittest.mock.patch.dict(os.environ, env):
            for k in (ON, TAIL):
                if k not in env:
                    os.environ.pop(k, None)
            self.assertEqual(self.post("/v1/chat/completions", body)[0], 200)
        return self.engine.last_prompt, self.engine.last_sampling

    def user_id(self):
        return self.svc.deepseek_turn_ids()[0]

    def test_marks_at_the_last_user_turn_and_the_pin_is_the_shared_start(self):
        self.start("ok")
        ids, s = self.run_request(self.chat_body())
        user = self.user_id()
        boundary = len(ids) - 1 - ids[::-1].index(user)
        self.assertEqual(s["_restore"]["marks"], [boundary])
        # the system prompt is already the request's shared start (pin=N): shared= is not asked twice
        self.assertNotIn("shared", s["_restore"])
        self.assertEqual(s["strata_prefix"]["tokens"], ids.index(user))
        keys = StrataEngine.sampling_keys(s).split()
        self.assertIn(f"marks={boundary}", keys)
        self.assertIn(f"pin={ids.index(user)}", keys)
        self.assertFalse([k for k in keys if k.startswith("shared=")])
        # the last message edited: the prompt agrees with this one up to the boundary, and leaves it before the end
        ids2, _ = self.run_request(self.chat_body("And region 5, over the whole year?"))
        self.assertGreaterEqual(lcp(ids, ids2), boundary)
        self.assertLess(lcp(ids, ids2), len(ids) - 1)

    def test_shared_is_asked_when_the_pin_is_switched_off(self):
        self.start("ok")
        with unittest.mock.patch.dict(os.environ, {"STRATA_DS4_AUTO_PIN": "0"}):
            ids, s = self.run_request(self.chat_body())
        self.assertNotIn("strata_prefix", s)
        self.assertEqual(s["_restore"]["shared"], ids.index(self.user_id()))
        self.assertIn(f"shared={ids.index(self.user_id())}", StrataEngine.sampling_keys(s).split())

    def test_tail_point_for_a_long_last_message(self):
        self.start("ok")
        ids, s = self.run_request({"messages": [{"role": "user", "content": DOC + "\nWhich region rose most?"}],
                                   "max_tokens": 40})
        self.assertEqual(s["_restore"], {"tail": len(ids) - 512})
        _, s = self.run_request({"messages": [{"role": "user", "content": DOC}], "max_tokens": 40}, {TAIL: "100"})
        self.assertEqual(s["_restore"], {"tail": len(self.engine.last_prompt) - 100})
        self.assertIn(f"tail={len(self.engine.last_prompt) - 100}", StrataEngine.sampling_keys(s).split())

    def test_off_with_0_and_never_the_clients_own(self):
        self.start("ok")
        _, on = self.run_request(self.chat_body())   # (on by default)
        self.assertIn("_restore", on)
        _, off = self.run_request(self.chat_body(), {ON: "0"})
        self.assertNotIn("_restore", off)
        for k in ("marks=", "shared=", "tail="):
            self.assertNotIn(k, StrataEngine.sampling_keys(off))
        # a client's own `_restore` never reaches the engine
        _, forged = self.run_request({**self.chat_body(), "_restore": {"marks": [5], "shared": 7}}, {ON: "0"})
        self.assertNotIn("_restore", forged)
        _, forged = self.run_request({**self.chat_body(), "_restore": {"marks": [5], "shared": 7}}, {ON: "4"})
        self.assertNotEqual(forged["_restore"].get("marks"), [5])

    def test_short_prompts_get_none(self):
        self.start("ok")
        _, s = self.run_request({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 40})
        self.assertNotIn("_restore", s)


if __name__ == "__main__":
    unittest.main()
