"""#6984: `strata serve` decides "this is the Inkling-2-Mini engine" from the exe's file name only
(`inkling_kind.is_inkling_exe`: basename starts with "strata-inkling").  A wrapper script around the
engine (a .bat/.sh launcher, a renamed copy, a portable bundle) therefore comes up without Inkling's
end token in the server's stop set: every reply runs to max_tokens and reports finish_reason
"length", so agent clients (pi, DSH, ...) never see a completed turn and their tool calls fail.

The run config already says which model this is: the shape hint in serve/server.py sets
cfg["arch"] = "inkling2_loop" from the model file name (a .strata or .gguf called *inkling*), and
`inkling_kind.is_inkling(cfg)` reads it.  Detection must use the config as well as the exe name.

No files, no GPU, no engine process: the engine is built lazy, and its generate() is a script.

Red against main c23477403 (`v0.2.22-1-gc23477403`): test_a_wrapper_exe_does_not_hide_the_inkling_model,
test_the_flag_survives_a_restart, test_the_flag_survives_a_start_retry and
test_a_reply_ends_at_the_end_sampling_token fail; the other two pass before and after (they pin the config
opinion and that another model stays not-Inkling).

Run: python -m pytest serve/test_inkling_serve_detect_6984.py
"""
from __future__ import annotations

import sys
import threading
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve import inkling_kind, server                      # noqa: E402
from serve.frontend import ChatTemplate                     # noqa: E402
from serve.server_parts.engines import ByteTokenizer        # noqa: E402

TEMPLATE = ROOT / "tests" / "data" / "inkling" / "chat_template.jinja"
END_TEXT = "<|content_model_end_sampling|>"
MAX_NEW = 64
WRAPPER = "/opt/strata/launch-my-inkling.sh"                # a launcher's name says nothing about the engine inside
MODEL = "/models/Inkling-2-Mini-SQ-10.strata"


class InkTokenizer(ByteTokenizer):
    SPECIALS = ["<|endoftext|>", END_TEXT, "<|message_model|>", "<|message_user|>", "<|message_system|>",
                "<|message_tool|>", "<|content_thinking|>", "<|content_text|>", "<|content_xml|>",
                "<|content_invoke_tool_json|>", "<|end_message|>"]


def scripted_generate(self, ids, max_new, sampling, cancel, embeddings=None):
    """The engine's side of the write: one Inkling reply, then filler, and no stop of its own --
    the real engine without the server's stop ids also writes until the server's max_new."""
    opener = self.tok.encode("<|message_model|><|content_text|>", parse_special=True)
    body = self.tok.encode("The answer is 4.")
    end = self.tok.encode(END_TEXT, parse_special=True)
    filler = self.tok.encode(" and it kept writing after the end of the answer")
    out = opener + body + end + filler * 40
    row = {"prompt": len(ids), "sent": 0, "finish": "length"}
    self.last = row
    self.passes.append(row)
    i = 0
    while row["sent"] < max_new:
        t = out[i] if i < len(out) else out[-1]
        i += 1
        row["sent"] += 1
        yield t


class WrapperExe(unittest.TestCase):
    def setUp(self):
        self.tok = InkTokenizer()
        self.cfg = {"arch": "inkling2_loop", "exe": WRAPPER, "args": ["--native", MODEL],
                    "backend": "cuda", "cwd": "/opt/strata", "drafting": False}
        self.engine = server.start_engine_or_fallback(self.cfg, self.cfg["exe"], None, {}, True, None)
        # the process side is stubbed (the test is about detection and the stop set, not about a real engine)
        self.engine.tok, self.engine.passes = self.tok, []
        self.engine.alive = lambda: True
        self.engine.generate = types.MethodType(scripted_generate, self.engine)
        self.svc = server.Service(self.engine, self.tok, ChatTemplate(TEMPLATE))
        self.svc.model_arch = self.cfg["arch"]          # what main() sets for a.strata engine

    # ---- the gap between the two opinions -------------------------------------------------
    def test_the_config_says_inkling_and_the_wrapper_name_does_not(self):
        self.assertTrue(inkling_kind.is_inkling(self.cfg))               # the config knows the model
        self.assertFalse(inkling_kind.is_inkling_exe(self.cfg["exe"]))   # the exe name cannot say

    # ---- the engine object the server then uses --------------------------------------------
    def test_a_wrapper_exe_does_not_hide_the_inkling_model(self):
        self.assertTrue(self.engine.inkling_engine,
                        "the engine runs the Inkling model (cfg arch=inkling2_loop) behind a launcher; "
                        "serve sees inkling_engine=False and leaves Inkling's end token out of the stop set")
        self.assertIn(self.tok.encode(END_TEXT, parse_special=True)[0], self.svc.stop_ids)

    # ---- the flag survives a restart, and other models stay not-Inkling --------------------
    def test_the_flag_survives_a_restart(self):
        self.assertTrue(self.engine.inkling_engine)
        self.engine.close = lambda: None          # nothing to end
        exe, args, cwd, log, env, _lazy, cpus = self.engine.spawn
        self.engine.spawn = (exe, args, cwd, log, env, True, cpus)   # a lazy start: restart needs no process
        self.engine.restart(tries=1)
        self.assertTrue(self.engine.inkling_engine)

    def test_the_flag_survives_a_start_retry(self):
        self.assertTrue(self.engine.inkling_engine)
        exe, args, cwd, log, env, _lazy, cpus = self.engine.spawn
        # the start retries (server.py:948, 961, 974) re-enter __init__ with ten positional values, no hint
        server.StrataEngine.__init__(self.engine, exe, args, cwd, log, env, True, cpus)
        self.assertTrue(self.engine.inkling_engine)

    def test_another_model_stays_not_inkling(self):
        engine = server.StrataEngine(WRAPPER, ["--native", "/models/Qwen3.8-27B.strata"], lazy=True)
        self.assertFalse(engine.inkling_engine)

    # ---- what a client sees ----------------------------------------------------------------
    def test_a_reply_ends_at_the_end_sampling_token(self):
        ids = self.tok.encode("<|message_user|><|content_text|>What is 2+2?<|end_message|><|message_model|>",
                              parse_special=True)
        content, done = "", None
        for kind, payload in self.svc.run(ids, True, [], MAX_NEW, {}, threading.Event()):
            if kind == "event" and getattr(payload, "kind", None) == "content":
                content += payload.text or ""
            elif kind == "done":
                done = payload
        self.assertIsNotNone(done)
        self.assertEqual(done["finish"], "stop",
                         f"reply ran to max_tokens (finish={done['finish']!r}, {done['completion_tokens']} tokens)")
        self.assertLess(done["completion_tokens"], MAX_NEW)      # it stopped at the model's end token
        self.assertNotIn(END_TEXT, content)                     # and the end token is not in the answer


if __name__ == "__main__":
    unittest.main()
