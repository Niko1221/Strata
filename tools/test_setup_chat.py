"""tools/chat.py (make chat): python -m unittest tools.test_setup_chat"""
import contextlib
import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat as C  # noqa: E402

ENV = {"PROMPT": "Hi", "MAX_TOKENS": "800", "EFFORT": "high", "REASONING_BUDGET": "", "URL": "http://127.0.0.1:9"}


def reply(content, finish="stop"):
    return {"choices": [{"message": {"content": content, "reasoning_content": "thinking..."}, "finish_reason": finish}],
            "timings": {"predicted_n": 53, "predicted_per_second": 31.4}}


class Chat(unittest.TestCase):
    def run_main(self, env, response=None, error=None):
        sent = []

        def urlopen(req, timeout):
            sent.append(req)
            if error:
                raise error
            return contextlib.nullcontext(io.BytesIO(json.dumps(response).encode()))
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(C.urllib.request, "urlopen", urlopen), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = C.main({**ENV, **env})
        return code, out.getvalue(), err.getvalue(), sent

    def test_bad_numbers_send_nothing(self):
        for env in ({"MAX_TOKENS": "abc"}, {"MAX_TOKENS": ""}, {"MAX_TOKENS": "0"}, {"MAX_TOKENS": "-5"},
                    {"REASONING_BUDGET": "1.5"}, {"REASONING_BUDGET": "-1"}, {"REASONING_BUDGET": "+7"},
                    {"MAX_TOKENS": "400", "REASONING_BUDGET": "337"}, {"MAX_TOKENS": "400", "REASONING_BUDGET": "400"},
                    {"REASONING_BUDGET": "9" * 20}):
            with self.subTest(env=env):
                code, out, err, sent = self.run_main(env, reply("46"))
                self.assertEqual((code, out, sent), (1, "", []))
                self.assertTrue(err.startswith("chat: "), err)

    def test_the_budget_field(self):
        self.assertNotIn("reasoning_budget_tokens", C.request_body(ENV))                          # empty: server default
        self.assertEqual(C.request_body({**ENV, "REASONING_BUDGET": "0"})["reasoning_budget_tokens"], 0)   # no cap
        body = C.request_body({**ENV, "MAX_TOKENS": "400", "REASONING_BUDGET": "336"})                # 64 left: enough
        self.assertEqual((body["max_tokens"], body["reasoning_budget_tokens"], body["reasoning_effort"]), (400, 336, "high"))
        self.assertEqual(C.request_body({**ENV, "REASONING_BUDGET": "007"})["reasoning_budget_tokens"], 7)

    def test_an_answer_goes_to_stdout_and_the_count_to_stderr(self):
        code, out, err, sent = self.run_main({"REASONING_BUDGET": "32", "API_KEY": "k3y"}, reply("46"))
        self.assertEqual((code, out), (0, "46\n"))
        self.assertIn("53 tokens", err)
        self.assertEqual(sent[0].get_header("Authorization"), "Bearer k3y")
        self.assertEqual(json.loads(sent[0].data)["reasoning_budget_tokens"], 32)

    def test_no_answer_is_a_failure_not_the_thinking(self):
        code, out, err, _ = self.run_main({}, reply("", finish="length"))
        self.assertEqual((code, out), (1, ""))                     # the thinking is not printed as if it were the answer
        self.assertIn("REASONING_BUDGET", err)

    def test_server_errors_exit_1(self):
        refused = urllib.error.HTTPError("u", 400, "Bad Request", {}, io.BytesIO(b'{"error": "reasoning_budget_tokens"}'))
        for error, says in ((refused, "400"), (urllib.error.URLError("refused"), "could not reach")):
            with self.subTest(says=says):
                code, out, err, _ = self.run_main({}, error=error)
                self.assertEqual((code, out), (1, ""))
                self.assertIn(says, err)


if __name__ == "__main__":
    unittest.main()
