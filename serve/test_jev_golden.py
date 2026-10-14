"""serve/test_jev_golden.py - the golden battery against /v1/systemone (mock engine here; the
live-engine variant is the same battery over a --jev-brow engine, run on the rig).

    python3 -m unittest serve.test_jev_golden -v

Asserts only INVARIANTS (type correctness, probability sums, bounds, determinism on repeat) -
never model-specific answers; exact probabilities are pinned by the live golden run instead.
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, serve  # noqa: E402
from serve.jev_golden import check_invariants, load_battery  # noqa: E402
from serve.test_systemone import DecisionEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class GoldenBattery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        engine = DecisionEngine(tok, "A")
        cls.svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_battery_invariants(self):
        items = load_battery()
        self.assertEqual(len(items), 12)
        for i, item in enumerate(items):
            with self.subTest(item=i):
                st, body = post(self.base, "/v1/systemone", item)
                self.assertEqual(st, 200, f"item {i}: {body}")
                bad = check_invariants(item, body.get("answers", {}))
                self.assertEqual(bad, [], f"item {i} violations: {bad}")

    def test_deterministic_on_repeat(self):
        items = load_battery()
        for i, item in enumerate(items):
            with self.subTest(item=i):
                _, a = post(self.base, "/v1/systemone", item)
                _, b = post(self.base, "/v1/systemone", item)
                self.assertEqual(a.get("answers"), b.get("answers"),
                                 f"item {i}: repeat gave different answers")


if __name__ == "__main__":
    unittest.main()
