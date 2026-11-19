"""Projection request records stay bound to the actual child; no native process."""
import json
from pathlib import Path
import unittest
import urllib.request

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve


class RefusalProjectionObservation(unittest.TestCase):
    def test_loaded_off_and_child_replacement(self) -> None:
        tok = ByteTokenizer()
        engine = MockEngine(tok, "Hello.", max_context=4096)
        engine.info = {"cvec": "project:4-44"}
        engine._child_generation = "child-one"
        service = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(service, port=0)
        try:
            for value, summary, generation, expected in (
                (True, "project:4-44", "child-one", True),
                (False, "project:4-44", "child-one", False),
                (True, 0, "child-two", None),
                (True, "project:4-44", "child-three", True),
            ):
                with self.subTest(value=value, summary=summary, generation=generation):
                    engine.info = {"cvec": summary}
                    engine._child_generation = generation
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                        headers={"Content-Type": "application/json"},
                        data=json.dumps({"messages": [{"role": "user", "content": "Hello"}],
                                         "experimental_speed_projection": value,
                                         "max_tokens": 64}).encode(),
                    )
                    with urllib.request.urlopen(request, timeout=10) as response:
                        self.assertEqual(response.status, 200)
                        response.read()
                    last = service.metrics()["requests"][0]
                    self.assertIs(last["projection"], expected)
                    self.assertEqual(last["projection_child_generation"], generation if summary else None)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_existing_engine_request_switch_contract(self) -> None:
        for enabled in (True, False):
            self.assertEqual(StrataEngine.projection_key({"experimental_speed_projection": enabled}),
                             f" cvec={int(enabled)}")
        self.assertEqual(StrataEngine.projection_key({}), "")


if __name__ == "__main__":
    unittest.main()
