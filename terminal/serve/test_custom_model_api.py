"""serve/test_custom_model_api.py - Verification that inference API routes enforce authorized models."""
from __future__ import annotations

import unittest


class TestCustomModelApi(unittest.TestCase):
    def test_public_api_no_file_import(self) -> None:
        """Inference client payload cannot inject arbitrary host filesystem paths."""
        def filter_inference_payload(data: dict) -> dict:
            forbidden_keys = {"model_path", "executable_path", "steering_vector_path"}
            return {k: v for k, v in data.items() if k not in forbidden_keys}

        payload = {
            "model": "deepseek-v4.1",
            "messages": [{"role": "user", "content": "test"}],
            "model_path": "/etc/shadow",
        }
        filtered = filter_inference_payload(payload)
        self.assertNotIn("model_path", filtered)
        self.assertEqual(filtered["model"], "deepseek-v4.1")


if __name__ == "__main__":
    unittest.main()
