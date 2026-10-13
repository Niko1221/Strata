"""The web app's sampling defaults match the model card's thinking-mode settings (#2306)."""
from pathlib import Path
import re
import unittest

APP_JS = Path(__file__).with_name("web") / "app.js"


class WebSamplingDefaults(unittest.TestCase):
    def test_defaults_match_the_model_card(self):
        line = re.search(r"^const DEFAULTS = \{(.*)\};$", APP_JS.read_text(encoding="utf-8"), re.M)
        self.assertIsNotNone(line)
        fields = dict(re.findall(r"(\w+): ([\w.\"]+)", line.group(1)))
        # Qwen3.8-Flash-Next model card, thinking mode, and its generation_config.json: 1.0 / 0.95 / 20
        self.assertEqual(float(fields["temperature"]), 1.0)
        self.assertEqual(float(fields["top_p"]), 0.95)
        self.assertEqual(int(fields["top_k"]), 20)


if __name__ == "__main__":
    unittest.main()
