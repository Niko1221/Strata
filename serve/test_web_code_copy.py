"""#1257: code blocks have Copy actions before and after the code, including unfinished fences.

    python -m unittest serve.test_web_code_copy -v

Runs the web app's actual Markdown renderer in Node; skipped when Node is absent.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from html.parser import HTMLParser
from pathlib import Path

APP = Path(__file__).parent / "web" / "app.js"
NODE = shutil.which("node")
RUN = r"""
const icon = (name) => `[${name}]`;
const texts = JSON.parse(require("fs").readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(texts.map(markdown)));
"""


def render(text: str) -> str:
    js = APP.read_text(encoding="utf-8")
    esc = next(line for line in js.splitlines() if line.startswith("const esc = "))
    part = js[js.index("function inline(s)"):js.index("// ------------------------------------------------------------------ Chat")]
    result = subprocess.run([NODE, "-e", esc + "\n" + part + RUN], input=json.dumps([text]),
                            capture_output=True, text=True, encoding="utf-8", timeout=60)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)[0]


class CodeBlocks(HTMLParser):
    def __init__(self, html: str):
        super().__init__()
        self.order, self.names, self.code, self.tags = [], [], [], []
        self.in_code = False
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        attrs = dict(attrs)
        if tag == "button" and "data-code-copy" in attrs:
            self.order.append("copy")
            self.names.append(attrs.get("aria-label"))
        if tag == "pre":
            self.order.append("code")
            self.code.append("")
            self.in_code = True

    def handle_endtag(self, tag):
        if tag == "pre":
            self.in_code = False

    def handle_data(self, data):
        if self.in_code:
            self.code[-1] += data


@unittest.skipUnless(NODE, "Node is not installed")
class CopyControls(unittest.TestCase):
    def test_each_block_has_actions_before_and_after_its_code(self):
        blocks = CodeBlocks(render("```python\nprint(1)\n```\n\n```bash\necho second\n```"))
        self.assertEqual(blocks.order, ["copy", "code", "copy", "copy", "code", "copy"])
        self.assertEqual(blocks.names, ["Copy code"] * 4)
        self.assertEqual(blocks.code, ["print(1)", "echo second"])

    def test_unfinished_and_empty_fences_have_both_actions(self):
        for text, code in (("```python\nprint(1)", "print(1)"), ("```\n", "")):
            with self.subTest(text=text):
                block = CodeBlocks(render(text))
                self.assertEqual(block.order, ["copy", "code", "copy"])
                self.assertEqual(block.names, ["Copy code", "Copy code"])
                self.assertEqual(block.code, [code])

    def test_code_and_language_stay_text_instead_of_markup(self):
        code = '<button data-code-copy>literal</button>\n<script>"<&"</script>\n  café `x`'
        block = CodeBlocks(render(f"```<img>\n{code}\n```"))
        self.assertEqual(block.code, [code])
        self.assertEqual(block.order, ["copy", "code", "copy"])
        self.assertNotIn("script", block.tags)
        self.assertNotIn("img", block.tags)


if __name__ == "__main__":
    unittest.main()
