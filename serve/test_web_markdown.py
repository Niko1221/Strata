"""serve/test_web_markdown.py - the web app's Markdown (serve/web/app.js: inline, blocks, markdown), run in Node.

    python -m unittest serve.test_web_markdown -v

No model, no GPU, no browser: the Markdown part of app.js is cut out and run by `node`.  Skipped when Node is not
installed (the server itself never needs it).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

APP = Path(__file__).parent / "web" / "app.js"
NODE = shutil.which("node")

RUN = r"""
const icon = (name) => `[${name}]`;
const texts = JSON.parse(require("fs").readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(texts.map(markdown)));
"""


def render(texts: list[str], *, timeout: float = 60) -> list[str]:
    js = APP.read_text(encoding="utf-8")
    esc = next(line for line in js.splitlines() if line.startswith("const esc = "))
    part = js[js.index("function inline(s)"):js.index("// ------------------------------------------------------------------ Chat")]
    r = subprocess.run([NODE, "-e", esc + "\n" + part + RUN], input=json.dumps(texts), capture_output=True,
                       text=True, encoding="utf-8", timeout=timeout)
    if r.returncode != 0:
        raise AssertionError(r.stderr)
    return json.loads(r.stdout)


@unittest.skipUnless(NODE, "Node is not installed")
class Lists(unittest.TestCase):
    def html(self, text: str) -> str:
        return render([text])[0]

    def test_items_separated_by_blank_lines_are_one_list(self):
        """#671: a loose list is one <ol>, so the browser numbers it 1, 2, 3 (it was three lists, each "1.")."""
        loose = ("1. **Keep a regular schedule**\n   Go to bed at the same time.\n\n"
                 "2. **Optimise the bedroom**\n   Quiet and dark.\n\n\n"
                 "3. **Cut screen time**\n   Blue light.\n")
        self.assertEqual(self.html(loose),
                         "<ol><li><strong>Keep a regular schedule</strong> Go to bed at the same time.</li>"
                         "<li><strong>Optimise the bedroom</strong> Quiet and dark.</li>"
                         "<li><strong>Cut screen time</strong> Blue light.</li></ol>")
        self.assertEqual(self.html("- a\n\n- b\n\n* c"), "<ul><li>a</li><li>b</li><li>c</li></ul>")
        self.assertEqual(self.html("1. a\n2. b"), "<ol><li>a</li><li>b</li></ol>")      # a tight list: as before

    def test_a_long_run_of_blank_lines_does_not_stall_rendering(self):
        """Skipping blank lines must not scan the same run again for each line."""
        text = "1. a\n" + "\n" * 100_000 + "2. b"
        try:
            html = render([text], timeout=5)[0]
        except subprocess.TimeoutExpired:
            self.fail("100,000 blank lines between items took over five seconds to render")
        self.assertEqual(html, "<ol><li>a</li><li>b</li></ol>")

    def test_a_blank_line_still_ends_a_list_before_anything_else(self):
        cases = {"1. a\n\ntext": "<ol><li>a</li></ol><p>text</p>",
                 "1. a\n\n## H": "<ol><li>a</li></ol><h3>H</h3>",
                 "1. a\n\n> q": "<ol><li>a</li></ol><blockquote>q</blockquote>",
                 "1. a\n\n---": "<ol><li>a</li></ol><hr>",
                 "- a\n\n- - -": "<ul><li>a</li></ul><hr>",
                 "1. a\n\n| x |\n|---|\n| y |": "<ol><li>a</li></ol><table><thead><tr><th>x</th></tr></thead>"
                                                "<tbody><tr><td>y</td></tr></tbody></table>",
                 "1. a\n\n- b": "<ol><li>a</li></ol><ul><li>b</li></ul>",          # another kind of list: a new one
                 "- a\n\n1. b": "<ul><li>a</li></ul><ol><li>b</li></ol>",
                 "1. a\n\n": "<ol><li>a</li></ol>",                                # the end of the message
                 "1. a\n\n**b** c": "<ol><li>a</li></ol><p><strong>b</strong> c</p>"}
        for text, want in cases.items():
            self.assertEqual(self.html(text), want, text)

    def test_a_list_continued_later_keeps_its_numbers(self):
        """Steps with a code block or a paragraph between them go on counting: the list starts at its own number."""
        steps = "1. Install it:\n```bash\npip install x\n```\n2. Run it:\n```bash\nx\n```\n3. Done."
        html = self.html(steps)
        self.assertEqual([html.count(t) for t in ("<ol>", '<ol start="2">', '<ol start="3">')], [1, 1, 1])
        self.assertEqual(self.html("1. a\n\n   more about a\n\n2. b\n3. c"),
                         '<ol><li>a</li></ol><p>   more about a</p><ol start="2"><li>b</li><li>c</li></ol>')
        self.assertEqual(self.html("0. a\n1. b"), '<ol start="0"><li>a</li><li>b</li></ol>')
        self.assertEqual(self.html("7) a"), '<ol start="7"><li>a</li></ol>')
        self.assertEqual(self.html("- a"), "<ul><li>a</li></ul>")                  # a bullet list has no number


if __name__ == "__main__":
    unittest.main()
