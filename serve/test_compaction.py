"""The fold rules of web/compaction.js, run through node so `unittest discover` executes them with
the rest of the suite (skip when node is not installed - the browser path is unaffected)."""
import shutil
import sys
import subprocess
import unittest
from pathlib import Path

WEB = Path(__file__).parent / "web"


NODE = shutil.which("node")
if not NODE:
    # loud on purpose: a skip the reader can miss means the fold rules - which decide what
    # reaches the model - quietly stopped being tested on this machine
    print("SKIP serve/test_compaction.py: node is not installed - the fold rules are NOT being tested here",
          file=sys.stderr)


@unittest.skipUnless(NODE, "node is not installed")
class FoldRules(unittest.TestCase):
    def test_fold_rules(self):
        r = subprocess.run(["node", str(WEB / "test_compaction.mjs")], capture_output=True, text=True,
                           timeout=60, cwd=WEB)
        self.assertEqual(r.returncode, 0, f"fold rules failed:\n{r.stdout}\n{r.stderr}")
        self.assertIn("ok -", r.stdout)


if __name__ == "__main__":
    unittest.main()
