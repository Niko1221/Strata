import json

cases = json.load(open(r"F:/Strata-dev/.zcode-obj/schema_cases.json", encoding="utf-8"))
# single line, no wrapping: textwrap can split inside a JSON string value and inject a raw newline
blob = json.dumps(cases, ensure_ascii=True, indent=None)
wrapped = blob

body = '''#!/usr/bin/env python3
"""Golden tests for tools/grammar_schema (the vendored llama.cpp JSON-Schema->GBNF converter).

The cases are the goldens from llama.cpp 3cf03257 tests/test-json-schema-to-grammar.cpp (77 cases: 75
conversions, 2 expected failures), embedded below.  The comparison mirrors that harness: trim the whole
text, strip leading whitespace of every line, compare exactly.  Run as a ctest (grammar_schema_test)
or directly: python tools/test_grammar_schema.py [path-to-grammar_schema.exe].
"""
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

EXE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "build" / "grammar_schema.exe"
sys.argv = sys.argv[:1]  # unittest.main() must not see the exe path

CASES = json.loads(r"""
%s
""")


def norm(text):
    text = text.replace("\\r\\n", "\\n")  # the CLI writes text-mode on Windows: CRLF
    text = text.strip(" \\n\\r\\t")
    return re.sub(r"(^|\\n)[ \\t]+", r"\\1", text)


def run(exe, stdin_text):
    return subprocess.run([str(exe)], input=stdin_text.encode("utf-8"),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)


class GrammarSchemaGoldens(unittest.TestCase):
    def test_cases(self):
        if not EXE.exists():
            self.skipTest("grammar_schema.exe not built: " + str(EXE))
        for c in CASES:
            with self.subTest(c["name"]):
                r = run(EXE, c["schema"])
                if c["status"] == "SUCCESS":
                    self.assertEqual(r.returncode, 0, c["name"] + ": " + r.stderr.decode("utf-8", "replace"))
                    self.assertEqual(norm(r.stdout.decode("utf-8")), norm(c["grammar"]), c["name"])
                else:
                    self.assertNotEqual(r.returncode, 0, c["name"] + " should fail")
                    self.assertTrue(r.stderr.decode("utf-8", "replace").strip(), c["name"] + " needs a stderr line")


if __name__ == "__main__":
    unittest.main(verbosity=2)
''' % wrapped

open(r"F:/Strata-dev/tools/test_grammar_schema.py", "w", encoding="utf-8").write(body)
print("wrote", len(body), "bytes,", len(cases), "cases")
