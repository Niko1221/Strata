import io
for p, old, new in [
    ("tools/test_grammar_builder.py",
     'self.assertIn("answer ::= ws answer-root", g)',
     'self.assertIn("answer ::= ws ( fenced | answer-root )", g)\n'
     '        self.assertIn("fenced ::= ', g)                              # the fenced-answer habit is legal, not banned'),
    ("serve/test_grammar.py",
     'self.assertIn("answer ::= ws answer-root", text)',
     'self.assertIn("answer ::= ws ( fenced | answer-root )", text)  # the fence habit is legal, canary strips it'),
]:
    s = io.open(p, encoding="utf-8").read()
    assert s.count(old) == 1, (p, s.count(old))
    s = s.replace(old, new, 1)
    io.open(p, "w", encoding="utf-8", newline="").write(s)
    print("patched", p)
