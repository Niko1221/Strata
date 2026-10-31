import io
B = chr(92)  # one literal backslash as it appears in the file source
def src(s):  # '"\\n..."' file text uses doubled backslashes
    return s.replace("@", B + B + "n")

def patch(path, pairs):
    s = io.open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old))
        s = s.replace(old, new, 1)
    io.open(path, "w", encoding="utf-8", newline="").write(s)
    print("patched", path)

# tools/test_grammar_builder.py:119 — Shape thinking: closer literal
patch("tools/test_grammar_builder.py", [
    ('self.assertIn(\'"' + src("@") + '</think>' + src("@@") + '"\', g)',
     'self.assertIn(\'root ::= reason "</think>' + src("@") + '" ws answer\', g)'),
    # :146 ThinkPrologue root line + the ws/answer pair
    ('self.assertIn(\'root ::= reason "' + src("@") + '</think>' + src("@@") + '" answer-root\', g)',
     'self.assertIn(\'root ::= reason "</think>' + src("@") + '" answer\', g)\n'
     '        self.assertIn("answer ::= ws answer-root", g)\n'
     '        self.assertIn(\'ws ::= " " | "\\\\t" | "' + src("@") + '", g)'),
])

# serve/test_grammar.py:89 response_format thinking root; :258 anthropic tool thinking root
patch("serve/test_grammar.py", [
    ('self.assertIn(\'root ::= reason "' + src("@") + '</think>' + src("@@") + '" answer-root\', text)   # reasoning, the closer, the answer',
     'self.assertIn(\'root ::= reason "</think>' + src("@") + '" answer\', text)   # reasoning, the closer, the answer\n'
     '        self.assertIn("answer ::= ws answer-root", text)'),
    ('self.assertIn(\'root ::= reason "' + src("@") + '</think>' + src("@@") + '" answer\', engine.grammars[-1])',
     'self.assertIn(\'root ::= reason "</think>' + src("@") + '" ws answer\', engine.grammars[-1])'),
])
