import io
p = "serve/grammar_tools.py"
s = io.open(p, encoding="utf-8").read()
anchor = 'THINK_CLOSE = "</think>' + chr(92) + 'n"                    # the closer as a model writes it, + one newline (see with_think_prologue)'
assert s.count(anchor) == 1, s.count(anchor)
new = anchor + ("\nTHINK_BANNED = [\"</think>\", \"```json\"]       # the reason region refuses the closer (it opens the answer)"
                " and the model's fenced-JSON answer habit (free-text reasoning that never ends)")
s = s.replace(anchor, new, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
