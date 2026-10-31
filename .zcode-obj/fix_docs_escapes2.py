import io
p = "docs/DETAILS.md"
s = io.open(p, encoding="utf-8").read()
N = chr(92) + "n"
old = ("  only calls; \"auto\" adds a prose branch whose grammar refuses the literal `" + N + "<tool_call>`"
       " - prose may start a call only to finish it; parallel calls and the `" + N + "</think>` prologue are"
       " carried. The forced-prefix")
new = ("  only calls; \"auto\" adds a prose branch whose grammar refuses the literal `" + N + "<tool_call>` -\n"
       "  prose may start a call only to finish it; parallel calls and the `" + N + "</think>` prologue (the\n"
       "  opening `\\n` is the template's) are carried. The forced-prefix")
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("fixed2")
