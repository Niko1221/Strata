import io
p = "docs/DETAILS.md"
s = io.open(p, encoding="utf-8").read()
N = chr(92) + "n"  # backslash + n, the GBNF escape for a newline
broken = ("  only calls; \"auto\" adds a prose branch whose grammar refuses the literal `\n"
          + "<tool_call>` - prose may start\n"
          + "  a call only to finish it; parallel calls and the `\n"
          + "</think>` prologue are carried. The forced-prefix")
fixed = ("  only calls; \"auto\" adds a prose branch whose grammar refuses the literal `" + N + "<tool_call>`"
         " - prose may start a call only to finish it; parallel calls and the `" + N + "</think>` prologue are"
         " carried. The forced-prefix")
assert s.count(broken) == 1, s.count(broken)
s = s.replace(broken, fixed, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("fixed")
