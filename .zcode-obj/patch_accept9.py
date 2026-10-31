P = "F:/Strata-dev/tools/grammar_acceptance.py"
src = open(P, encoding="utf-8").read()
old = "    while time.time() - t0 < 180:"
new = "    while time.time() - t0 < 420:   # the model load takes 1-3 minutes (page-locked cache build)"
assert old in src
src = src.replace(old, new, 1)
open(P, "w", encoding="utf-8").write(src)
print("patched9")
