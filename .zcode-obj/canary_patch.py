import io
p = "serve/grammar.py"
s = io.open(p, encoding="utf-8").read()
old = ('                except StructuredOutputError as e:\n'
       '                    raise StructuredOutputError(\n'
       '                        f"grammar engine bug: a constrained stream failed its own schema: {e}") from e\n')
new = ('                except StructuredOutputError as e:\n'
       '                    # an unfinished stream is the budget, not the constraint: say which\n'
       '                    if finish != "stop":\n'
       '                        raise StructuredOutputError(\n'
       '                            f"the constrained answer did not fit the output budget: {e}") from e\n'
       '                    raise StructuredOutputError(\n'
       '                        f"grammar engine bug: a constrained stream failed its own schema: {e}") from e\n')
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("canary ok")
