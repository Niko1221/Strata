P = "F:/Strata-dev/tools/grammar_acceptance.py"
src = open(P, encoding="utf-8").read()

old = '''def json_suite(base, n, out_dir=None):'''
new = '''def parse_answer(text):
    """One JSON value out of the answer: a code fence around it is a legal grammar path (the think
    prologue's fenced alternative), so strip it the way the serve canary does before parsing."""
    s = (text or "").strip()
    if s.startswith("```"):
        first = s.find("\\n")
        if first != -1:
            end = s.find("```", first + 1)
            if end != -1:
                s = s[first + 1:end].strip()
    return json.loads(s)


def json_suite(base, n, out_dir=None):'''
assert old in src
src = src.replace(old, new, 1)

old2 = '''                try:
                    value = json.loads(msg.get("content") or "")
                except Exception as e:'''
new2 = '''                try:
                    value = parse_answer(msg.get("content"))
                except Exception as e:'''
assert old2 in src
src = src.replace(old2, new2, 1)

open(P, "w", encoding="utf-8").write(src)
print("patched8")
