import json, re, subprocess
cases = json.load(open(r"F:/Strata-dev/.zcode-obj/schema_cases.json", encoding="utf-8"))
def norm(text):
    text = text.strip(" \n\r\t")
    return re.sub(r"(^|\n)[ \t]+", r"\1", text)
c = cases[0]
r = subprocess.run([r"F:/Strata-dev/.zcode-obj/grammar_schema.exe"], input=c["schema"].encode(),
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
a = r.stdout.decode("utf-8")
print("ACT:", repr(norm(a)[:80]))
print("EXP:", repr(norm(c["grammar"])[:80]))
print("EQ:", norm(a) == norm(c["grammar"]))
