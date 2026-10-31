import re, json, pathlib
src = pathlib.Path(r"F:/Strata/third_party/llama.cpp/tests/test-json-schema-to-grammar.cpp").read_text(encoding="utf-8")
pat = re.compile(r'test\(\{\s*(SUCCESS|FAILURE),\s*"((?:[^"\\]|\\.)*)",\s*R"""\((.*?)\)""",\s*R"""\((.*?)\)"""\s*\}\);', re.S)
cases = [{"status": m[0], "name": m[1], "schema": m[2], "grammar": m[3]} for m in pat.findall(src)]
print(len(cases), "cases:", sum(c["status"] == "SUCCESS" for c in cases), "success")
with open(r"F:/Strata-dev/.zcode-obj/schema_cases.json", "w", encoding="utf-8") as f:
    json.dump(cases, f, ensure_ascii=False)
