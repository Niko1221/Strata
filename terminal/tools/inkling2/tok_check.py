"""tok_check.py: Strata's tokenizer (tools/strata_tokenizer.py, pre "gpt-4o") vs the HF tokenizer of Inkling-2-Mini.

1. GPT4O_PATTERN must equal the regex in the model's tokenizer.json (copied, not re-typed);
2. ids must match HF on a corpus: per paragraph and per whole file (EN/FR prose, the bench prompts, Strata docs, C++ and
   Python sources, unicode edge cases), and on rendered chat prompts with parse_special;
3. decode(encode(s)) == s.
usage: tok_check.py MODEL_DIR STRATA_REPO [TEXT...] [--extract OUT_DIR]  (TEXT: extra corpus files)
"""
import glob, json, pathlib, sys, time

model, repo = sys.argv[1], sys.argv[2]
sys.path.insert(0, f"{repo}/terminal/tools")
import strata_tokenizer as st
from transformers import AutoTokenizer

tj = json.loads(pathlib.Path(model, "tokenizer.json").read_text())
hf_pat = tj["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
assert hf_pat == st.GPT4O_PATTERN, "GPT4O_PATTERN differs from tokenizer.json"
print("pattern: identical to tokenizer.json")

hf = AutoTokenizer.from_pretrained(model)
t0 = time.time()
tk = st.Tokenizer.from_hf(pathlib.Path(model, "tokenizer.json"), "gpt-4o", {"eos": 200006})
print(f"strata tokenizer: {len(tk.tokens)} tokens, {len(tk.ranks)} merges, {len(tk.special_tokens)} special, {time.time() - t0:.1f}s")

files = sys.argv[3:]   # extra corpus files, minus "--extract OUT_DIR"
if "--extract" in files:
    i = files.index("--extract")
    files = files[:i] + files[i + 2:]
files += sorted(glob.glob(f"{repo}/docs/*.md"))[:12]
files += [f"{repo}/engine/src/inkling2/ink2_engine.cu", f"{repo}/terminal/serve/frontend.py"]
edge = ["Hello, world! It's 2026; they'll've done it.", "L'été où j'ai vu l'œuf — «déjà» ?", "ÉCOLE École école ÉCOLE's",
        "naïve café́ é́ कि", "😀👍🏽 🇫🇷 a‍b", "你好，世界 こんにちは 안녕하세요", "مرحبا بالعالم",
        "   leading\t\ttabs\n\n\n  trailing   ", "x = 1234567 + 89.01e-5; // 0x1F", "path/to/file.py:42 http://a.b/c?d=e&f",
        "\x00\x01\x1f\x7f ctrl", "CamelCaseIdentifierHTTPServer getHTTPResponseCode", "a\r\nb\rc\n\r\nd", "I'M SHOUTING'S"]
docs = [(f, open(f, errors="replace").read()) for f in files if pathlib.Path(f).is_file()]
pieces = [("edge", s) for s in edge]
for f, txt in docs:
    pieces.append((f, txt))
    pieces += [(f + "#p", p) for p in txt.split("\n\n") if p.strip()]
chats = [[{"role": "user", "content": "Explique en une phrase pourquoi le ciel est bleu."}],
         [{"role": "system", "content": "Tu es concis."}, {"role": "user", "content": open(f"{repo}/README.md").read()}],
         [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4", "reasoning_content": "easy"},
          {"role": "user", "content": "et 3+3 ?"}]]
bad, n, nchar = 0, 0, 0
t0 = time.time()
for name, s in pieces:
    a = tk.encode(s)  # raw text: markers are text (Strata's parse_special=False); HF matches them unless told not to
    b = hf(s, add_special_tokens=False, split_special_tokens=True).input_ids
    n += 1
    nchar += len(s)
    if a != b:
        bad += 1
        if bad <= 5:
            k = next(i for i in range(min(len(a), len(b)) + 1) if i >= min(len(a), len(b)) or a[i] != b[i])
            print(f"MISMATCH {name}: at token {k}: strata {a[k:k + 6]} hf {b[k:k + 6]} | "
                  f"{tk.decode(a[max(0, k - 3):k + 4])!r} vs {hf.decode(b[max(0, k - 3):k + 4])!r}")
    if tk.decode(a) != s:
        print(f"ROUNDTRIP {name}: decode(encode(s)) != s")
        bad += 1
for c in chats:
    s = hf.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
    a = tk.encode(s, parse_special=True)
    b = hf(s, add_special_tokens=False).input_ids
    n += 1
    if a != b:
        bad += 1
        print(f"MISMATCH chat ({len(a)} vs {len(b)} ids)")
print(f"{n} strings, {nchar} chars, {bad} mismatches, encode {time.time() - t0:.1f}s")
if "--extract" in sys.argv and bad == 0:
    out = sys.argv[sys.argv.index("--extract") + 1]
    print("pack:", st.extract_hf(model, out, "gpt-4o", {"eos": 200006}))
print("TOK-CHECK-DONE", "OK" if bad == 0 else "FAIL")
