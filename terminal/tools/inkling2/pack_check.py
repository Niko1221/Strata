"""pack_check.py REPO PACK MODEL: load the extracted pack exactly as server.py does, compare to HF on a few files."""
import json, pathlib, sys
repo, pack, model = sys.argv[1:4]
sys.path.insert(0, f"{repo}/terminal/tools")
import strata_tokenizer as ST
from transformers import AutoTokenizer
tpath = pathlib.Path(pack, "tokenizer")
vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
tokens = [None] * len(vocab)
for t, i in vocab.items():
    tokens[i] = t
merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
types = json.loads((tpath / "token_type.json").read_text())
tcfg = json.loads((tpath / "tokenizer.json").read_text(encoding="utf-8"))
tok = ST.Tokenizer(tokens, merges, types, tcfg.get("pre") or "qwen35", tcfg.get("special_ids") or None,
                   bool(tcfg.get("ignore_merges")))
hf = AutoTokenizer.from_pretrained(model)
bad = 0
for f in ["/var/fastasfuck/inkling2/kl/long.txt", "/var/fastasfuck/bench/q1.md", "/var/fastasfuck/bench/q2.md",
          f"{repo}/terminal/serve/frontend.py"]:
    s = open(f).read()
    a, b = tok.encode(s), hf(s, add_special_tokens=False, split_special_tokens=True).input_ids
    bad += a != b
    print(f, len(a), "OK" if a == b else "MISMATCH")
print("PACK-CHECK", "OK" if bad == 0 else "FAIL")
