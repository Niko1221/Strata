import json, os, sys
sys.path.insert(0, "tools")
import strata_tokenizer as T
import  sys
gguf = sys.argv[1]
tok = T.Tokenizer.from_gguf(gguf)
eos = [e[0] for e in (tok.encode(s, parse_special=True) for s in ("<|im_end|>", "<|endoftext|>")) if len(e) == 1]
cfg = {
    "exe": os.path.abspath("build/src/qwen36/strata-qwen36"),
    "args": ["--native", gguf, "--max-context", "32768", "--eos-ids", ",".join(map(str, eos))],
    "tokenizer": os.path.abspath("qwen36/tokenizer"),
    "model_name": "qwen3.6-35b-a3b",
    "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
}
json.dump(cfg, open("qwen36.json", "w"), indent=2)
print("scritto qwen36.json, eos =", eos)
