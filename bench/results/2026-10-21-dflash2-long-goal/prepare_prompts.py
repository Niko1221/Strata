"""Freeze additional long prompts from baseline source, without model downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import strata_tokenizer as ST
from serve.frontend import ChatTemplate

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--tokenizer", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
o = p.parse_args()
vocab = json.loads((o.tokenizer / "vocab.json").read_text())
tokens = [None] * len(vocab)
for token, index in vocab.items():
    tokens[index] = token
tok = ST.Tokenizer(tokens, (o.tokenizer / "merges.txt").read_text().split("\n"),
                   json.loads((o.tokenizer / "token_type.json").read_text()))
template = ChatTemplate(o.tokenizer / "chat_template.jinja")
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
cases = [
    ("policy", ["include/strata/spec/dflash_policy.hpp", "src/spec/dflash_policy_test.cpp"], 3000,
     "Explain how this policy chooses a draft width, and propose a focused boundary-case regression test."),
    ("vram", ["strata_control/vram_fit.py"], 4800,
     "Explain this VRAM-fit code and describe a test for a model that cannot fit within its memory budget."),
    ("launch", ["docs/TERMINAL_LAUNCH.md"], 5200,
     "Explain how a user should preview a launch, reuse local model files, and keep the server local."),
]
prompts, provenance = {}, {"revision": revision, "cases": {}}
for name, paths, budget, question in cases:
    sections, hashes = [], {}
    for path in paths:
        data = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)
        hashes[path] = hashlib.sha256(data).hexdigest()
        sections.append(f"FILE: {path}\n{data.decode()}\n")
    body = tok.decode(tok.encode("\n".join(sections), parse_special=False)[:budget])
    text = f"Use the following source excerpts.\n\n{body}\n\nEND OF EXCERPTS.\n{question}"
    rendered = template.render([{"role": "user", "content": text}])
    prompts[name] = tok.encode(rendered, parse_special=True)
    provenance["cases"][name] = {"source_sha256": hashes, "input_tokens": len(prompts[name]),
                                "question": question, "body_budget": budget}
o.output.parent.mkdir(parents=True, exist_ok=True)
if o.output.exists() or o.output.with_suffix(".provenance.json").exists():
    p.error("output already exists; choose a new file")
o.output.write_text(json.dumps(prompts, indent=2))
o.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2))
print(json.dumps({name: len(ids) for name, ids in prompts.items()}))
