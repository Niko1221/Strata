#!/usr/bin/env python3
"""dsv4_tokenize.py - text in, comma-separated token ids out, for dsv4_run --prompt-ids.

Uses Strata's own BPE tokenizer (tools/strata_tokenizer.py), which reads the vocabulary out of the
GGUF metadata: no weights, no HuggingFace download. Add --chat to apply the model's chat template
first (the same Jinja template serve/server.py uses), which is what a real request needs.

    python3 tools/dsv4_tokenize.py --model DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf "ciao mondo"
    python3 tools/dsv4_tokenize.py --model M.gguf --chat --messages '[{"role":"user","content":"ciao"}]'

Pipe it:

    IDS=$(python3 tools/dsv4_tokenize.py --model M.gguf --chat --messages "$MSG")
    ./build-out/dsv4_run --model M.gguf --prompt-ids "$IDS" --n-predict 32

WHY THIS FILE IS NOT CALLED tokenize.py.  `python3 tools/serve_dsv4.py` puts this directory at the
front of sys.path, so a module named tokenize.py here shadows the standard library's `tokenize`.
The first time the server formats a traceback, linecache calls tokenize.open() and dies with
"module 'tokenize' has no attribute 'open'" - a crash in the error handler, which looks like a
tokenizer bug and is nothing of the sort.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]        # the Strata checkout that holds dsv4_ext
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("text", nargs="?", default=None, help="the text to tokenize (or use --messages)")
    ap.add_argument("--model", required=True, help="the first GGUF shard (its metadata carries the tokenizer)")
    ap.add_argument("--chat", action="store_true", help="render the model's chat template first")
    ap.add_argument("--messages", help="JSON list of {role, content} messages (needs --chat)")
    ap.add_argument("--tools", help="JSON list of tool definitions for the template")
    ap.add_argument("--no-generation-prompt", action="store_true",
                    help="template without the trailing assistant turn")
    ap.add_argument("--ids", action="store_true", help="print the ids only (default: ids plus a stderr summary)")
    a = ap.parse_args()

    import strata_tokenizer as ST
    tk = ST.Tokenizer.from_gguf(a.model)

    text = a.text
    if a.chat:
        if not a.messages:
            raise SystemExit("--chat needs --messages (a JSON list of {role, content})")
        from serve.frontend import ChatTemplate
        tpl_path = Path(a.model).with_suffix("")  # the template usually lives next to the tokenizer dir
        from gguf_reader import GGUFFile
        src = GGUFFile(Path(a.model)).metadata.get("tokenizer.chat_template")
        if not src:
            src = (ROOT / "serve" / "chat_template.jinja").read_text(encoding="utf-8")
            print("the GGUF carries no tokenizer.chat_template: using Strata's serve/chat_template.jinja",
                  file=sys.stderr)
        tmp = Path("/tmp/dsv4_chat_template.jinja")
        tmp.write_text(src, encoding="utf-8", newline="\n")
        rendered = ChatTemplate(tmp).render(json.loads(a.messages), tools=json.loads(a.tools) if a.tools else None,
                                            add_generation_prompt=not a.no_generation_prompt)
        text = rendered
    if text is None:
        raise SystemExit("give the text, or --chat --messages …")

    ids = tk.encode(text, parse_special=True)
    sys.stdout.write(",".join(str(i) for i in ids))
    if not a.ids:
        print(f"\n{len(ids)} tokens, vocab {len(tk.tokens)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
