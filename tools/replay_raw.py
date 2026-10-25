"""Replay the captured raw model outputs (~/.cache/pai/strata-raw/raw-*.jsonl, field raw_text) through the CURRENT
OutputParser, streamed (token-sized splits) and whole, with Claude Code's tool schemas; flag every parsed call whose
value holds call markup or that lacks a required argument.

    python tools/replay_raw.py [-v] [DIR_OR_FILE ...]
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from serve.frontend import OutputParser  # noqa: E402

S = {"type": "string"}
B = {"type": "boolean"}
N = {"type": "number"}
TOOLS = {   # name -> (required, properties)
    "Bash": (["command"], {"command": S, "timeout": N, "description": S, "run_in_background": B,
                           "dangerouslyDisableSandbox": B}),
    "Read": (["file_path"], {"file_path": S, "offset": N, "limit": N, "pages": S}),
    "Write": (["file_path", "content"], {"file_path": S, "content": S}),
    "Edit": (["file_path", "old_string", "new_string"], {"file_path": S, "old_string": S, "new_string": S,
                                                          "replace_all": B}),
    "Grep": (["pattern"], {"pattern": S, "path": S, "glob": S, "output_mode": S, "-B": N, "-A": N, "-C": N,
                           "context": N, "-n": B, "-i": B, "type": S, "head_limit": N, "offset": N, "multiline": B}),
    "Glob": (["pattern"], {"pattern": S, "path": S}),
}
SCHEMAS = [{"name": n, "parameters": {"type": "object", "required": r, "properties": p}} for n, (r, p) in TOOLS.items()]
MARKUP = re.compile(r"</?(?:parameter|function|tool_call|invoke|arguments)\b|<\|im_|</(?:%s)>" % "|".join(
    sorted({k for _, p in TOOLS.values() for k in p}, key=len, reverse=True)))


def parse(text, how):
    p = OutputParser(thinking=False, tools=SCHEMAS, stream_tools=(how == "stream"), recover=True)
    pieces = [text] if how == "whole" else [text[i:i + 5] for i in range(0, len(text), 5)]
    events = []
    for piece in pieces:
        events += p.feed(piece)
    events += p.finish("stop")
    return [e.call for e in events if e.kind == "tool_call"]


def flaws(call):
    out = []
    if call.name not in TOOLS:
        return out
    out += ["missing:" + k for k in TOOLS[call.name][0] if k not in call.arguments]
    out += ["markup:" + k for k, v in call.arguments.items() if isinstance(v, str) and MARKUP.search(v)]
    return out


def main(argv):
    verbose = "-v" in argv
    paths = [a for a in argv if a != "-v"] or [os.path.expanduser("~/.cache/pai/strata-raw")]
    files = sorted(f for p in paths for f in (glob.glob(os.path.join(p, "raw-*.jsonl")) if os.path.isdir(p) else [p]))
    calls = flagged = lines = 0
    ids = {}
    for f in files:
        for line in open(f):
            if not line.strip():
                continue
            row = json.loads(line)
            lines += 1
            for how in ("whole", "stream"):
                for c in parse(row["raw_text"], how):
                    calls += 1
                    fl = flaws(c)
                    if fl:
                        flagged += 1
                        ids.setdefault(row["id"], set()).update(fl)
                        if verbose:
                            print(row["id"], how, c.name, fl)
    print("lines=%d parsed_calls=%d (whole+stream) flagged_calls=%d flagged_ids=%d" % (lines, calls, flagged, len(ids)))
    for i, fl in sorted(ids.items()):
        print(" ", i, sorted(fl))


if __name__ == "__main__":
    main(sys.argv[1:])
