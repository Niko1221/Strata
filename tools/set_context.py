"""Set the context window of the installed model(s): python tools/set_context.py 131072

`make run CONTEXT=N` calls it.  It changes only --max-context in each strata-*.json (setup's own re-run would also
redo the questions and can drop hand-set keys such as "parallel"), and it stays set for later starts.  Unlike setup it
does not check the memory: a bigger context needs more of it (the KV cache grows with it).  Past the model's trained
262,144 tokens a context needs rope scaling, which setup sets up: ./setup.sh --setup --context N.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINED_CTX = 262144
MIN_CTX = 4096


def set_context(n: int, root: Path = ROOT) -> list[Path]:
    if not MIN_CTX <= n <= TRAINED_CTX:
        raise ValueError(f"the context must be {MIN_CTX}-{TRAINED_CTX} tokens (got {n}); "
                         f"past {TRAINED_CTX}: ./setup.sh --setup --context {n} (rope scaling)")
    changed = []
    for p in sorted(root.glob("strata-*.json")):
        cfg = json.loads(p.read_text(encoding="utf-8"))
        args = cfg.get("args")
        if not isinstance(args, list):
            continue
        if "--max-context" in args:
            args[args.index("--max-context") + 1] = str(n)
        else:
            args += ["--max-context", str(n)]
        p.write_text(json.dumps(cfg, indent=1), encoding="utf-8")   # setup.py writes its configs the same way
        changed.append(p)
    if not changed:
        raise ValueError("no installed model (strata-*.json): run make pull or make setup first")
    return changed


if __name__ == "__main__":
    try:
        n = int(sys.argv[1])
        for p in set_context(n):
            print(f"context: {n:,} tokens in {p.name}")
    except (IndexError, ValueError) as e:
        sys.exit(f"set_context: {e}")
