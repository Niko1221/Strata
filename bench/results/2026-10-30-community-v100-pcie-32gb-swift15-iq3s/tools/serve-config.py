#!/usr/bin/env python3
"""TASK-20261026-04: make a v100host serving config from a Strata setup config, with the IQ3_S serving shape: 262,144
context, 2 parallel slots, the thinking controller (budget 28672, close-retry, loop recovery), a -serve log.
Usage: serve-config.py strata-swift-iq3_s.json strata-swift-iq3_s-serve.json"""
import json, sys

src, dst = sys.argv[1], sys.argv[2]
c = json.load(open(src))
for key in ("args", "args_written"):
    a = c.get(key) or []
    if "--max-context" in a:
        a[a.index("--max-context") + 1] = "262144"
    else:
        a += ["--max-context", "262144"]
c["log"] = c["log"].replace(".log", "-serve.log")
c.update({"parallel": 2, "reasoning_budget_tokens": 28672, "reasoning_close_retry": True,
          "reasoning_loop_recovery": "recover"})
json.dump(c, open(dst, "w"), indent=2)
print("wrote", dst)
