#!/usr/bin/env python3
"""RES-20261029-01 (TASK-20261029-07): the combined serve stacks, derived from v100host's Swift 1.5 IQ3_S serve config
(strata-swift-iq3_s-serve.json, 262K x 2) so that only the named fields change. Dev rel build throughout.
  serve-ref  the config as setup wrote it (2 slots)
  serve-s2   V1 (PR #3411's 13 Volta env switches) + V2 (--kv-resident 32768), 2 slots
  serve-s1   S2 + V3 (one slot)
Both stacks also carry --vram-reserve-mib 768, the engine's own advice for the <512 MiB-free start (#2065).
usage: make-stacks.py OUTDIR LOGDIR"""
import copy, json, os, sys

PROD = "<strata>"
REL = "<strata-dev>/build-v100-rel/strata"
V1 = {"STRATA_SIDE_MIXER": "2", **{f"STRATA_{k}": "1" for k in (
    "GR_UP75 B6_MMVQ_ROWS GU_SWIGLU QSA_ROWS ATTN_DAILY SM70_TABLE GR_NORM_FAST ARGMAX_ROWS_SM70 QIL EXPDEC "
    "GR_DOWN_TG DEQUANT_WIDE").split()}}
STACK_ARGS = ["--kv-resident", "32768", "--vram-reserve-mib", "768"]
ARMS = {   # name: (extra env, extra engine args, config keys)
    "serve-ref": ({}, [], {}),
    "serve-s2": (V1, STACK_ARGS, {}),
    "serve-s1": (V1, STACK_ARGS, {"parallel": 1}),
}

out, logdir = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
src = json.load(open(f"{PROD}/strata-swift-iq3_s-serve.json"))
for name, (env, args, keys) in ARMS.items():
    c = copy.deepcopy(src)
    c["exe"] = REL
    c["args"] = c["args"] + args
    c["args_written"] = list(c["args"])
    c["env"] = {**(c.get("env") or {}), **env}
    c.update(keys)
    c["log"] = f"{logdir}/engine-{name}.log"
    c["port"] = 8081
    json.dump(c, open(f"{out}/{name}.json", "w"), indent=2)
    print(name, " ".join(args), keys or "", f"{len(env)} env")
