#!/usr/bin/env python3
"""TASK-20261030-01: production's Strata serving config in the S1 form (RES-20261029-01 TASK-20261029-07; operator
2026-10-30: "Apply S1 (1 slot)"), from the config production serves now. Only these change:
  env    + PR #3411's 13 Volta switches (V1)
  args   + --kv-resident 32768 (the 262K KV in pinned RAM, V2) + --vram-reserve-mib 768 (the engine's low-VRAM advice)
  parallel 2 -> 1 (one slot, V3)
Also writes the two check configs (the current one and the candidate) on port 8081 with their logs in LOGDIR, named
PREFIX-ref and PREFIX-s1 (default prefix qwen; operator 2026-10-30 "Yes, serve Swift" uses it for Swift 1.5 too).
usage: make-s1.py PROD_CONFIG CANDIDATE_OUT CHECK_DIR LOGDIR [PREFIX]"""
import copy, json, os, sys

V1 = {"STRATA_SIDE_MIXER": "2", **{f"STRATA_{k}": "1" for k in (
    "GR_UP75 B6_MMVQ_ROWS GU_SWIGLU QSA_ROWS ATTN_DAILY SM70_TABLE GR_NORM_FAST ARGMAX_ROWS_SM70 QIL EXPDEC "
    "GR_DOWN_TG DEQUANT_WIDE").split()}}
S1_ARGS = ["--kv-resident", "32768", "--vram-reserve-mib", "768"]

prod_path, cand_path, check_dir, logdir = sys.argv[1:5]
prefix = sys.argv[5] if len(sys.argv) > 5 else "qwen"
prod = json.load(open(prod_path))
if any(a in prod["args"] for a in ("--kv-resident", "--vram-reserve-mib")) or prod.get("env"):
    sys.exit("production config already carries S1 settings or an env block: not touched")
cand = copy.deepcopy(prod)
cand["args"] = prod["args"] + S1_ARGS
cand["args_written"] = list(cand["args"])
cand["env"] = dict(V1)
cand["parallel"] = 1
os.makedirs(os.path.dirname(os.path.abspath(cand_path)), exist_ok=True)
json.dump(cand, open(cand_path, "w"), indent=2)
os.makedirs(check_dir, exist_ok=True)
for name, c in ((f"{prefix}-ref", prod), (f"{prefix}-s1", cand)):
    t = copy.deepcopy(c)
    t["port"] = 8081
    t["log"] = f"{logdir}/engine-{name}.log"
    json.dump(t, open(f"{check_dir}/{name}.json", "w"), indent=2)
print("candidate", cand_path, "parallel", cand["parallel"], "args +", " ".join(S1_ARGS), "env", len(cand["env"]))
