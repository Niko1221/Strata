#!/usr/bin/env python3
"""RES-20261029-01 G1.3 (TASK-20261029-04): the arm configs, derived from v100host's Strata setup configs for Swift 1.5
IQ3_S (operator 2026-10-29: "model to use/test swift 1.5") so that only the named fields change.
bench = strata-swift-iq3_s.json (32K x 1 slot), serve = strata-swift-iq3_s-serve.json (262K x 2).
usage: make-configs.py OUTDIR LOGDIR"""
import copy, json, os, sys

PROD = "<strata>"
EXE = {"prod": f"{PROD}/engine-cuda12/strata",
       "rel": "<strata-dev>/build-v100-rel/strata",
       "prof": "<strata-dev>/build-v100-prof/strata",
       "nsys": "<harness>/g13/nsys-wrap.sh"}       # runs the prof binary under nsys
REPRO_ENV = {"STRATA_IQ_MT_MIN": "1"}                 # docs/DETAILS.md "Reproducible output"
REPRO_ARGS = ["--prompt-cache", "0", "--adapt-swaps", "0", "--pcie-frac", "0"]
PROFILE_ENV = {"STRATA_VERIFY_PROFILE": "1", "STRATA_DECODE_TIMING": "1"}
ARMS = {   # name: (shape, binary, extra env, extra engine args)
    "bench-prod": ("bench", "prod", {}, []),
    "bench-rel": ("bench", "rel", {}, []),
    "bench-prod-repro": ("bench", "prod", REPRO_ENV, REPRO_ARGS),
    "bench-rel-repro": ("bench", "rel", REPRO_ENV, REPRO_ARGS),
    "bench-prof": ("bench", "prof", PROFILE_ENV, []),
    "bench-nsys": ("bench", "nsys", {}, []),
    "serve-rel": ("serve", "rel", {}, []),
    "serve-prof": ("serve", "prof", PROFILE_ENV, []),
}

out, logdir = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
src = {"bench": json.load(open(f"{PROD}/strata-swift-iq3_s.json")),
       "serve": json.load(open(f"{PROD}/strata-swift-iq3_s-serve.json"))}
for name, (shape, binary, env, args) in ARMS.items():
    c = copy.deepcopy(src[shape])
    c["exe"] = EXE[binary]
    c["args"] = c["args"] + args
    c["args_written"] = list(c["args"])
    c["env"] = {**(c.get("env") or {}), **env}
    if binary == "nsys":                                  # the wrapper's session name, read by g13-run.sh
        c["env"]["NSYS_SESSION"] = "g13"
    c["log"] = f"{logdir}/engine-{name}.log"
    c["port"] = 8081
    path = f"{out}/{name}.json"
    json.dump(c, open(path, "w"), indent=2)
    print(path, c["exe"], " ".join(args), env or "")
