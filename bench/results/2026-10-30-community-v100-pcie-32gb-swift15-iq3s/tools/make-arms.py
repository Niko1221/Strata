#!/usr/bin/env python3
"""RES-20261029-01 G2 flag trials (TASK-20261029-06): the arm configs, derived from v100host's Strata setup configs for
Swift 1.5 IQ3_S so that only the named fields change. All arms run the dev rel build (`build-v100-rel`, parity with
production checked in G1.3). bench = strata-swift-iq3_s.json (32K x 1), serve = strata-swift-iq3_s-serve.json (262K x 2).
usage: make-arms.py OUTDIR LOGDIR"""
import copy, json, os, sys

PROD = "<strata>"
EXE = {"rel": "<strata-dev>/build-v100-rel/strata", "prof": "<strata-dev>/build-v100-prof/strata",
       "nsys": "<harness>/g13/nsys-wrap.sh"}
REPRO_ENV = {"STRATA_IQ_MT_MIN": "1"}                 # docs/DETAILS.md "Reproducible output"
REPRO_ARGS = ["--prompt-cache", "0", "--adapt-swaps", "0", "--pcie-frac", "0"]
# V1: PR #3411's Volta stack, every switch already in 0.1.52
V1 = {"STRATA_SIDE_MIXER": "2", **{f"STRATA_{k}": "1" for k in (
    "GR_UP75 B6_MMVQ_ROWS GU_SWIGLU QSA_ROWS ATTN_DAILY SM70_TABLE GR_NORM_FAST ARGMAX_ROWS_SM70 QIL EXPDEC "
    "GR_DOWN_TG DEQUANT_WIDE").split()}}
# V1x: + the switches that only the sm_60 / sm_86 default tables carry (nv_defaults.cpp, "identity gate IDENTICAL")
V1X = {**V1, "STRATA_FUSE_WAIT": "2", "STRATA_FUSE_DECODE": "1", "STRATA_CPU_IQ_ALL_EXACT1": "1"}
V4 = {"STRATA_ADAPT_NOWAIT": "1", "STRATA_ADAPT_LAG": "2"}   # both tables' largest item
NODE0_CORES = ",".join(str(c) for c in range(1, 20))     # node 0's physical cores but the host's (0)
ARMS = {   # name: (shape, binary, extra env, extra engine args, replaced engine args, config keys)
    "bench-ref": ("bench", "rel", {}, [], {}, {}),
    "bench-v1": ("bench", "rel", V1, [], {}, {}),
    "bench-v1x": ("bench", "rel", V1X, [], {}, {}),
    "bench-ref-repro": ("bench", "rel", REPRO_ENV, REPRO_ARGS, {}, {}),
    "bench-v1-repro": ("bench", "rel", {**V1, **REPRO_ENV}, REPRO_ARGS, {}, {}),
    "bench-v1x-repro": ("bench", "rel", {**V1X, **REPRO_ENV}, REPRO_ARGS, {}, {}),
    "bench-nsys": ("bench", "nsys", {"NSYS_SESSION": "g13"}, [], {}, {}),
    "serve-ref": ("serve", "rel", {}, [], {}, {}),
    "serve-v1": ("serve", "rel", V1, [], {}, {}),
    "serve-v4": ("serve", "rel", V4, [], {}, {}),
    "serve-v2": ("serve", "rel", {}, ["--kv-resident", "32768"], {}, {}),
    "serve-v3": ("serve", "rel", {}, [], {}, {"parallel": 1}),
    "serve-pc0": ("serve", "rel", {}, ["--pool-cores", NODE0_CORES], {}, {}),
    "serve-pf32k": ("serve", "rel", {}, [], {"--prefill": "auto:32768"}, {}),
}

out, logdir = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
src = {"bench": json.load(open(f"{PROD}/strata-swift-iq3_s.json")),
       "serve": json.load(open(f"{PROD}/strata-swift-iq3_s-serve.json"))}
for name, (shape, binary, env, args, repl, keys) in ARMS.items():
    c = copy.deepcopy(src[shape])
    c["exe"] = EXE[binary]
    a = list(c["args"])
    for flag, val in repl.items():
        a[a.index(flag) + 1] = val
    c["args"] = a + args
    c["args_written"] = list(c["args"])
    c["env"] = {**(c.get("env") or {}), **env}
    c.update(keys)
    c["log"] = f"{logdir}/engine-{name}.log"
    c["port"] = 8081
    json.dump(c, open(f"{out}/{name}.json", "w"), indent=2)
    print(name, c["exe"].rsplit("/", 2)[-2], " ".join(args), repl or "", keys or "", " ".join(f"{k}={v}" for k, v in env.items()))
