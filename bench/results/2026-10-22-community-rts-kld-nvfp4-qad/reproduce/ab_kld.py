#!/usr/bin/env python3
"""Driver for this report: quant_quality.py with per-arm env and one tested flag allowed through FIXED.

Reproduction driver for bench/results/2026-10-22-community-rts-kld-nvfp4-qad - a thin wrapper over
the committed yardstick (tools/quant_quality.py): its engine_args/FIXED/FIXED_ENV/read_run/compare
(the engine start is a thin copy of quant_quality run) plus a --test-flag (lets the
config's --pcie-frac win over the yardstick FIXED 0 - the skip fires either way, its scan runs before
the PCIe/CPU placement decision, but with 0 every remaining miss goes to the CPU and the arms leave
the streaming regime the guard exists for), the identity stamp each arm carries, and the STRATA_LOGPOS* guard on
--env (the per-arm env itself is the yardstick's own feature).

    run     one arm: start the engine on the arm's config, teacher-force the fixed id set through the decode
            windows with STRATA_LOGPOS (quant_quality.py's yardstick), and write the logpos tsv + an out json
            carrying the arm identity (label, model, context, build, corpus, topk, positions, perplexity).
    compare KL(base || arm) + top-1 agreement + perplexity for every arm against the base arm, from the tsvs
            (quant_quality.read_run/compare - the same math the QUANTS.md table uses).

The engine is started by THIS script on the arm's config (like quant_quality run and setup.py --calibrate),
so stop any serving model first - the arm's flag/env value is already in the config.
The yardstick's FIXED args (prompt-cache 0, short-read, prefill 512, ...) stay fixed EXCEPT the flag under
test: sweeping --kv or --pcie-frac must reach the engine, while --max-context/--short-read/--prefill/
--prompt-cache are refused as tested knobs (they size or gate the measurement itself).

Note: the context field of the identity blocks committed with this report was stamped from the config's
--max-context (262144); the engine ran the yardstick-fixed --max-context 2048 (seg+512). This driver now
stamps the value the engine actually received.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent   # bench/results/<session>/reproduce
ROOT = HERE.parents[3]                             # the repository root
sys.path.insert(0, str(ROOT / "tools"))
import quant_quality as QQ  # noqa: E402

V1_IDS = ROOT / "data" / "quant_quality" / "v1" / "set.ids.json"
# knobs the measurement itself depends on: sweeping them changes the yardstick, not the model's quality
REFUSED = {"--max-context", "--short-read", "--prefill", "--prompt-cache"}


def corpus_hash(ids_path: pathlib.Path) -> str:
    return hashlib.sha256(ids_path.read_bytes()).hexdigest()[:8]


def arm_args(cfg: dict, gpu_experts, seg_tokens: int, test_flag: str) -> list[str]:
    """quant_quality's engine_args, then the tested flag's value from the arm's config wins over FIXED
    (engine_args drops it with the rest of DROP_VALUE). test_flag "" keeps the yardstick untouched."""
    args = QQ.engine_args(cfg, gpu_experts, seg_tokens)
    if test_flag and test_flag not in REFUSED:
        base = cfg.get("args", [])
        val = base[base.index(test_flag) + 1] if test_flag in base else None
        if val is None:
            return args                                        # the arm leaves the flag unset (a "default" arm)
        if test_flag in args:                                  # a FIXED entry (e.g. --kv fp16): the arm's value wins
            args[args.index(test_flag) + 1] = str(val)
        else:
            args += [test_flag, str(val)]
    return args


def cmd_run(a) -> int:
    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    st = json.load(open(a.ids, encoding="utf-8"))
    out = pathlib.Path(a.out).resolve()
    tsv = pathlib.Path(a.tsv).resolve()
    if tsv.exists():
        tsv.unlink()                                           # a stale tsv would score as this arm's
    if a.test_flag in REFUSED:
        raise SystemExit(f"--test-flag {a.test_flag}: the KLD yardstick fixes it (it sizes/gates the measurement)")
    env = dict(os.environ)
    env.update(cfg.get("env", {}))
    if cfg.get("lib_dirs"):
        env["LD_LIBRARY_PATH"] = ":".join(cfg["lib_dirs"] + [env.get("LD_LIBRARY_PATH", "")])
    env.update(QQ.FIXED_ENV)
    for kv in a.env or []:
        k, _, v = kv.partition("=")
        if k.startswith("STRATA_LOGPOS"):
            raise SystemExit(f"--env {k}: the yardstick owns it (it writes the measurement)")
        env[k] = v                                           # arm env after FIXED_ENV (e.g. STRATA_ROUTE_TAIL_SKIP=7)
    env["STRATA_LOGPOS"] = str(tsv)
    env["STRATA_LOGPOS_TOPK"] = str(QQ.TOPK)
    seg_tokens = max(len(s["ids"]) for s in st["segments"])
    args = arm_args(cfg, a.gpu_experts or False, seg_tokens, a.test_flag)
    errf = open(str(tsv) + ".stderr.txt", "w")
    print("engine:", " ".join(args), flush=True)
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, text=True,
                         cwd=cfg.get("cwd"), env=env, bufsize=1)
    t_start = time.time()

    def lines() -> int:
        return sum(1 for _ in open(tsv)) if tsv.exists() else 0

    segs, info = [], []
    try:
        for line in p.stdout:
            if line.startswith("READY"):
                break
        else:
            raise SystemExit(f"the engine ended before READY (exit {p.wait()}); see {tsv}.stderr.txt")
        for s in st["segments"]:
            n0 = lines()
            p.stdin.write("GEN 1 " + ",".join(map(str, s["ids"])) + "\n")
            p.stdin.flush()
            done = None
            for line in p.stdout:
                if line.startswith(("DONE", "ERR")):
                    done = line.strip()
                    break
            if done is None or not done.startswith("DONE") or "nonfinite" in done:
                raise SystemExit(f"{s['name']}: the engine answered {done!r}; see {tsv}.stderr.txt")
            n1 = lines()
            segs.append({"name": s["name"], "first_line": n0, "end_line": n1})
            info.append(f"{s['name']}: {n1 - n0} positions, {time.time() - t_start:.0f} s")
            print(info[-1], flush=True)
            t_start = time.time()
            if n1 - n0 < len(s["ids"]) // 2:
                raise SystemExit(f"{s['name']}: only {n1 - n0} positions were scored (the prompt took the batched path?)")
    finally:
        try:
            p.stdin.write("QUIT\n")
            p.stdin.flush()
        except (OSError, ValueError):
            pass
        try:
            p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            p.kill()
        errf.close()
    n_pos = segs[-1]["end_line"] if segs else 0
    # quant_quality.read_run pairs the tsv with <tsv>.seg.json - write it or compare cannot find the segments
    pathlib.Path(str(tsv) + ".seg.json").write_text(json.dumps({"segments": segs, "config": a.config,
                                                                 "args": args, "env": []}, indent=1))
    nll = 0.0
    for line in tsv.read_text(errors="replace").splitlines():
        f = line.split("\t")
        if len(f) >= 3:
            nll -= float(f[2])
    ctx = args[args.index("--max-context") + 1] if "--max-context" in args else str(seg_tokens + 512)
    summary = {"measure": "kld", "label": a.label, "model": cfg.get("model_name") or pathlib.Path(a.config).stem,
               "context": ctx, "build": a.build, "prompt_cache": "cold", "corpus": corpus_hash(pathlib.Path(a.ids)),
               "topk": QQ.TOPK, "positions": n_pos, "ppl": math.exp(nll / n_pos) if n_pos else None,
               "tsv": str(tsv), "ids": str(pathlib.Path(a.ids).resolve())}
    out.write_text(json.dumps({"summary": summary, "segments": segs}, indent=1))
    print(f"arm {a.label}: {n_pos} positions, ppl {summary['ppl']}", flush=True)
    return 0


def cmd_compare(a) -> int:
    """KL(base || arm) etc. from the arms' tsvs; a json {label: metrics} plus the shared identity for ab-diff."""
    base = json.load(open(a.base, encoding="utf-8"))["summary"]
    ref = QQ.read_run(base["tsv"])
    res = {"identity": {k: base.get(k) for k in ("build", "model", "context", "corpus", "topk", "prompt_cache")}}
    res["identity"]["base_label"] = base.get("label")
    res["identity"]["base_ppl"] = base.get("ppl")
    print(f"reference: {base['label']} ({base['tsv']})")
    for run in a.runs:
        s = json.load(open(run, encoding="utf-8"))["summary"]
        r = QQ.compare(ref, QQ.read_run(s["tsv"]))["all"]
        res[s["label"]] = {"positions": r["positions"], "top1": round(100 * r["top1"], 2),
                           "kl": round(r["kl"], 6), "kl_p99": round(r["kl_p99"], 6),
                           "ppl": round(r["ppl"], 3), "ppl_ref": round(r["ppl_ref"], 3)}
        print(f"{s['label']:>16s}: pos {r['positions']} top1 {100 * r['top1']:6.2f}% "
              f"mean_kl {r['kl']:.6f} p99_kl {r['kl_p99']:.6f} ppl {r['ppl']:.3f} (ref {r['ppl_ref']:.3f})")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tsv", required=True)
    p.add_argument("--ids", default=str(V1_IDS))
    p.add_argument("--label", default="arm")
    p.add_argument("--build", default="")
    p.add_argument("--test-flag", default="")
    p.add_argument("--gpu-experts", help="--expert-cache N (or auto) instead of one CPU slot")
    p.add_argument("--env", action="append", help="K=v applied AFTER FIXED_ENV (arm env; STRATA_LOGPOS* refused)")
    p = sub.add_parser("compare")
    p.add_argument("--base", required=True, help="the base arm's out json")
    p.add_argument("--runs", nargs="+", required=True, help="the other arms' out jsons")
    p.add_argument("--json")
    a = ap.parse_args()
    return {"run": cmd_run, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
