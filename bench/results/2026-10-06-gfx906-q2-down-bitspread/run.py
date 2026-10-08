#!/usr/bin/env python3
"""One fresh native --serve measurement; run mode order 13,15,15,13,13,15 manually."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

def parse(stdout, n, count, context):
    lines = stdout.splitlines()
    assert f"READY {context}" in lines, "unexpected context/readiness"
    ids = [int(s.split()[1]) for s in lines if s.startswith("T ")]
    rows = [s.split() for s in lines if s.startswith("DONE ")]
    assert len(rows) == 1
    d = rows[0]
    assert len(d) >= 15 and int(d[1]) == len(ids) == count
    assert int(d[2]) == n and int(d[8]) == 0 and int(d[14]) == n
    assert d[5] == "length"
    pp_ms, tg_ms = float(d[3]), float(d[4])
    assert pp_ms > 0 and tg_ms > 0
    return {"prompt_tokens": n, "output_tokens": count,
            "prompt_ms": pp_ms, "decode_ms": tg_ms,
            "PP": n*1000/pp_ms, "TG": count*1000/tg_ms,
            "accepted": int(d[6]), "offered": int(d[7]),
            "output_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--mode", type=int, choices=[13, 15], required=True)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--context", type=int, default=204800)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    ap.add_argument("--greedy", action="store_true")
    a = ap.parse_args()
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    assert cmd, "supply the engine command after --"
    ids = json.loads(a.ids.read_text())
    assert ids and all(isinstance(x, int) and 0 <= x < 248320 for x in ids)
    assert len(ids) + a.tokens <= a.context
    env = dict(os.environ, STRATA_EXP_MODE=str(a.mode),
               DO_NOT_TRACK="1", HF_HUB_DISABLE_TELEMETRY="1")
    sampling = "temperature=0 top_p=1 top_k=1" if a.greedy else "temperature=1 top_p=0.95 top_k=20"
    request = f"GEN {a.tokens} {sampling} seed={a.seed} "
    request += ",".join(map(str, ids)) + chr(10) + "QUIT" + chr(10)
    # stderr may include local paths; keep it local and review it before sharing.
    with a.output.with_suffix(".stderr").open("x") as err:
        p = subprocess.run(cmd, input=request, text=True, stdout=subprocess.PIPE,
                           stderr=err, env=env, timeout=5400, check=True)
    result = parse(p.stdout, len(ids), a.tokens, a.context)
    result.update(mode=a.mode, seed=a.seed, greedy=a.greedy)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))

if __name__ == "__main__":
    main()
