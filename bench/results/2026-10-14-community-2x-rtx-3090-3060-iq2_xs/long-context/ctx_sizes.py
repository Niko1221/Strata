"""Context-window sweep: prompt and decode speed at a series of prompt sizes.

Same measurement method as tools/strata_bench.py - the same chat()/digest()/stat()/long_prompt()
helpers, the same calibration of tokens per word - only the prompt sizes are the point here.
One config, one server start, one prompt size per configuration (the session carve is allocated
for the whole --max-context at start, so a configuration IS its --max-context).

Run it through ctxtest.sh, which stops the instance on 8090 first: a quiet measurement needs the
cards to itself.

  .venv/bin/python tools/ctx_sizes.py --config bench/ctxconfigs/ctx262144.json --port 8092 \
      --api-key KEY --sizes 262144 --runs 2 --out bench/results/2026-10-14-ctx --quiet
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

_here = pathlib.Path(__file__).resolve().parent
for _cand in (_here, _here.parent / "tools", _here.parent.parent / "tools"):
    if (_cand / "strata_bench.py").exists():      # works from tools/ and from a report's own folder
        sys.path.insert(0, str(_cand))
        break
import strata_bench as B  # noqa: E402  the same helpers, so the numbers are comparable

MAX_NEW = 64               # tokens generated after the long prompt: enough for a decode median
SLACK = 16                 # the server rejects prompt + max_new + 8 > context


def calibrate(base, key, log):
    """Tokens per word, the bench tool's way: two tiny requests, the slope, not the intercept."""
    pts = []
    for words in (200, 800):
        r = B.digest(B.chat(base, key, [{"role": "user", "content": f"[ctx cal {words}]\n" + B.long_prompt(words)}], 1))
        pts.append(r.get("prompt_tokens"))
    slope = (pts[1] - pts[0]) / 600 if all(pts) and pts[1] > pts[0] else None
    tpw = slope or 1.1
    overhead = max(0, (pts[0] or 0) - 200 * tpw) if slope else 40
    log(f"  calibrated: about {tpw:.2f} tokens per word")
    return tpw, overhead


def prompt_for(tokens, run, tpw, overhead):
    words = max(50, int((tokens - overhead) / tpw))
    return [{"role": "user", "content": f"[ctx run{run} {tokens}p]\n" + B.long_prompt(words)}]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="prompt and decode speed at long contexts")
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--sizes", default="262144", help="comma-separated prompt sizes in tokens")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=MAX_NEW,
                    help=f"tokens generated after each prompt (default {MAX_NEW}; the prompt target shrinks by the same amount)")
    ap.add_argument("--ready-timeout", type=float, default=600)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    say = (lambda *x, **k: None) if a.quiet else (lambda *x, **k: print(*x, **k, flush=True))

    cfg_path = pathlib.Path(a.config)
    if not cfg_path.is_absolute() and not cfg_path.exists():
        cfg_path = B.ROOT / cfg_path
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    key = a.api_key or (cfg.get("api_key") or "")
    base = f"http://127.0.0.1:{a.port}"
    sizes = [int(x) for x in a.sizes.split(",") if x.strip()]

    log_path = pathlib.Path("/tmp") / f"ctxsizes-{B.slug(cfg_path.name)}.log"
    say(f"starting {cfg_path.name} on port {a.port} ...")
    proc = B.start_server(cfg_path, a.port, None, log_path)
    if not B.wait_ready(base, key, proc, a.ready_timeout, say):
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:] if log_path.exists() else []
        print(f"the server did not become ready. last log lines:\n{B.redactor(key=key)(chr(10).join(tail))}", file=sys.stderr)
        return 1

    machine = B.collect_machine(base, key, cfg)
    tpw, overhead = calibrate(base, key, say)
    res = {"machine": machine, "max_new": a.max_new, "results": {}}
    for size in sizes:
        target = size - a.max_new - SLACK
        rows = []
        for run in range(a.runs):
            r = B.chat(base, key, prompt_for(target, run, tpw, overhead), a.max_new)
            d = B.digest(r)
            d["run"] = run
            rows.append(d)
            if not d["ok"]:
                say(f"  {size}: run {run + 1} FAILED: {d.get('err')}")
            else:
                say(f"  {size}: run {run + 1}: {d['prompt_tokens']} tokens, prompt {d['prompt_tps']} tok/s "
                    f"({d['reused_tokens'] or 0} reused), first token {d['ttft_s']} s, decode {d['decode_tps']} tok/s")
        good = [r for r in rows if r["ok"]]
        res["results"][str(size)] = {
            "target_tokens": target,
            "requests": rows,
            "prompt_tps": B.stat(r["prompt_tps"] for r in good),
            "ttft_s": B.stat(r["ttft_s"] for r in good),
            "decode_tps": B.stat(r["decode_tps"] for r in good),
            "reused_tokens": B.stat(r["reused_tokens"] or 0 for r in good),
        }

    B.stop_server(proc)
    red = B.redactor(key=key)
    out = pathlib.Path(a.out)
    if not out.is_absolute():
        out = B.ROOT / out
    folder = out / f"{time.strftime('%Y-%m-%d')}-ctx-{B.slug(cfg_path.name, 24)}"
    folder.mkdir(parents=True, exist_ok=True)
    res["engine_log"] = [red(x) for x in B.log_excerpt(log_path)]
    (folder / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    (folder / "engine-log-excerpt.txt").write_text("\n".join(res["engine_log"]), encoding="utf-8")

    lines = [f"# Context sweep: {machine.get('model') or 'unknown'} on {', '.join(machine.get('gpu') or ['?'])}",
             "",
             f"Build {machine.get('strata_version')} / commit {machine.get('strata_commit')}, "
             f"engine {machine.get('engine_version')}. Config: {red(B.launch_line(B.config_view(cfg)))}",
             f"CPU {machine.get('cpu')}, RAM {machine.get('ram_gib')} GiB. "
             f"{a.max_new} tokens generated after each prompt; the server rejects a prompt that does not fit.",
             "",
             "| max-context | prompt tok/s | first token (s) | decode tok/s | reused |",
             "|---|---|---|---|---|"]
    for size in sizes:
        r = res["results"][str(size)]
        lines.append(f"| {size} | {B.fmt(r['prompt_tps'])} | {B.fmt(r['ttft_s'], 2)} | {B.fmt(r['decode_tps'])} | "
                     f"{B.fmt(r['reused_tokens'], 0)} |")
    lines += ["", "Engine lines (the session carve, the cache slots, the KV streaming hit rate):", ""]
    lines += [f"    {x}" for x in res["engine_log"]]
    (folder / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(f"\nreport: {folder / 'README.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
