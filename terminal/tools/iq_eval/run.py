r"""Run the IQ-eval subsets through one engine config and record every answer.

    python run.py CFG --out runs/B.jsonl [--subsets D:\exstrata-work\evalsets\subsets.jsonl] [--port 8090]
                  [--max-tokens 4096] [--exe PATH] [--xargs "a;b"] [--env K=V ...] [--sets gsm8k,math500] [--limit N]
    python run.py --base-url http://127.0.0.1:8090 --out runs/B.jsonl     # a server that is already running

The server is started like tools/ab_engine.py does (serve/server.py --engine strata --config CFG --port P, then wait_ready).  Each item is one
POST /v1/chat/completions: greedy (temperature 0), the same max_tokens, the model's default thinking (no chat_template_kwargs), the prompt exactly
as subsets.py wrote it.  One JSON line per item: id, set, output (the message content), reasoning, finish_reason, prompt_tokens,
completion_tokens, seconds.  Resumable: ids already in --out (and without an error) are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
ROOT = TOOLS.parent
DEFAULT_SUBSETS = r"D:\exstrata-work\evalsets\subsets.jsonl"


def post(base_url: str, prompt: str, max_tokens: int, timeout: float = 3600) -> dict:
    # Same shape as ab_engine.post(), minus chat_template_kwargs: the model's default thinking stays on.
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def load_done(out: Path) -> set[str]:
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue                                    # a half-written last line: that item runs again
            if "error" not in row:
                done.add(row["id"])
    return done


def run_items(base_url: str, items: list[dict], out: Path, max_tokens: int, retries: int = 2) -> int:
    done = load_done(out)
    todo = [x for x in items if x["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for n, it in enumerate(todo, 1):
            t0 = time.time()
            row = {"id": it["id"], "set": it["set"], "max_tokens": max_tokens}
            for attempt in range(retries + 1):
                try:
                    r = post(base_url, it["prompt"], max_tokens)
                    msg, ch = r["choices"][0]["message"], r["choices"][0]
                    u = r.get("usage", {})
                    row.update(output=msg.get("content") or "", reasoning=msg.get("reasoning_content") or "",
                               finish_reason=ch.get("finish_reason"), prompt_tokens=u.get("prompt_tokens"),
                               completion_tokens=u.get("completion_tokens"))
                    row.pop("error", None)
                    break
                except Exception as e:                       # a failed request is written with "error" and runs again on resume
                    row["error"] = f"{type(e).__name__}: {e}"
                    time.sleep(2)
            row["seconds"] = round(time.time() - t0, 3)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n}/{len(todo)}] {it['id']} {row.get('completion_tokens')} tok {row['seconds']} s"
                  f"{' ERROR' if 'error' in row else ''}", flush=True)
    return len(todo)


def start_server(a) -> subprocess.Popen:
    sys.path.insert(0, str(TOOLS))
    from ab_engine import wait_ready                         # the same readiness check as the A/B tool
    cfg = json.loads(Path(a.config).read_text())
    if a.exe:
        cfg["exe"] = a.exe
    if a.xargs:
        cfg["args"] = list(cfg["args"]) + a.xargs.split(";")
    cfg["port"] = a.port
    cfg["log"] = str(Path(tempfile.gettempdir()) / f"iq_eval_{a.port}.log")
    Path(cfg["log"]).write_text("", encoding="utf-8")
    cpath = Path(tempfile.gettempdir()) / f"iq_eval_{a.port}.json"
    cpath.write_text(json.dumps(cfg, indent=1))
    env = dict(os.environ, **dict(x.split("=", 1) for x in a.env))
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cpath),
                             "--port", str(a.port)], cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.time()
        wait_ready(a.port, proc)
        print(f"server ready in {time.time() - t0:.0f} s", flush=True)
    except BaseException:
        proc.terminate()
        raise
    return proc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subsets", default=DEFAULT_SUBSETS)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--base-url", help="use a server that is already running instead of starting one")
    ap.add_argument("--exe")
    ap.add_argument("--xargs")
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    ap.add_argument("--sets", help="comma-separated subset names")
    ap.add_argument("--limit", type=int, help="only the first N items of each set (a smoke test)")
    a = ap.parse_args(argv)
    if not a.base_url and not a.config:
        ap.error("give CFG or --base-url")
    items = [json.loads(x) for x in Path(a.subsets).read_text(encoding="utf-8").splitlines() if x.strip()]
    if a.sets:
        items = [x for x in items if x["set"] in a.sets.split(",")]
    if a.limit:
        seen: dict[str, int] = {}
        keep = []
        for x in items:
            seen[x["set"]] = seen.get(x["set"], 0) + 1
            if seen[x["set"]] <= a.limit:
                keep.append(x)
        items = keep
    proc = None if a.base_url else start_server(a)
    try:
        run_items(a.base_url or f"http://127.0.0.1:{a.port}", items, Path(a.out), a.max_tokens)
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(60)
            except subprocess.TimeoutExpired:
                proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
