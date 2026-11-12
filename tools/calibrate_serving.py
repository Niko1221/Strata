#!/usr/bin/env python3
"""A calibration for a Strata server: one request at a time (single) or several cold prompts together
and staggered (multi).  It starts the server from a run config, reads each request's queue wait / prompt
time from the JSON /metrics, and compares env-only arms.

The engine's own `strata engine calibrate` measures its decode at a fixed single-stream workload.  This
tool is the serving-side half: `--mode single` checks the solo path, `--mode multi` checks what parallel
requests actually wait for.  `--single` / `--multi` are the same choice as short flags.

Arms are env-only (the server and engine read them at start):
  baseline         the config as it is
  routed           STRATA_PREFILL_ROUTED_STREAM=1 (the prompt stream drops experts no token routes to)
  routed_adapt     routed + STRATA_PIPE_ADAPT_EVERY=32 (the pipelined groups re-balance the expert tier)
  coalesce         STRATA_COALESCE_PREFILL=4 (prompts that arrive together are read in one pass; multi only)

Only the winner's env is written (with --apply): engine keys into the run config's "env" (with a
.bak-<time> copy), server keys into <config>.server-env.json for the profile/server environment.

Run it with the Strata venv and the same Python that starts the server:
  module/strata/.venv/bin/python module/strata/tools/calibrate_serving.py \
      --config module/strata/strata-nvfp4-nvfp4-131072-p4.json --out data/serving-cal.json --apply --multi

It needs the exclusive GPUs: stop the server for this config first (the launcher's calibration does that).
The engine pins tens of GiB; run it under a scope with OOM protection on this PC, as any Strata start.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "serve" / "server.py"   # the default; --root points the tool at a fork clone

WORD_BANK = (
    "the quick brown fox jumps over lazy dogs while a quiet system measures token "
    "throughput and latency across parallel workers using stable random text "
    "samples benchmarks reveal hidden bottlenecks under sustained concurrent load "
    "every run produces fresh measurements that operators compare across profiles "
    "backends engines quantizations kernels and scheduling policies memory pools "
    "cache pages decode steps prefill batches scheduler queues tokens per second"
).split()

ARMS: dict[str, dict[str, str]] = {
    "baseline": {},
    "routed": {"STRATA_PREFILL_ROUTED_STREAM": "1"},
    "routed_adapt": {"STRATA_PREFILL_ROUTED_STREAM": "1", "STRATA_PIPE_ADAPT_EVERY": "32"},
    "coalesce": {"STRATA_COALESCE_PREFILL": "4"},
}
#: The arms the default run tries; the others are there for --arms.
DEFAULT_ARMS = ("baseline", "routed", "routed_adapt")
#: env keys the *server* process reads at import, not the engine.  Writing these to the run config's
#: "env" only reaches the engine, so --apply writes them to a sidecar for the profile/server env instead.
SERVER_ENV = {"STRATA_COALESCE_PREFILL", "STRATA_ADMIT_SHORT_FIRST"}


def make_prompt(words: int, seed: int) -> str:
    rnd = random.Random(seed)
    text = " ".join(rnd.choice(WORD_BANK) for _ in range(max(1, words)))
    return text + "\n\nSummarize the passage above in one sentence."


def http_json(url: str, timeout: float = 3.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) != 0


class Server:
    def __init__(self, config: Path, port: int, parallel: int, env: dict[str, str], log: Path,
                 cfg_args: list[str] | None = None, root: Path | None = None):
        self.config, self.port, self.parallel, self.env, self.log = config, port, parallel, env, log
        self.cfg_args = cfg_args or []
        self.root = root or ROOT
        self.server = self.root / "serve" / "server.py"
        self.proc: subprocess.Popen | None = None
        self.fh = None

    def start(self, timeout: float) -> str:
        if not port_free(self.port):
            raise RuntimeError(f"port {self.port} is already in use (stop the running server first)")
        argv = [sys.executable, str(self.server), "--config", str(self.config), "--engine", "strata"]
        args = self.cfg_args
        if "--batch" not in args and "--slots" not in args:
            argv += ["--parallel", str(self.parallel)]
        if "--batch-groups" not in args:
            argv += ["--batch-groups", "auto"]
        argv += ["--host", "127.0.0.1", "--port", str(self.port)]
        env = dict(os.environ)
        env.update(self.env)
        self.fh = self.log.open("wb")
        self.proc = subprocess.Popen(argv, cwd=str(self.root), env=env, stdout=self.fh,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"the server exited during start (see {self.log})")
            try:
                doc = http_json(f"http://127.0.0.1:{self.port}/v1/models", 2.0)
                models = doc.get("data") or []
                if models:
                    return str(models[0].get("id") or "default")
            except Exception:  # noqa: BLE001 - not ready yet
                pass
            time.sleep(2)
        raise RuntimeError(f"the server did not answer within {timeout:.0f} s (see {self.log})")

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait()
        if self.fh is not None:
            self.fh.close()


def stream_request(base: str, model: str, prompt: str, max_tokens: int, out: dict[int, dict],
                   idx: int, start_at: float) -> None:
    now = time.time()
    if start_at > now:
        time.sleep(start_at - now)
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0, "stream": True}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    rec = {"idx": idx, "s": time.time(), "first": None, "last": None, "gaps": [],
           "usage": None, "err": None, "n": 0}
    last = None
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if rec["first"] is None:
                    rec["first"] = time.time()
                if obj.get("usage"):
                    rec["usage"] = obj["usage"]
                ch = obj.get("choices") or []
                if not ch:
                    continue
                delta = ch[0].get("delta") or {}
                text = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if text:
                    t = time.time()
                    if last is not None:
                        rec["gaps"].append(t - last)
                    last = t
                    rec["last"] = t
                    rec["n"] += 1
    except Exception as e:  # noqa: BLE001 - a failed request is part of the measurement
        rec["err"] = repr(e)
    rec["e"] = time.time()
    out[idx] = rec


def phases(base: str, requests: list[dict], t0: float) -> None:
    """Match the server's JSON /metrics entries to the requests, and fill arr / p / q / gen."""
    try:
        met = http_json(base + "/metrics?requests=all", 5.0)
    except Exception:  # noqa: BLE001
        return
    pool = [e for e in (met.get("requests") or []) if isinstance(e, dict)
            and isinstance(e.get("time"), (int, float))]
    used: set[int] = set()
    for rec in requests:
        tin = (rec.get("usage") or {}).get("prompt_tokens")
        best = None
        for j, e in enumerate(pool):
            if j in used:
                continue
            dt = float(e["time"]) - rec["s"]
            if not (-0.5 <= dt <= 3.0):
                continue
            if tin is not None and e.get("prompt_tokens") not in (None, tin):
                continue
            if best is None or abs(dt) < best[1]:
                best = (j, abs(dt))
        if not best:
            continue
        used.add(best[0])
        e = pool[best[0]]
        rec["arr"] = round(float(e["time"]) - t0, 2)
        rec["p"] = round(max(0.0, float(e.get("prompt_ms") or 0.0) / 1000.0), 2)
        ft = float(e.get("first_engine_token_s") or 0.0)
        rec["ft"] = round(ft, 2)
        rec["q"] = round(max(0.0, ft - rec["p"]), 2) if ft > 0 else None
        rec["gen"] = e.get("generated")
        if ft > 0:
            rec["first_engine_abs"] = float(e["time"]) + ft   # (arrival + the engine's own TTFT)


def run_probe(base: str, model: str, n: int, words: int, max_tokens: int, gap: float,
              seeds: list[int]) -> dict[str, Any]:
    out: dict[int, dict] = {}
    t0 = time.time()
    threads = [threading.Thread(target=stream_request,
                                args=(base, model, make_prompt(words, seeds[i]), max_tokens, out, i,
                                      t0 + (i * gap if gap > 0 else 0.0)))
               for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    recs = [out[i] for i in range(n) if i in out]
    phases(base, recs, t0)
    rows = []
    for r in recs:
        tok = (r.get("usage") or {}).get("completion_tokens") or r["n"]
        ttft = (r.get("ft") if r.get("ft") is not None else
                ((r["first"] - r["s"]) if r["first"] else None))
        t_first = r.get("first_engine_abs") or r["first"]
        dec = ((tok - 1) / (r["e"] - t_first)) if (t_first and tok > 1 and r["e"] > t_first) else None
        rows.append({"i": r["idx"], "s": round(r["s"] - t0, 2), "ttft": round(ttft, 2) if ttft else None,
                     "q": r.get("q"), "p": r.get("p"), "n": tok,
                     "dec": round(dec, 2) if dec else None,
                     "max_gap": round(max(r["gaps"]), 2) if r["gaps"] else 0.0,
                     "err": r["err"]})
    qs = [r["q"] for r in rows if r["q"] is not None]
    ps = [r["p"] for r in rows if r["p"] is not None]
    ttfts = [r["ttft"] for r in rows if r["ttft"] is not None]
    toks = sum(r["n"] for r in rows)
    return {"wall": round(wall, 2), "rows": rows, "agg_tps": round(toks / wall, 2) if wall > 0 else 0.0,
            "max_q": round(max(qs), 2) if qs else None,
            "sum_q": round(sum(qs), 2) if qs else None,
            "sum_p": round(sum(ps), 2) if ps else None,
            "ttft_p90": round(sorted(ttfts)[max(0, int(len(ttfts) * 0.9) - 1)], 2) if ttfts else None,
            "max_gap": max([r["max_gap"] for r in rows] or [0.0])}


def score(arm: dict[str, Any], base: dict[str, Any], mode: str = "multi") -> tuple[float, str]:
    """Lower is better.  Multi: queue wait, then prompt time, then throughput (never below 97% of the baseline),
    then decode gaps.  Single: throughput first (below 98% of the baseline is refused), the prompt time breaks a
    near-tie.  The components stay in the report so a trade-off is visible, not hidden in one number."""
    tps = float(arm.get("agg_tps") or 0.0)
    b_tps = float(base.get("agg_tps") or 0.0) or 1.0
    if mode == "single":
        if tps < b_tps * 0.98:
            return 1e6 + (b_tps - tps), "throughput below the baseline"
        sp = float(arm.get("sum_p") or 0.0)
        gap = float(arm.get("max_gap") or 0.0)
        return (b_tps / tps if tps > 0 else 1e6) + sp * 0.05 + gap * 0.1, ""
    if tps < b_tps * 0.97:
        return 1e6 + (b_tps - tps), "throughput below the baseline"
    mq = float(arm.get("max_q") or 0.0)
    sp = float(arm.get("sum_p") or 0.0)
    gap = float(arm.get("max_gap") or 0.0)
    return mq * 10.0 + sp * 0.5 + gap, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="the Strata run config (strata-<model>.json)")
    ap.add_argument("--out", required=True, type=Path, help="where the JSON report is written")
    ap.add_argument("--port", type=int, default=1929)
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="the Strata folder to start (default: this tool's module/strata; use module/fork/strata for the fork backend)")
    ap.add_argument("--parallel", type=int, default=None, help="batch slots (the config's parallel is used when omitted)")
    ap.add_argument("--clients", type=int, default=4, help="requests in each probe")
    ap.add_argument("--input-words", type=int, default=1024, help="prompt words (~1.2 tokens each)")
    ap.add_argument("--output-tokens", type=int, default=512)
    ap.add_argument("--gap", type=float, default=5.0, help="seconds between the staggered sends")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--mode", choices=["single", "multi"], default="multi",
                      help="single = one request at a time (the solo path); multi = parallel requests (default)")
    mode.add_argument("--single", action="store_true", help="same as --mode single")
    mode.add_argument("--multi", action="store_true", help="same as --mode multi")
    ap.add_argument("--shape", choices=["both", "stagger", "loop"], default="both",
                    help="multi only: both (loop + stagger), loop, or stagger")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS), help="comma-separated arm names (see ARMS)")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--ready-timeout", type=float, default=600.0)
    ap.add_argument("--apply", action="store_true", help="write the winner's env into the config (with a backup)")
    args = ap.parse_args()

    cfg_path = args.config.resolve()
    cfg = json.loads(cfg_path.read_text("utf-8"))
    mode = "single" if args.single else "multi" if args.multi else args.mode
    shape = "single" if mode == "single" else args.shape
    clients = 1 if mode == "single" else max(1, args.clients)
    parallel = 1 if mode == "single" else (args.parallel or int(cfg.get("parallel") or 4))
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        ap.error(f"unknown arm(s): {', '.join(unknown)}; known: {', '.join(ARMS)}")
    if mode == "single" and "coalesce" in arms:
        print("[cal] coalesce is a multi-request arm; it is skipped in single mode", flush=True)
        arms = [a for a in arms if a != "coalesce"]
    logs = cfg_path.parent / f"{mode}-cal-logs"
    logs.mkdir(exist_ok=True)
    if "STRATA_PREFILL_ROUTED_STREAM" in (cfg.get("env") or {}):
        print("[cal] note: the config already sets STRATA_PREFILL_ROUTED_STREAM; the baseline includes it, "
              "so the routed arm measures the same start", flush=True)
    report: dict[str, Any] = {"schema": 1, "config": str(cfg_path), "engine": "strata", "mode": mode,
                              "shape": shape, "parallel": parallel, "clients": clients,
                              "arms": {}, "winner": None}
    base_agg = None
    try:
        for arm in arms:
            env = dict(ARMS[arm])
            server = Server(cfg_path, args.port, parallel, env, logs / f"{arm}.log",
                            cfg_args=[str(a) for a in (cfg.get("args") or [])], root=args.root.resolve())
            try:
                print(f"[cal] {arm}: starting ...", flush=True)
                model = server.start(args.ready_timeout)
                base = f"http://127.0.0.1:{args.port}"
                # one short warm-up (not measured), then the probes
                for i in range(args.warmup):
                    w: dict[int, dict] = {}
                    stream_request(base, model, make_prompt(args.input_words, 90000 + i), 64, w, 0, 0.0)
                results: dict[str, Any] = {"env": env}
                if mode == "single":
                    results["single"] = run_probe(base, model, 1, args.input_words,
                                                  args.output_tokens, 0.0, [60000])
                else:
                    if shape in ("both", "loop"):
                        results["loop"] = run_probe(base, model, clients, args.input_words,
                                                    args.output_tokens, 0.0, [70000 + i for i in range(clients)])
                    if shape in ("both", "stagger"):
                        results["stagger"] = run_probe(base, model, clients, args.input_words,
                                                       args.output_tokens, args.gap,
                                                       [80000 + i for i in range(clients)])
                # multi scores the stagger shape when it ran (the queue-wait shape), else the loop shape
                key = "single" if mode == "single" else ("stagger" if "stagger" in results else "loop")
                results["score_probe"] = key
                report["arms"][arm] = results
                if arm == "baseline":
                    base_agg = results[key]
                probe = results[key]
                print(f"[cal] {arm}: agg {probe['agg_tps']} tok/s, max queue {probe['max_q']} s, "
                      f"sum prefill {probe['sum_p']} s, max gap {probe['max_gap']} s", flush=True)
            except Exception as e:  # noqa: BLE001 - a failed arm is excluded, the baseline is fatal
                print(f"[cal] {arm}: failed: {e}", flush=True)
                report["arms"][arm] = {"env": env, "error": str(e)[:400]}
                if arm == "baseline":
                    raise
            finally:
                server.stop()
        if base_agg is None:
            raise RuntimeError("the baseline arm did not run")
        scored = []
        for arm, res in report["arms"].items():
            if "error" in res:
                continue
            probe = res[res.get("score_probe", "stagger")]
            s, why = score(probe, base_agg, mode)
            scored.append((s, arm, why, probe))
        scored.sort(key=lambda x: x[0])
        if scored:
            base_score = next((row[0] for row in scored if row[1] == "baseline"), None)
            best_score, winner, why, probe = scored[0]
            # a 2% band: the run-to-run noise is larger than that, and a 0.01 s "win" must not rewrite the config
            if winner != "baseline" and base_score is not None and best_score >= base_score * 0.98:
                winner, why, probe, best_score = "baseline", "no arm beat the baseline by 2%", base_agg, base_score
            improvement = round((base_score - best_score) / base_score * 100, 1) if base_score else 0.0
            report["winner"] = {"arm": winner, "env": ARMS[winner], "score": best_score,
                                "config_env": {k: v for k, v in ARMS[winner].items() if k not in SERVER_ENV},
                                "server_env": {k: v for k, v in ARMS[winner].items() if k in SERVER_ENV},
                                "improvement_pct": improvement, "reason": why, "probe": probe}
            print(f"[cal] winner: {winner} {ARMS[winner]} ({improvement:+.1f}% on the score)", flush=True)
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
        print(f"[cal] report: {args.out}", flush=True)
        if args.apply and report["winner"]:
            env = dict(report["winner"]["env"])
            cfg_env = {k: v for k, v in env.items() if k not in SERVER_ENV}
            server_env = {k: v for k, v in env.items() if k in SERVER_ENV}
            if cfg_env:
                backup = cfg_path.with_suffix(cfg_path.suffix + ".bak-" + time.strftime("%Y%m%d-%H%M%S"))
                shutil.copy2(cfg_path, backup)
                cfg = json.loads(cfg_path.read_text("utf-8"))
                cfg["env"] = {**(cfg.get("env") or {}), **cfg_env}
                tmp = cfg_path.with_suffix(cfg_path.suffix + ".tmp")
                tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", "utf-8")
                tmp.replace(cfg_path)
                print(f"[cal] applied {cfg_env} to {cfg_path} (backup {backup.name})", flush=True)
            elif env:
                print("[cal] the winner has no engine-side env to write", flush=True)
            if server_env:
                sidecar = cfg_path.with_suffix(cfg_path.suffix + ".server-env.json")
                sidecar.write_text(json.dumps(server_env, ensure_ascii=False, indent=1) + "\n", "utf-8")
                print(f"[cal] server-side env {server_env} written to {sidecar}; set it in the launcher "
                      "profile's env (the config's env only reaches the engine)", flush=True)
            if not env:
                print("[cal] the baseline won: nothing written", flush=True)
        return 0
    finally:
        pass
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("[cal] interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as e:  # noqa: BLE001 - a one-line failure, not a traceback for the user
        print(f"[cal] failed: {e}", file=sys.stderr)
        sys.exit(1)
