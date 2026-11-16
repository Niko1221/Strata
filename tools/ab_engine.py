"""A/B two engine builds or environments on one config: restart the server per arm, alternate arms over rounds.

    python tools/ab_engine.py strata-q2_0.json A:exe=/path/a B:exe=/path/b,STRATA_EXP_MODE=0 [--rounds 3]

An arm is NAME:key=value,...; `exe` replaces the config's engine, `xargs` appends ';'-separated engine arguments,
every other key is set in the engine's environment.  Each round starts the server once per arm (the arms' order alternates), warms it up with one short
answer, then sends the same requests: greedy, fixed prompts, a short chat set and one long prompt.  Prints the
server's own timings (llama.cpp's `timings`: decode and prompt tokens per second) per request and the per-arm
medians.  Decode numbers move run to run (the adaptive expert tier, the OS), so compare medians over rounds.

Versions mode compares Strata releases (engine + control of one release, with that release's own defaults), each in its own
home under --bench-dir, and writes results.json and a report there:

    python tools/ab_engine.py --model-path <gguf|folder> --engines 0.1.55,0.1.56 [--tune quick] [--rounds 3]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = (
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
    "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
    "List ten practical tips for learning a new language as an adult, one sentence each.",
)


def long_prompt(words: int) -> str:
    rng = random.Random(7)
    vocab = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember "
             "ledger quartz meadow anchor violet thunder").split()
    body = " ".join(rng.choice(vocab) for _ in range(words))
    return f"Here is a list of words:\n{body}\n\nSummarize what kinds of words appear in the list, in five sentences."


def post(port: int, prompt: str, max_tokens: int) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                         "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def wait_ready(port: int, proc: subprocess.Popen, timeout: float = 1800) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    post(port, "Say hi.", 8)
                    return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError("server not ready")


def run_requests(port: int, reqs: list[tuple[str, str]], max_new: int, rnd: int, arm: str, out) -> list[dict]:
    """Send `reqs` once each; every row goes to `out` (jsonl) and to the screen as it comes in."""
    rows = []
    for tag, p in reqs:
        r = post(port, p, max_new)
        t = r.get("timings", {})
        row = {"round": rnd, "arm": arm, "req": tag, "decode": t.get("predicted_per_second"),
               "prompt": t.get("prompt_per_second"), "prompt_n": t.get("prompt_n"),
               "n": r.get("usage", {}).get("completion_tokens"),
               "text": r["choices"][0]["message"].get("content", "")[:80]}
        out.write(json.dumps(row) + "\n")
        out.flush()
        rows.append(row)
        print(f"  {tag:6s} decode {row['decode']} tok/s  prompt {row['prompt']} tok/s ({row['prompt_n']})  "
              f"n {row['n']}", flush=True)
    return rows


def detect() -> tuple[str, str | None]:
    """(target, engine family) of this machine, e.g. ("windows-x86_64", "cuda")."""
    import ab_versions
    ab_versions._control_on_path()
    from strata_control.engine_component import detect_backend
    from strata_control.targets import host_platform
    return host_platform(), detect_backend()[0]


def gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True,
                              text=True, timeout=30).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return ""


def fail(msg: str):
    raise SystemExit(f"error: {msg}")


def pick_pairs(releases: list[dict], target: str, backend: str, spec: str) -> tuple[str, list, dict]:
    """(backend, versions, pairs) for `spec`. A bare family ("cuda") is matched to the asset variants that carry every
    version asked for ("cuda-cuda13"); several such variants -> an error naming them."""
    import ab_versions
    backends = {backend}
    if "-" not in backend:
        found = (re.fullmatch(rf"strata-engine-[\d.]+-{re.escape(target)}-({re.escape(backend)}(?:-[\w.]+)?)\.tar\.gz", x["name"])
                 for rel in releases for x in rel.get("assets", []))
        backends = {m[1] for m in found if m} or backends
    fits, errors = {}, []
    for b in sorted(backends):
        pairs = ab_versions.pairs_from_releases(releases, target, b)
        try:
            versions = ab_versions.resolve_versions(spec, pairs)
        except ValueError as e:
            errors.append(f"{b}: {e}")
            continue
        if not versions:
            fail(f"--engines {spec!r} names no version")
        fits[b] = (versions, pairs)
    if not fits:
        fail("; ".join(errors))
    if len(fits) > 1:
        fail(f"several engine builds fit {target}/{backend}; pick one with --backend: {', '.join(fits)}")
    (b, (versions, pairs)), = fits.items()
    return b, versions, pairs


def run_versions(a: argparse.Namespace) -> int:
    """Each Strata version (engine + control of one release) in its own home under the bench dir, with that release's own defaults."""
    import ab_report
    import ab_versions as av
    model = Path(a.model_path).absolute()                 # live_roots finds the data folder from the parents
    if not model.exists():
        fail(f"--model-path {model} does not exist")
    bench = Path(a.bench_dir).expanduser()
    target, family = detect()
    backend = a.backend or family
    if not backend:
        fail("no graphics card found; name the engine build with --backend (e.g. cuda-cuda13)")
    releases = av.list_releases()
    backend, versions, pairs = pick_pairs(releases, target, backend, a.engines)
    reqs = [(f"chat{i}", p) for i, p in enumerate(PROMPTS)] + [("long", long_prompt(a.long_words))]
    print(f"{len(versions)} versions ({', '.join(versions)}) for {target}/{backend}: {len(versions) * a.rounds} starts, "
          f"{len(versions) if a.tune else 0} tunings.", flush=True)
    try:
        av.assert_idle([a.port])
    except av.Busy as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    roots = av.live_roots(os.environ, model)
    before = av.snapshot(roots)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = bench / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    data = bench / "data"
    data.mkdir(parents=True, exist_ok=True)
    arms = {v: {"version": v, "status": "skipped", "reason": "the run stopped before it", "setup_s": None, "start_s": [], "rows": []} for v in versions}
    homes, env, pkgs, dead = {}, {}, {}, set()
    code = 0
    out = open(run_dir / "ab_engine.jsonl", "a")
    try:
        for v in versions:
            homes[v] = bench / "homes" / v
            try:
                pkgs[v] = av.ensure_packages(pairs[v], bench / "packages")
                env[v] = (av.provision_home(homes[v], pkgs[v]), av.isolated_env(homes[v], os.environ))
            except Exception as e:                                  # noqa: BLE001 - one version that cannot be set up skips, the rest run
                arms[v].update(reason=f"setup: {e}")
                dead.add(v)
        for rnd in range(a.rounds):
            order = versions if rnd % 2 == 0 else versions[::-1]
            for v in order:
                arm = arms[v]
                if v in dead:
                    continue
                control, venv = env[v]
                try:
                    r = av.launch(control, venv, model, data, a.port, a.tune if rnd == 0 else None)
                    av.collect_configs(homes[v], bench / "configs" / v / stamp)
                    if rnd == 0:
                        arm["setup_s"] = round(r.seconds, 1)
                    elif r.ok:
                        arm["start_s"].append(round(r.seconds, 1))
                    reason = av.version_problem(homes[v], pkgs[v]) if r.ok else r.reason
                    if reason:
                        arm.update(status="skipped", reason=reason)
                        dead.add(v)
                        print(f"round {rnd} v{v}: skipped: {reason}", flush=True)
                        continue
                    print(f"round {rnd} v{v}: ready in {r.seconds:.0f} s", flush=True)
                    post(a.port, "Say hi.", 8)                       # warm-up, as the config mode's wait_ready does
                    arm["rows"] += [{k: row[k] for k in ("round", "req", "decode", "prompt", "prompt_n", "n")}
                                    for row in run_requests(a.port, reqs, a.max_new, rnd, f"v{v}", out)]
                    arm.update(status="ran", reason="")
                finally:
                    av.stop(control, venv)                           # also after a failed or timed-out start: nothing survives into the next arm
    except KeyboardInterrupt:
        print("\ninterrupted; the results so far are kept", file=sys.stderr)
        code = 130
    finally:
        out.close()
        diff = av.changed(before, av.snapshot(roots))
        results = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model.name, "gpu": gpu_name(), "tune": a.tune,
                   "rounds": a.rounds, "requests": [t for t, _ in reqs], "arms": list(arms.values()), "untouched": not diff,
                   "changed": diff}
        (run_dir / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        html, png = ab_report.write_report(results, run_dir)
        print(f"\nresults: {run_dir / 'results.json'}\nreport:  {html}" + (f"\ncard:    {png}" if png else ""))
    if diff:
        print("error: files of the real Strata install changed during the run:\n  " + "\n  ".join(diff), file=sys.stderr)
        return 1
    return code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("arms", nargs="*")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--long-words", type=int, default=6000)
    ap.add_argument("--log", default="ab_engine.jsonl")
    ap.add_argument("--model-path", help="versions mode: the model (GGUF file or folder); only read")
    ap.add_argument("--engines", help="versions mode: engine versions, `0.1.55,0.1.56` or `0.1.50..0.1.56`")
    ap.add_argument("--tune", choices=("quick", "thorough"), help="versions mode: each version tunes once before round 0")
    ap.add_argument("--bench-dir", default=str(Path.home() / "strata-ab"), help="versions mode: packages, homes, configs, runs")
    ap.add_argument("--backend", help="versions mode: engine build, e.g. cuda-cuda13 (default: this machine's)")
    a = ap.parse_intermixed_args(argv)
    if a.config is not None and (a.model_path or a.engines):
        ap.error("use either <config> <arms...> or --model-path with --engines, not both")
    if a.config is None and not (a.model_path and a.engines):
        ap.error("give <config> <arms...>, or --model-path and --engines")
    if a.config is not None and not a.arms:
        ap.error("the following arguments are required: arms")
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    if a.config is None:
        return run_versions(a)
    base = json.loads(Path(a.config).read_text())
    arms = []
    for s in a.arms:
        name, _, kv = s.partition(":")
        env = dict(x.split("=", 1) for x in kv.split(",") if x)
        arms.append((name, env))
    reqs = [(f"chat{i}", p) for i, p in enumerate(PROMPTS)] + [("long", long_prompt(a.long_words))]
    out = open(a.log, "a")
    res: dict = {n: {} for n, _ in arms}
    for rnd in range(a.rounds):
        order = arms if rnd % 2 == 0 else arms[::-1]
        for name, env in order:
            cfg = dict(base)
            e = dict(env)
            if "exe" in e:
                cfg["exe"] = e.pop("exe")
            if "xargs" in e:                       # extra engine arguments, ';'-separated
                cfg["args"] = list(base["args"]) + e.pop("xargs").split(";")
            cfg["port"] = a.port
            cfg["log"] = str(Path(tempfile.gettempdir()) / f"ab_{name}_r{rnd}.log")
            cpath = Path(tempfile.gettempdir()) / f"ab_{name}.json"
            cpath.write_text(json.dumps(cfg, indent=1))
            penv = dict(os.environ, **e)
            proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
                                     str(cpath), "--port", str(a.port)], cwd=str(ROOT), env=penv,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                t0 = time.time()
                wait_ready(a.port, proc)
                print(f"round {rnd} {name}: ready in {time.time() - t0:.0f} s", flush=True)
                for row in run_requests(a.port, reqs, a.max_new, rnd, name, out):
                    res[name].setdefault(row["req"], []).append(row)
            finally:
                proc.terminate()
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                time.sleep(5)
    print("\nmedians over rounds:")
    for tag, _ in reqs:
        cells = []
        for name, _ in arms:
            rows = res[name].get(tag, [])
            d = statistics.median([x["decode"] for x in rows if x["decode"]]) if rows else float("nan")
            pr = statistics.median([x["prompt"] for x in rows if x["prompt"]]) if rows else float("nan")
            cells.append(f"{name}: decode {d:6.1f}  prompt {pr:7.1f}")
        print(f"  {tag:6s} | " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
