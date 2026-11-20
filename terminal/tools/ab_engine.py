"""A/B two engine builds or environments on one config: restart the server per arm, alternate arms over rounds.

    python tools/ab_engine.py strata-q2_0.json A:exe=/path/a B:exe=/path/b,STRATA_EXP_MODE=0 [--rounds 3]

An arm is NAME:key=value,...; `exe` replaces the config's engine, `xargs` appends ';'-separated engine arguments,
every other key is set in the engine's environment.  Each round starts the server once per arm (the arms' order alternates), warms it up with one short
answer, then sends the same requests: greedy, fixed prompts, a short chat set and one long prompt.  Prints the
server's own timings (llama.cpp's `timings`: decode and prompt tokens per second) per request and the per-arm
medians.  Decode numbers move run to run (the adaptive expert tier, the OS), so compare medians over rounds.

Versions mode compares Strata releases.  Each version is the engine and the control of one release, run with that
control's own defaults.  `--tune quick|thorough` tunes each version once, before round 1.

    python terminal/tools/ab_engine.py --model-path <gguf|folder> --engines 0.1.55,0.1.56 [--tune quick] [--rounds 3] \
        [--suite ladder,think,agent] [--context N] [--think-tokens 8192] [--run-code]

`--suite` adds a second start per version and round, with one shared `--context` (default: the model's maximum), that
runs the chosen suites; their texts come from git tag v0.2.12 of this checkout.

`--engines` takes a list (`0.1.55,0.1.56`) or a range (`0.1.50..0.1.56`).  The packages are downloaded from the GitHub
release, checked against its SHA256SUMS.txt and kept in `~/strata-ab` (or `--bench-dir`).  Each version runs in a
fresh home of its own, so the tool does not change your models, settings or engines; it compares the install's files
before and after the run and exits with 1 if any of them changed.  Stop the normal Strata server and close the Strata
app first: the tool refuses to run beside a server.

On NVIDIA it takes the CUDA 13 build when a release has both; use `--backend cuda-cuda12` for an older driver.  On AMD
with several builds, name one with `--backend`.

Results go to `<bench-dir>/runs/<timestamp>/`: results.json, report.html and report.png (the PNG needs Edge, Chrome or
Chromium).  The generated and tuned configs are kept in `<bench-dir>/configs/<version>/<timestamp>/`.  Nothing in the
bench folder is deleted.  The configs can contain the API key of the test home: do not attach that folder to an issue.
Disk: each engine is in `<bench-dir>/packages` and again in each run's home, `<bench-dir>/homes/<version>/<timestamp>/`.
Delete old homes by hand when you no longer need them.
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
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("ladder", "think", "agent")
ROW_KEYS = ("round", "req", "decode", "prompt", "prompt_n", "n", "suite", "ok", "check", "predicted_n", "wall_s")
RUN_CODE_WARNING = "--run-code: code written by the model runs on this PC with your user rights; it is not isolated."
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


def send(port: int, body: dict, timeout: float = 3600) -> dict:
    """POST `body` as post() does. An HTTP error comes back as {"error": ...} so one bad request does not stop the run."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"}


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


def run_requests(port: int, reqs: list[tuple[str, str]], max_new: int, rnd: int, arm: str, out, rows: list | None = None,
                 check=None) -> list[dict]:
    """Send `reqs` once each; every row goes to `out` (jsonl), to the screen and to `rows` (if given) as it comes in, so an
    interrupt keeps the rows already answered. With `check` (response -> (ok, text)) each row also gets suite "base", ok, check."""
    rows = [] if rows is None else rows
    for tag, p in reqs:
        r = post(port, p, max_new)
        t = r.get("timings", {})
        row = {"round": rnd, "arm": arm, "req": tag, "decode": t.get("predicted_per_second"),
               "prompt": t.get("prompt_per_second"), "prompt_n": t.get("prompt_n"),
               "n": r.get("usage", {}).get("completion_tokens"),
               "text": r["choices"][0]["message"].get("content", "")[:80]}
        if check:
            row["suite"] = "base"
            row["ok"], row["check"] = check(r)
        out.write(json.dumps(row) + "\n")
        out.flush()
        rows.append(row)
        print(f"  {tag:6s} decode {row['decode']} tok/s  prompt {row['prompt']} tok/s ({row['prompt_n']})  "
              f"n {row['n']}", flush=True)
    return rows


def run_suite(port: int, reqs: list, rnd: int, arm: str, out, rows: list) -> list[dict]:
    """Send suite requests (ab_suites.Req) once each and check every answer. A request the server refuses is a wrong row,
    not a crash. Rows go to `out`, the screen and `rows` as they come in."""
    for req in reqs:
        t0 = time.monotonic()
        r = send(port, req.body)
        wall = round(time.monotonic() - t0, 1)
        if "error" in r:
            ok, check, t, text = False, r["error"], {}, ""
        else:
            ok, check = req.check(r)
            t, text = r.get("timings", {}), r["choices"][0]["message"].get("content") or ""
        row = {"round": rnd, "req": req.tag, "suite": req.suite, "decode": t.get("predicted_per_second"),
               "prompt": t.get("prompt_per_second"), "prompt_n": t.get("prompt_n"),
               "n": r.get("usage", {}).get("completion_tokens"), "predicted_n": t.get("predicted_n"),
               "wall_s": wall, "ok": ok, "check": check}
        out.write(json.dumps({**row, "arm": arm, "text": text[:200]}) + "\n")
        out.flush()
        rows.append(row)
        print(f"  {req.tag:24s} decode {row['decode']} tok/s  prompt {row['prompt']} tok/s ({row['prompt_n']})  "
              f"n {row['n']}  {wall}s  {'ok' if ok else 'WRONG'} {check}", flush=True)
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


# The tie-break of feed.select_engine / engine_component.resolve (same version: the build for the newer NVIDIA driver) reads the
# packages' manifests; from release asset names alone it is this order. HIP builds depend on the card, so several of them still ask.
BUILD_PREFERENCE = ("cuda-cuda13", "cuda-cuda12")


def pick_pairs(releases: list[dict], target: str, backend: str, spec: str) -> tuple[list, dict]:
    """(versions, pairs) for `spec`. An exact build ("cuda-cuda12") is used as given. A bare family ("cuda") takes, per version, the
    build BUILD_PREFERENCE ranks first, or the only one that version has; several unranked builds -> an error naming them."""
    import ab_versions
    builds = [backend]
    if "-" not in backend:
        found = (re.fullmatch(rf"strata-engine-[\d.]+-{re.escape(target)}-({re.escape(backend)}(?:-[\w.]+)?)\.tar\.gz", x["name"])
                 for rel in releases for x in rel.get("assets", []))
        rank = {b: i for i, b in enumerate(BUILD_PREFERENCE)}
        builds = sorted({m[1] for m in found if m}, key=lambda b: (rank.get(b, len(rank)), b)) or builds
    pairs, choices = {}, {}
    for b in builds:
        for v, pair in ab_versions.pairs_from_releases(releases, target, b).items():
            pairs.setdefault(v, pair)
            choices.setdefault(v, []).append(b)
    try:
        versions = ab_versions.resolve_versions(spec, pairs)
    except ValueError as e:
        fail(f"{e} ({target}/{backend})")
    if not versions:
        fail(f"--engines {spec!r} names no version")
    unclear = [f"{v}: {', '.join(choices[v])}" for v in versions if len(choices[v]) > 1 and choices[v][0] not in BUILD_PREFERENCE]
    if unclear:
        fail(f"several engine builds fit; pick one with --backend ({'; '.join(unclear)})")
    return versions, pairs


def run_versions(a: argparse.Namespace) -> int:
    """Each Strata version (engine + control of one release) in its own home under the bench dir, with that release's own defaults."""
    import ab_report
    import ab_suites                                      # not at the top: ab_suites -> agent_turns_bench -> ab_engine
    import ab_versions as av
    model = Path(a.model_path).absolute()                 # live_roots finds the data folder from the parents
    if not model.exists():
        fail(f"--model-path {model} does not exist")
    context = None
    if a.suite:                                           # before anything is downloaded
        ab_suites.need_corpus()
        context = a.context or ab_suites.model_context(model)
    bench = Path(a.bench_dir or Path.home() / "strata-ab").expanduser()
    roots = av.live_roots(os.environ, model)
    inside = next((r for r in roots if bench.resolve().is_relative_to(Path(r).resolve())), None)
    if inside:
        fail(f"--bench-dir {bench} is inside {inside}, a folder of the real Strata install; pick another folder")
    target, family = detect()
    backend = a.backend or family
    if not backend:
        fail("no graphics card found; name the engine build with --backend (e.g. cuda-cuda13)")
    releases = av.list_releases()
    versions, pairs = pick_pairs(releases, target, backend, a.engines)
    reqs = [(f"chat{i}", p) for i, p in enumerate(PROMPTS)] + [("long", long_prompt(a.long_words))]
    print(f"{len(versions)} versions for {target}/{backend}: {len(versions) * a.rounds} starts, "
          f"{len(versions) if a.tune else 0} tunings.", flush=True)
    if a.suite:
        from agent_turns_bench import SESSIONS
        steps = ab_suites.ladder_steps(context)
        plan = {"ladder": lambda: f"ladder {len(steps)} steps" + (f" ({steps[0]}..{steps[-1]} tokens)" if steps else ""),
                "think": lambda: f"think {len(ab_suites.think_reqs(a.think_tokens, a.run_code))} tasks",
                "agent": lambda: f"agent {sum(len(st) for _, _, st in SESSIONS)} turns"}
        print(f"suites at context {context}: {', '.join(plan[x]() for x in a.suite)}; a second start per version and round.",
              flush=True)
        if a.run_code:
            print(RUN_CODE_WARNING, flush=True)
    for v in versions:
        print(f"  v{v}: {pairs[v].engine_name} ({pairs[v].tag})", flush=True)
    try:
        av.assert_idle([a.port])
    except av.Busy as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    before = av.snapshot(roots)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = bench / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    data = bench / "data"
    data.mkdir(parents=True, exist_ok=True)
    arms = {v: {"version": v, "status": "skipped", "reason": "the run stopped before it", "setup_s": None, "start_s": [], "rows": [],
                "n_ctx": None, "suites_status": "skipped" if a.suite else None,
                "suites_reason": "the run stopped before it" if a.suite else "", "left_out": []} for v in versions}
    homes, env, pkgs, dead, suites_dead = {}, {}, {}, set(), set()
    built: dict[int, tuple[list, list[str]]] = {}         # suite requests per n_ctx; agent_reqs reads git for seconds
    code = 0
    out = open(run_dir / "ab_engine.jsonl", "a")
    try:
        for v in versions:
            homes[v] = bench / "homes" / v / stamp                # fresh per run: no preset or launch config of an earlier run
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
                got: list[dict] = []
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
                    run_requests(a.port, reqs, a.max_new, rnd, f"v{v}", out, got, check=ab_suites.base_check)
                except Exception as e:                               # noqa: BLE001 - one version that crashes skips, the rest run
                    arm.update(status="skipped", reason=f"{type(e).__name__}: {e}")
                    dead.add(v)
                    print(f"round {rnd} v{v}: skipped: {arm['reason']}", flush=True)
                finally:
                    if got:                                          # also the rows answered before an interrupt
                        arm["rows"] += [{k: row[k] for k in ROW_KEYS if k in row} for row in got]
                        if v not in dead:
                            arm.update(status="ran", reason="")
                    av.stop(control, venv)                         # also after a failed or timed-out start: nothing survives into the next arm
                if not a.suite or v in dead or v in suites_dead:
                    continue
                got = []
                try:                                                 # the suites' own start: a failure here never touches base
                    r = av.launch(control, venv, model, data, a.port, None, context=context)
                    if not r.ok:
                        arm.update(suites_status="skipped", suites_reason=r.reason)
                        suites_dead.add(v)
                        print(f"round {rnd} v{v}: suites skipped: {r.reason}", flush=True)
                        continue
                    n_ctx = arm["n_ctx"] = av.served_context(a.port)
                    n = n_ctx or context
                    if n not in built:
                        sreqs, left = [], []
                        if "ladder" in a.suite:
                            sreqs += ab_suites.ladder_reqs(n)
                        if "think" in a.suite:
                            sreqs += ab_suites.think_reqs(a.think_tokens, a.run_code)
                        if "agent" in a.suite:
                            agent, left = ab_suites.agent_reqs(n)
                            sreqs += agent
                        built[n] = sreqs, left
                    sreqs, arm["left_out"] = built[n]
                    print(f"round {rnd} v{v}: suites ready in {r.seconds:.0f} s, n_ctx {n_ctx}", flush=True)
                    post(a.port, "Say hi.", 8)
                    run_suite(a.port, sreqs, rnd, f"v{v}", out, got)
                except Exception as e:                               # noqa: BLE001 - a crash in the suites skips them, base rows stay
                    arm.update(suites_status="skipped", suites_reason=f"{type(e).__name__}: {e}")
                    suites_dead.add(v)
                    print(f"round {rnd} v{v}: suites skipped: {arm['suites_reason']}", flush=True)
                finally:
                    if got:
                        arm["rows"] += [{k: row[k] for k in ROW_KEYS if k in row} for row in got]
                        if v not in suites_dead:
                            arm.update(suites_status="ran", suites_reason="")
                    av.stop(control, venv)
    except KeyboardInterrupt:
        print("\ninterrupted; the results so far are kept", file=sys.stderr)
        code = 130
    finally:
        out.close()
        diff = av.changed(before, av.snapshot(roots))
        results = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model.name, "gpu": gpu_name(), "tune": a.tune,
                   "rounds": a.rounds, "requests": [t for t, _ in reqs], "arms": list(arms.values()), "untouched": not diff,
                   "changed": diff, "suites": a.suite, "context": context, "corpus": ab_suites.CORPUS_REV if a.suite else None,
                   "run_code": a.run_code}
        (run_dir / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        if diff:                                                     # in finally, before the report: shown even if either fails
            print("error: files of the real Strata install changed during the run:\n  " + "\n  ".join(diff), file=sys.stderr)
        html, png = ab_report.write_report(results, run_dir)
        print(f"\nresults: {run_dir / 'results.json'}\nreport:  {html}" + (f"\ncard:    {png}" if png else ""))
    return 1 if diff else code


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
    ap.add_argument("--bench-dir", help="versions mode: packages, homes, configs, runs (default: ~/strata-ab)")
    ap.add_argument("--backend", help="versions mode: engine build, e.g. cuda-cuda13 (default: this machine's)")
    ap.add_argument("--suite", help="versions mode: extra test suites, comma list of ladder, think, agent")
    ap.add_argument("--context", type=int, help="versions mode: context every version starts with for the suites "
                                                "(default: the model's maximum)")
    ap.add_argument("--think-tokens", type=int, help="versions mode: token budget of a thinking task (default 8192)")
    ap.add_argument("--run-code", action="store_true", help="versions mode: run and test the code the think suite's model "
                    "writes; it runs on this PC with your user rights and is not isolated.")
    a = ap.parse_intermixed_args(argv)
    if a.config is not None and (a.run_code or any(x is not None for x in (a.model_path, a.engines, a.tune, a.bench_dir, a.backend,
                                                                          a.suite, a.context, a.think_tokens))):
        ap.error("--model-path, --engines, --tune, --bench-dir, --backend, --suite, --context, --think-tokens and --run-code "
                 "belong to the versions mode, not to <config> <arms...>")
    a.suite = list(dict.fromkeys(x for x in (a.suite or "").split(",") if x))
    if any(x not in SUITES for x in a.suite):
        ap.error(f"--suite takes {', '.join(SUITES)}, not {a.suite!r}")
    if not a.suite and (a.run_code or a.context is not None or a.think_tokens is not None):
        ap.error("--context, --think-tokens and --run-code need --suite")
    if a.think_tokens is None:
        a.think_tokens = 8192
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
            Path(cfg["log"]).write_text("", encoding="utf-8")   # do not append to a previous A/B run (#4545)
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
