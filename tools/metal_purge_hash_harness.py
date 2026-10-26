#!/usr/bin/env python3
"""metal purge + state-hash harness (run while sushi and every other big model server is OFFLINE).

Proves the 2026-10-21 fix: the stream window gives back (purges) the CUDA-shaped weight/session device copies
>= 32 MB at startup; the state fingerprint (STRATA_STATE_HASH=1, run by tools/conversation_cache_*.py and
strata_batch_validation.py) used to READ them, tripping the runtime's check_purged abort and stopping a serving
engine on its first cached request (SIGABRT: "a copy touches a device allocation ... that was given back").
Writes now drop silently and the fingerprint skips given-back buffers, so the two settings coexist.

It starts its OWN server on --port (default 8181) with a small context and an 8 GB arena, runs two requests in
one conversation (the second gets prompt-cache reuse, which is when the fingerprint runs), and checks:
  1. the purge happened ("given back" in the engine log) - the precondition,
  2. the fingerprint ran on the cached request ("STATE_HASH" in the engine log),
  3. the engine never died (no "touches a device allocation", process alive, both replies returned),
  4. the second reply recalls the word.

Usage:  python3 tools/metal_purge_hash_harness.py [--config strata-iq3_xxs.json] [--port 8181]
Exit 0 = PASS, 1 = FAIL (the tails printed say why).
"""
import argparse, json, os, pathlib, subprocess, sys, time, urllib.request

PY = sys.executable
REPO = pathlib.Path(__file__).resolve().parent.parent


def post(url, body, timeout=240):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--config", default=str(REPO / "strata-iq3_xxs.json"))
    a.add_argument("--port", type=int, default=8181)
    args = a.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    cfg = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    try:
        with urllib.request.urlopen(base + "/health", timeout=3):
            sys.exit(f"[harness] something already answers {base} - free the port (it double-serves and OOMs)")
    except Exception:
        pass

    arg = cfg["args"]
    for k, v in (("--max-context", "8192"), ("--conversation-cache-min-tokens", "32")):
        if k in arg:
            arg[arg.index(k) + 1] = v
        else:
            arg += [k, v]
    env = dict(cfg.get("env") or {})
    env.pop("STRATA_METAL_KEEP_HOST_WEIGHTS", None)      # the crux: the startup purge must happen
    env["STRATA_METAL_ARENA_GB"] = "8"                   # fits beside whatever little is left
    env["STRATA_STATE_HASH"] = "1"
    out = pathlib.Path("/tmp/metal_purge_hash.config.json")
    cfg["env"] = env
    cfg["log"] = "/tmp/metal_purge_hash.engine.log"
    out.write_text(json.dumps(cfg, indent=1))
    model = cfg.get("model") or cfg.get("name") or "m"

    server_log = open("/tmp/metal_purge_hash.server.log", "w")
    p = subprocess.Popen([PY, "-m", "serve.server", "--engine", "strata", "--config", str(out),
                          "--host", "127.0.0.1", "--port", str(args.port)],
                         cwd=str(REPO), env={**os.environ, **env},
                         stdout=server_log, stderr=subprocess.STDOUT)
    fails = []
    try:
        for _ in range(84):                              # the load is the slow part (a minute or two)
            if p.poll() is not None:
                sys.exit(f"[harness] the server exited early (code {p.returncode}): see /tmp/metal_purge_hash.server.log")
            try:
                if json.loads(urllib.request.urlopen(base + "/health", timeout=5).read()).get("loaded"):
                    break
            except Exception:
                pass
            time.sleep(5)
        else:
            sys.exit("[harness] the model never loaded in 7 minutes")

        msgs = [{"role": "user", "content": "Remember this secret word: PLUTONIUM. It matters for later. "
                                            "Also note: the workshop meets Tuesdays and the budget code is FR-8821. Reply just OK."}]
        url = base + "/v1/chat/completions"
        c2 = ""
        try:
            r1 = post(url, {"model": model, "messages": msgs, "max_tokens": 48})
            c1 = r1["choices"][0]["message"]["content"]
            print(f"[harness] request 1 reply {c1[:40]!r}")
            msgs += [{"role": "assistant", "content": c1},
                     {"role": "user", "content": "Which word did I ask you to remember? Reply with the word only."}]
            r2 = post(url, {"model": model, "messages": msgs, "max_tokens": 48})
            c2 = r2["choices"][0]["message"]["content"]
            print(f"[harness] request 2 reply: {c2[:60]!r}")
        except Exception as e:                       # the dead engine answers 503 or drops the connection
            fails.append(f"a request failed ({e}): the engine probably died - see the log tails below")
        if p.poll() is not None:
            fails.append(f"the engine/server died during the requests (exit {p.returncode})")

        time.sleep(2)                                    # let the logs flush
        eng = pathlib.Path(cfg["log"]).read_text(encoding="utf-8", errors="replace") if pathlib.Path(cfg["log"]).exists() else ""
        srv = pathlib.Path("/tmp/metal_purge_hash.server.log").read_text(encoding="utf-8", errors="replace")
        if "given back" not in eng + srv:
            fails.append("precondition: the startup purge did not happen (no 'given back' line)")
        if "STATE_HASH" not in eng + srv:
            fails.append("the fingerprint never ran (no 'STATE_HASH' line): the cached request must be a hit")
        if "touches a device allocation" in eng + srv:
            fails.append("THE BUG IS BACK: a copy touched a given-back allocation (check_purged abort)")
        if "PLUTONIUM".lower() not in c2.lower():
            fails.append(f"request 2 did not recall the word: {c2!r}")
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
        server_log.close()

    if fails:
        print("[harness] FAIL:")
        for f in fails:
            print("  -", f)
        print("--- engine log tail ---")
        print("\n".join(eng.splitlines()[-25:]))
        print("--- server log tail ---")
        print("\n".join(srv.splitlines()[-8:]))
        return 1
    print("[harness] PASS: purge ran, the fingerprint skipped the given-back buffers, the engine served both turns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
