#!/usr/bin/env python3
"""parking_check.py: conversation-parking A/B on one machine, one sitting.

Boots `serve/server.py` twice against temporary copies of strata-iq3_xxs.json - once
with the conversation cache OFF (the flags stripped: last release's behaviour) and once
ON (8192 MiB, min-tokens 2048) - and drives the same conversation through both: a
growing tool-call transcript with short side requests interleaved between every main
turn, which is the omp shape that re-prefilled 30k+ tokens a turn.

Per main turn it reads /metrics and records prompt_tokens, reused and prompt_ms.
Verdict: parking ON must reuse the prefix across a side turn, and the prefill on a reused
turn is under 45% of the same-size full read the OFF arm measured (the new tokens
alone are ~24-44% of the prompt, so cold turn 1 is excluded: both arms prefill it in full).

Stop the other model servers first (needs ~40 GB free or it waits).  Run detached:
    nohup python3 metal/tools/parking_check.py > /tmp/parking_harness.log 2>&1 &
Results land in gap/results/<stamp>/parking_check.md (raw JSON beside it).
"""
import json, os, random, re, signal, subprocess, sys, time, urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE_CFG = os.path.join(REPO, "strata-iq3_xxs.json")
PORT = 8180
BASE = f"http://127.0.0.1:{PORT}"
TURNS, SIDES, GROW = 3, 1, 2000
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november "
         "oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()

def filler(rng, ntok):
    return " ".join(rng.choice(WORDS) + str(rng.randrange(10 ** 6)) for _ in range(int(ntok * 0.132)))

def post(body, timeout=2400):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        json.loads(r.read())
    return time.time() - t0

def metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
        return json.load(r)

def free_bytes():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pages = {re.match(r"Pages (\w+)", ln).group(1): float(re.search(r"([\d.]+)\.", ln).group(1))
             for ln in out.splitlines() if ln.startswith("Pages ")}
    return int((pages.get("free", 0) + pages.get("inactive", 0) + pages.get("speculative", 0)) * 16384)

def wait_headroom(need=40 << 30, hours=2.0):
    t0 = time.time()
    while free_bytes() < need:
        if time.time() - t0 > hours * 3600:
            sys.exit(f"give up: still only {free_bytes() >> 30} GB reclaimable after {hours} h")
        print(f"waiting for RAM: {free_bytes() >> 30} GB free-ish, need {need >> 30}", flush=True)
        time.sleep(30)

def spawn_serve(cfg_path, tag):
    log = open(f"/tmp/parking_serve_{tag}.log", "w")
    p = subprocess.Popen([sys.executable, "-u", os.path.join(REPO, "serve", "server.py"),
                          "--engine", "strata", "--config", cfg_path, "--port", str(PORT)],
                         cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return p, log

def ready_wait(p, path, seconds=1200):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if p.poll() is not None:
            sys.exit(f"serve exited early; see {path}")
        try:
            if "ready: http" in open(path).read():
                return
        except OSError:
            pass
        time.sleep(5)
    sys.exit(f"serve never became ready; see {path}")

def warm():   # one tiny prompt, retried once: it pays the boot's page-ins
    for attempt in range(2):
        try:
            return post({"model": "m", "messages": [{"role": "user", "content": "warm"}],
                         "max_tokens": 8, "temperature": 0})
        except Exception as e:
            print(f"warm {attempt}: {e}", flush=True)
            time.sleep(10)
    raise RuntimeError("warm failed")

SERVE = None


def stop_serve(p):
    if p is None or p.poll() is not None:
        return
    p.terminate()
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)

def conversation(mib, tag):
    cfg = json.load(open(BASE_CFG))
    a = cfg["args"]
    drop = {"--conversation-cache-mib", "--conversation-cache-min-tokens"}
    keep, skip = [], False
    for x in a:   # rebuild the flag list for this arm from the user's own config
        if skip: skip = False; continue
        if x in drop: skip = True; continue
        keep.append(x)
    if mib:
        keep += ["--conversation-cache-mib", str(mib), "--conversation-cache-min-tokens", "2048"]
    cfg["args"], cfg["log"] = keep, f"/tmp/parking_engine_{tag}.log"
    cfg_path = f"/tmp/parking_cfg_{tag}.json"
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    p, log = spawn_serve(cfg_path, tag)
    global SERVE
    SERVE = p
    ready_wait(p, log.name)
    w = warm()
    print(f"[{tag}] ready, warm prompt took {w:.1f} s", flush=True)
    rng = random.Random(7)
    msgs = [{"role": "system", "content": "You are Strata, an assistant driving a codebase. "
                                          "Tool outputs are long; answer with exactly OK.\n" + filler(rng, 2500)}]
    rows = []
    for turn in range(1, TURNS + 1):
        for k in range(SIDES):   # an omp auxiliary call: independent, short, no shared prefix
            side = [{"role": "system", "content": "Summarize in five words. " + filler(rng, 600)},
                    {"role": "user", "content": f"side {turn}.{k}: " + filler(rng, 200)}]
            try:
                post({"model": "m", "messages": side, "max_tokens": 8, "temperature": 0})
            except Exception as e:
                print(f"[{tag}] side failed: {e}", flush=True)
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"call_{turn}", "type": "function",
                                     "function": {"name": "read", "arguments": json.dumps({"path": f"f{turn}.py"})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{turn}",
                     "content": f"file {turn}:\n" + filler(rng, GROW)})
        wall = post({"model": "m", "messages": msgs, "max_tokens": 8, "temperature": 0})
        d = metrics()
        r = d["requests"][0]   # /metrics lists finished requests newest first
        r["wall"] = wall
        rows.append(r)
        print(f"[{tag}] turn {turn}: prompt={r['prompt_tokens']:>6} reused={r['reused']:>6} "
              f"prompt_ms={r['prompt_ms']:>8.0f} wall={wall:6.1f}s parked={d['conversation_cache']['parked']} "
              f"restores={d['conversation_cache']['restores']}", flush=True)
        msgs.append({"role": "assistant", "content": "OK"})
    cc = metrics()["conversation_cache"]
    stop_serve(p)
    SERVE = None
    time.sleep(10)   # let the engine's pages go before the next arm
    return rows, cc, w

def main():
    wait_headroom()
    stamp = time.strftime("%m%d-%H%M")
    outdir = os.path.join(REPO, "gap", "results", stamp)
    os.makedirs(outdir, exist_ok=True)
    print(f"free-ish now: {free_bytes() >> 30} GB", flush=True)
    off, on = {}, {}
    try:
        for mib, tag in ((0, "off"), (8192, "on")):
            rows, cc, w = conversation(mib, tag)
            (off if mib == 0 else on)["rows"] = rows
            (off if mib == 0 else on)["cache"] = cc
            (off if mib == 0 else on)["warm_s"] = w
            json.dump({"rows": rows, "cache": cc, "warm_s": w}, open(f"{outdir}/arm_{tag}.json", "w"), indent=1)
    finally:
        stop_serve(SERVE)
    lines = ["# conversation parking A/B", "",
             f"engine config: {BASE_CFG}; {TURNS} main turns, {SIDES} side requests between each, "
             f"+{GROW:.0f} tokens/turn; parking ON = 8192 MiB, min-tokens 2048.", ""]
    for tag, arm in (("OFF", off), ("ON", on)):
        lines += [f"## parking {tag}", "", "| turn | prompt | reused | prompt_ms | wall s |", "|---|---|---|---|---|"]
        lines += [f"| {i + 1} | {r['prompt_tokens']} | {r['reused']} | {r['prompt_ms']:.0f} | {r['wall']:.1f} |"
                  for i, r in enumerate(arm["rows"])]
        c = arm["cache"]
        lines += ["", f"cache: parks={c.get('parks')} restores={c.get('restores')} evictions={c.get('evictions')} "
                      f"bytes={c.get('bytes')}", ""]
    ok_reuse = all(on["rows"][i]["reused"] >= 0.8 * on["rows"][i - 1]["prompt_tokens"]
                   for i in range(1, len(on["rows"])))
    pairs = [(r_on["prompt_ms"], r_off["prompt_ms"]) for r_on, r_off in zip(on["rows"], off["rows"])
             if r_on["reused"] > 0 and r_off["prompt_ms"] > 0]
    worst = max((a / b for a, b in pairs), default=None)
    lines += ["## verdict", "",
              f"- reuses the prefix after every side turn (ON, turns 2-{TURNS}): {'PASS' if ok_reuse else 'FAIL'}",
              f"- worst ON/OFF prefill ratio on reused turns: {'n/a' if worst is None else f'{worst:.1%}'}"
              f" (gate: 45%); {'PASS' if worst is not None and worst <= 0.45 else 'FAIL'}", ""]
    md = "\n".join(lines)
    open(f"{outdir}/parking_check.md", "w").write(md)
    print("\n" + md, flush=True)

if __name__ == "__main__":
    main()
