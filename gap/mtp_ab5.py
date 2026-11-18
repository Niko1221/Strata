#!/usr/bin/env python3
"""Round 5 (regime test): decode at ~100K-token context, the length real sessions run at.

Each arm: restart host, PING, warm (cold 95K prefill fills the conversation cache, timing ignored),
then the timed repeat of the same prompt (prefix reused -> decode dominates, parking exercised).
Arms: N0 (no MTP) and S2 (MTP spec 2), alternating, three passes each.
Final: original config restored, host up, slot restored.
"""
import json, os, re, shutil, subprocess, time, urllib.request

FORK = "/Users/pc/src/Strata-fork"
CFG = f"{FORK}/strata-ud-q4_k_xl-huihui.json"
RUN = f"{FORK}/run-ud_q4_k_xl-huihui.sh"
LOG = f"{FORK}/strata-ud_q4_k_xl-huihui.log"
SESS = f"{FORK}/tools/huihui_session.sh"
MTP = "/Users/pc/src/Strata-data/mtp/rt"
OUT = os.environ["OUT"]
SUM = f"{OUT}/summary.txt"
SNAP = f"{OUT}/config.pristine.json"

PROMPT_RE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in [\d.]+ ms \(([\d.]+) tok/s\), "
                       r"(\d+) generated in ([\d.]+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
FAIL_RE = ("unknown option", "needs --native", "not supported", "exiting", "cannot open", "no such file")

ARMS = {
    "N0": {"add": []},
    "S2": {"add": ["--mtp", MTP, "--spec", "2"]},
}
TEXT = None  # built at start


def log(msg):
    print(msg, flush=True)
    with open(SUM, "a") as f:
        f.write(msg + "\n")


def sh(*args):
    return subprocess.run(args, capture_output=True, text=True)


def stop_host():
    sh("pkill", "-f", "serve/server.py.*ud-q4_k_xl-huihui")
    time.sleep(1)
    sh("pkill", "-9", "-f", "build-metal/strata.*ud-q4_k_xl-huihui")
    for _ in range(20):
        if not sh("pgrep", "-f", "ud-q4_k_xl-huihui").stdout.strip():
            break
        time.sleep(1)
    time.sleep(2)


def patch(arm):
    d = json.load(open(SNAP))
    a = [x for x in d["args"]]
    for flag in ("--mtp", "--spec"):
        while flag in a:
            i = a.index(flag)
            del a[i:i + 2]
    k = a.index("--max-context")
    a[k:k] = ARMS[arm]["add"]
    d["args"] = a
    json.dump(d, open(CFG, "w"), indent=4)
    open(CFG, "a").write("\n")


def post(max_tokens, timeout=2400):
    b = {"model": "huihui", "messages": [{"role": "user", "content": TEXT}],
         "temperature": 0.0, "max_tokens": max_tokens, "thinking_budget": 0, "user": "regime"}
    r = urllib.request.Request("http://127.0.0.1:8180/v1/chat/completions",
                               json.dumps(b).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    j = json.load(urllib.request.urlopen(r, timeout=timeout))
    return j, time.time() - t0


def measure(off, arm):
    time.sleep(1)
    body = open(LOG).read() if os.path.getsize(LOG) - off < 40 * 1024 * 1024 else ""
    m = None
    for m in PROMPT_RE.finditer(open(LOG).read()[off:]) if body else []:
        pass
    if not m:
        log(f"{arm}: NO PROMPT LINE")
        return None
    g = m.groups()
    ptoks, gen, gms, gtok = (float(g[3]), int(g[4]), float(g[5]), float(g[6]))
    da, db = int(g[7]), int(g[8])
    log(f"{arm}: ctx {g[0]} reused {g[1]} prefill {ptoks} tok/s | decode {gen} tok {gms/1000:.1f}s "
        f"{gtok} tok/s | drafts {da}/{db}")
    return gtok


def started(off):
    for _ in range(140):
        try:
            u = json.load(urllib.request.urlopen("http://127.0.0.1:8180/v1/models", timeout=3))
            if any("huihui" in m.get("id", "") for m in u.get("data", [])):
                return True, ""
        except Exception:
            pass
        for l in open(LOG).read()[off:].splitlines():
            for pat in FAIL_RE:
                if pat in l.lower():
                    return False, l.strip()[:160]
        time.sleep(3)
    return False, "never listed the model (7 min)"


def main():
    global TEXT
    shutil.copy(CFG, SNAP)
    orig = json.load(open(SNAP))
    parts = []
    for i in range(3800):
        parts.append(f"The lighthouse at river mile {i % 97} logged {i * 7 % 401} boats, "
                     f"{i * 13 % 311} gulls and {(i * 3 + 5) % 59} sealed crates this week. ")
    TEXT = "".join(parts) + "\nNow write 500 short sentences about rivers and bridges, one per line."
    sh("bash", SESS, "save", "pre-mtp-ab3")
    log(f"round-5 regime start {time.strftime('%T')}; prompt ~{(len(TEXT) + 5) // 4} est tokens")
    results = {a: [] for a in ARMS}
    dead = set()
    order = list(ARMS)
    try:
        for p in (1, 2, 3):
            seq = order if p % 2 else list(reversed(order))
            for arm in seq:
                if arm in dead:
                    continue
                stop_host()
                patch(arm)
                off = os.path.getsize(LOG)
                with open(f"{OUT}/{arm}.serve.log", "a") as lf:
                    subprocess.Popen(["bash", RUN], stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
                ok, why = started(off)
                if not ok:
                    log(f"{arm}: START FAILED: {why}")
                    dead.add(arm)
                    stop_host()
                    continue
                try:
                    post(4, timeout=1800)  # PING-style warm (long prefill path primes)
                except Exception as e:
                    log(f"{arm}: PING failed: {e}")
                    dead.add(arm)
                    stop_host()
                    continue
                try:
                    post(16, timeout=2400)  # warm the conversation cache at full length
                except Exception as e:
                    log(f"{arm}: warm failed: {e}")
                off2 = os.path.getsize(LOG)
                try:
                    post(500, timeout=2400)  # the timed repeat
                except Exception as e:
                    log(f"{arm}: timed failed: {e}")
                    continue
                r = measure(off2, arm)
                if r is not None:
                    results[arm].append(r)
            log(f"pass {p} done {time.strftime('%T')}")
    finally:
        json.dump(orig, open(CFG, "w"), indent=4)
        open(CFG, "a").write("\n")
        stop_host()
        with open(f"{OUT}/final.serve.log", "a") as lf:
            subprocess.Popen(["bash", RUN], stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        ok, why = started(os.path.getsize(LOG))
        sh("bash", SESS, "restore", "pre-mtp-ab3")
        log(f"final: config restored, host {'UP' if ok else 'DOWN ' + why}, slot restored {time.strftime('%T')}")
    log("== decode tok/s at ~100K context ==")
    for arm, rs in results.items():
        log(f"{arm}: {', '.join(str(x) for x in rs)}" + (f" | avg {sum(rs)/len(rs):.1f}" if rs else " | no data"))


main()
