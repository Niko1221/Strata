#!/usr/bin/env python3
"""Round 3 (goal: fastest decode for huihui): five serve arms on the metal-refresh exe, decode-dominated
greedy probe, two interleaved passes.

  N0   no MTP:            no --mtp, no --spec  (suffix drafts only, the engine's own speculation)
  S2   current default:   --mtp rt --spec 2
  S3   deeper:            --mtp rt --spec 3
  Q8   Q8_0 MTP drafter:  --mtp rt --spec 2 + STRATA_METAL_MTP_GGUF=/Users/pc/models/mtp-...Q8_0.gguf
  DF2  DFlash2 pack:      --mtp rt --spec 2 --draft dflash2 --draft-model flash-next-df2.pack

Every arm: fresh config from the pristine snapshot, host restart, PING (engine load), one ~6K-token
prompt asking for 300 greedy sentences, parse the serve line (prefill/decode tok/s, drafts accepted).
The first DF2 boot tells whether the Metal engine takes the CUDA/HIP drafter at all.
Final state: original config restored, host started, slot pre-mtp-ab3 restored.
"""
import copy, json, os, re, shutil, subprocess, sys, time, urllib.request

FORK = "/Users/pc/src/Strata-fork"
CFG = f"{FORK}/strata-ud-q4_k_xl-huihui.json"
RUN = f"{FORK}/run-ud_q4_k_xl-huihui.sh"
LOG = f"{FORK}/strata-ud_q4_k_xl-huihui.log"
SESS = f"{FORK}/tools/huihui_session.sh"
MTP = "/Users/pc/src/Strata-data/mtp/rt"
DF2PACK = "/Users/pc/src/Strata-data/dflash2/flash-next-df2.pack"
Q8MTP = "/Users/pc/models/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
OUT = os.environ["OUT"]
SUM = f"{OUT}/summary.txt"
SNAP = f"{OUT}/config.pristine.json"

PROMPT_RE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in [\d.]+ ms \(([\d.]+) tok/s\), "
                       r"(\d+) generated in ([\d.]+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
FAIL_RE = ("unknown option", "needs CUDA", "requires CUDA", "not supported", "exiting", "cannot open", "no such file")

ARMS = {
    "N0": {"del": [], "add": [], "env": {}},
    "S1": {"del": [], "add": ["--mtp", MTP, "--spec", "1"], "env": {}},
    "S2":  {"del": [], "add": ["--mtp", MTP, "--spec", "2"], "env": {}},



}

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
        r = sh("pgrep", "-f", "ud-q4_k_xl-huihui")
        if not r.stdout.strip():
            break
        time.sleep(1)
    time.sleep(2)

def patch(arm):
    d = json.load(open(SNAP))
    spec = ARMS[arm]
    a = list(d["args"])
    for flag in ("--mtp", "--spec", "--draft", "--draft-model"):
        while flag in a:
            i = a.index(flag)
            del a[i:i + 2]
    k = a.index("--max-context")
    a[k:k] = spec["add"]
    d["args"] = a
    env = dict(d.get("env", {}))
    env.update(spec["env"])
    d["env"] = env
    json.dump(d, open(CFG, "w"), indent=4)
    open(CFG, "a").write("\n")

def post(msgs, max_tokens, name="x", temperature=0.0, timeout=1800):
    b = {"model": "huihui", "messages": msgs, "temperature": temperature,
         "max_tokens": max_tokens, "thinking_budget": 0, "user": name}
    r = urllib.request.Request("http://127.0.0.1:8180/v1/chat/completions",
                               json.dumps(b).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    j = json.load(urllib.request.urlopen(r, timeout=timeout))
    return j, time.time() - t0

def tail_since(off):
    with open(LOG) as f:
        f.seek(off)
        return f.read()

def started(off):
    """True if the host lists the model; False if the serve log shows a startup rejection."""
    for _ in range(140):
        try:
            u = json.load(urllib.request.urlopen("http://127.0.0.1:8180/v1/models", timeout=3))
            if any("huihui" in m.get("id", "") for m in u.get("data", [])):
                return True, ""
        except Exception:
            pass
        body = tail_since(off)
        for pat in FAIL_RE:
            for l in body.splitlines():
                if pat in l.lower() or pat in l:
                    return False, l.strip()[:160]
        time.sleep(3)
    return False, "never listed the model (7 min)"

def filler(n, nonce):
    parts = [f"Probe nonce {nonce}. "]
    for i in range(n):
        parts.append(f"The lighthouse at river mile {i % 97} logged {i * 7 % 401} boats, "
                     f"{i * 13 % 311} gulls and {(i * 3 + 5) % 59} sealed crates this week. ")
    return "".join(parts)

def probe(arm, off):
    text = filler(240, f"{arm}-{int(time.time())}") + "\nNow write 500 short sentences about rivers and bridges, one per line."
    j, wall = post([{"role": "user", "content": text}], 500, name=arm)
    time.sleep(1)
    body = tail_since(off)
    m = None
    for m in PROMPT_RE.finditer(body):
        pass
    if not m:
        log(f"{arm}: NO PROMPT LINE (wall {wall:.0f}s)")
        return None
    pt, reused, read_, ptoks, gen, gms, gtok, da, db = m.groups()
    pt, reused, gen, da, db = (int(x) for x in (pt, reused, gen, da, db))
    ptoks, gms, gtok = (float(x) for x in (ptoks, gms, gtok))
    log(f"{arm}: prefill {ptoks} tok/s | decode {gen} tok {gms/1000:.1f}s {gtok} tok/s | "
        f"drafts {da}/{db} ({100.0*da/max(db,1):.0f}%)")
    return gtok

def main():
    shutil.copy(CFG, SNAP)
    orig = json.load(open(SNAP))
    sh("bash", SESS, "save", "pre-mtp-ab3")
    log(f"round-3 start {time.strftime('%T')}; slot pre-mtp-ab3 saved")
    order = list(ARMS)
    results = {a: [] for a in ARMS}
    dead = set()
    try:
        for p in (1, 2, 3):
            seq = order if p == 1 else list(reversed(order))
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
                    post([{"role": "user", "content": "Reply with just the word PING."}], 4, timeout=1800)
                except Exception as e:
                    log(f"{arm}: PING failed: {e}")
                    dead.add(arm)
                    stop_host()
                    continue
                off2 = os.path.getsize(LOG)
                r = probe(arm, off2)
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
    log("== decode tok/s by arm ==")
    for arm, rs in results.items():
        if rs:
            best = max(rs)
            log(f"{arm}: {', '.join(str(x) for x in rs)} | best {best}")
        else:
            log(f"{arm}: no data")

main()
