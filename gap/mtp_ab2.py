#!/usr/bin/env python3
"""Round 2, one host-stop window:
 A) serve decode A/B, same ~25K-token greedy probe, MTP off/on interleaved (metal-live engine);
 B) park sequence (long conv A, two side chats, follow-up on A) on metal-live (MTP) and on metal-refresh;
 C) metal-refresh serve probe + CLI decode ladder (no-MTP / MTP spec2 / DFlash2 smoke) vs the morning arms;
 D) park under MTP expected SKIP on metal-live, WORK on metal-refresh (PR #3541).
OUT env set by the caller. Final state: config mtp-on + metal-live exe, host started, slot pre-mtp-ab2 restored.
"""
import json, os, re, subprocess, sys, time, urllib.request

FORK = "/Users/pc/src/Strata-fork"
CFG = f"{FORK}/strata-ud-q4_k_xl-huihui.json"
RUN = f"{FORK}/run-ud_q4_k_xl-huihui.sh"
LOG = f"{FORK}/strata-ud_q4_k_xl-huihui.log"
MTP = "/Users/pc/src/Strata-data/mtp/rt"
DF2 = "/Users/pc/src/Strata-data/dflash2/flash-next-df2.pack"
PACK = "/Users/pc/src/Strata-data/packs/ud-q4_k_xl-huihui"
GGUF = "/Users/pc/src/Strata-data/metal/ud-q4_k_xl-huihui-metal.gguf"
PLE = "/Users/pc/models/Huihui-Qwen3.8-Flash-Next-abliterated-v2-UD-Q4_K_XL-V2/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf"
LIVE = f"{FORK}/build-metal/strata"
REF = "/private/tmp/wtR/build-metal/strata"
OUT = os.environ["OUT"]
SUM = f"{OUT}/summary.txt"

PROMPT_RE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in [\d.]+ ms \(([\d.]+) tok/s\), "
                       r"(\d+) generated in ([\d.]+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")

def log(msg):
    print(msg, flush=True)
    with open(SUM, "a") as f:
        f.write(msg + "\n")

def sh(*args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)

def stop_host():
    sh("pkill", "-f", "serve/server.py.*ud-q4_k_xl-huihui")
    time.sleep(1)
    sh("pkill", "-9", "-f", "build-metal/strata --serve.*ud-q4_k_xl-huihui")
    sh("pkill", "-9", "-f", "wtR/build-metal/strata --serve")
    time.sleep(3)

def patch(exe=None, mtp=True):
    d = json.load(open(CFG))
    a = d["args"]
    while "--mtp" in a:
        i = a.index("--mtp"); del a[i:i+2]
    if mtp:
        k = a.index("--max-context")
        a[k:k] = ["--mtp", MTP]
    if exe:
        d["exe"] = exe
    json.dump(d, open(CFG, "w"), indent=4)
    open(CFG, "a").write("\n")

def started_marker():
    return os.path.getsize(LOG)

def start_host(arm):
    with open(f"{OUT}/{arm}.serve.log", "a") as lf:
        subprocess.Popen(["bash", RUN], stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(120):
        try:
            u = json.load(urllib.request.urlopen("http://127.0.0.1:8180/v1/models", timeout=3))
            if any("huihui" in m.get("id", "") for m in u.get("data", [])):
                break
        except Exception:
            pass
    else:
        raise RuntimeError(f"{arm}: host never listed the model")
    post([{"role": "user", "content": "Reply with just the word PING."}], 4, timeout=900)  # engine load

def post(msgs, max_tokens, name="x", temperature=0.0, timeout=1800):
    b = {"model": "huihui", "messages": msgs, "temperature": temperature,
         "max_tokens": max_tokens, "thinking_budget": 0, "user": name}
    t0 = time.time()
    r = urllib.request.Request("http://127.0.0.1:8180/v1/chat/completions",
                               json.dumps(b).encode(), {"Content-Type": "application/json"})
    j = json.load(urllib.request.urlopen(r, timeout=timeout))
    return j, time.time() - t0

def tail_since(off):
    with open(LOG) as f:
        f.seek(off)
        return f.read()

def vmstat():
    o = sh("vm_stat").stdout
    free = re.search(r"Pages free:\s+(\d+)", o)
    fc = sh("sysctl", "-n", "vm.vm_page_filecache_min").stdout.strip()
    return f"free_pages {free.group(1) if free else '?'} fc_min {fc}"

def filler(n, nonce):
    parts = [f"Probe nonce {nonce}. "]
    for i in range(n):
        parts.append(f"The lighthouse at river mile {i % 97} logged {i * 7 % 401} boats, "
                     f"{i * 13 % 311} gulls and {(i * 3 + 5) % 59} sealed crates this week. ")
    return "".join(parts)

def probe25k(arm, off):
    text = filler(950, f"{arm}-{int(time.time())}") + "\nNow write 250 short sentences about rivers and bridges, one per line."
    j, wall = post([{"role": "user", "content": text}], 250, name=arm)
    time.sleep(1)
    body = tail_since(off)
    m = None
    for m in PROMPT_RE.finditer(body):
        pass
    if not m:
        log(f"{arm}: NO PROMPT LINE (wall {wall:.0f}s)")
        return
    pt, reused, read_, ptoks, gen, gms, gtok, da, db = (int(x) for x in m.groups())
    log(f"{arm}: prompt {pt} ({reused} reused) prefill {ptoks} tok/s | decode {gen} tok {gms/1000:.1f}s "
        f"{gtok} tok/s | drafts {da}/{db} ({100.0*da/max(db,1):.0f}%) | {vmstat()}")

def park_seq(arm, off):
    q = filler(240, f"{arm}-pk-{int(time.time())}") + "\nReply with just the word ALPHA."
    msgs = [{"role": "user", "content": q}]
    j, _ = post(msgs, 8, name=f"{arm}-pk")
    a1 = j["choices"][0]["message"]["content"]
    for s in ("sideone", "sidetwo"):
        post([{"role": "user", "content": f"Side chat {s}: name three tools. Two words each, six words total."}], 12, name=s)
    msgs += [{"role": "assistant", "content": a1},
             {"role": "user", "content": "Now write 160 short sentences about bridges and toll booths, one per line."}]
    post(msgs, 160, name=f"{arm}-pk")
    time.sleep(1)
    body = tail_since(off)
    ms = list(PROMPT_RE.finditer(body))
    if len(ms) >= 2:
        m = ms[-1]
        pt, reused, read_, ptoks, gen, gms, gtok, da, db = (int(x) for x in m.groups())
        log(f"{arm}-park: follow-up prompt {pt} reused {reused} ({100.0*reused/max(pt,1):.0f}%) "
            f"prefill {ptoks} tok/s decode {gtok} tok/s")
    ev = [l for l in body.splitlines() if "conversation cache:" in l and ("parked" in l or "restored" in l or "skip" in l)]
    for l in ev[-4:]:
        log(f"{arm}-park: {l.split('conversation cache:')[1].strip()[:110]}")

def cli_arm(name, binp, extra):
    log(f"=== {name} {time.strftime('%T')}")
    env = dict(os.environ, STRATA_METAL_STREAM="1", STRATA_METAL_PROMPT_CHUNK="2048",
               STRATA_METAL_PROMPT_MOE="mpp", STRATA_METAL_GDN_SCAN="scan_v",
               STRATA_METAL_ARENA_GB="33", STRATA_METAL_LEAVE_SHARE="0.35")
    cmd = [binp, "--pack", PACK, "--native", GGUF, "--ple-gguf", PLE, "--native-dense-gguf", GGUF,
           "--expert-profile", f"{FORK}/data/expert-profile.bin", "--expert-cache", "64", "--prefill", "auto",
           "--suffix-draft", "1", "--max-context", "40960", "--mmap-experts",
           "--tokens-file", f"{FORK}/gap/tokens/t357.txt", "--max-new", "192"] + extra
    with open(f"{OUT}/{name}.log", "w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
    body = open(f"{OUT}/{name}.log").read()
    for pat in ("decode ", "draft", "dflash", "DFlash", "speculat", "KERR", "error", "unknown", "needs"):
        for l in body.splitlines():
            if pat in l:
                log(f"{name}: {l.strip()[:160]}")
                break
    log(f"{name}: rc {rc}")

def final_restore():
    patch(exe=LIVE, mtp=True)
    stop_host()
    r = subprocess.run([f"{FORK}/tools/huihui_session.sh", "start", "pre-mtp-ab2"],
                       capture_output=True, text=True, timeout=900)
    log("restore: " + (r.stdout.strip() or r.stderr.strip())[-300:])

def main():
    os.makedirs(OUT, exist_ok=True)
    stop_host()
    r = subprocess.run([f"{FORK}/tools/huihui_session.sh", "stop", "pre-mtp-ab2"],
                       capture_output=True, text=True, timeout=600)
    log("saved: " + r.stdout.strip()[-160:])
    try:
        for arm, mtp in (("X_off1", False), ("X_on1", True), ("X_off2", False), ("X_on2", True)):
            log(f"=== {arm} mtp={mtp} {time.strftime('%T')}")
            patch(exe=LIVE, mtp=mtp)
            stop_host()
            off = started_marker()
            start_host(arm)
            probe25k(arm, off)
        log(f"=== P_live (park seq, metal-live, mtp on) {time.strftime('%T')}")
        off = started_marker()
        park_seq("P_live", off)
        log(f"=== Z_ref (metal-refresh serve, mtp+park) {time.strftime('%T')}")
        patch(exe=REF, mtp=True)
        stop_host()
        off = started_marker()
        start_host("Z_ref")
        probe25k("Z_ref", off)
        park_seq("Z_ref", off)
        eng = open(f"{OUT}/Z_ref.serve.log").read() + tail_since(off)
        for pat in ("MTP draft layer", "dflash", "DFlash"):
            for l in eng.splitlines():
                if pat in l:
                    log(f"Z_ref: {l.strip()[:150]}")
                    break
        stop_host()
        cli_arm("R_nmtp", REF, ["--spec", "2"])
        cli_arm("R_mtp2", REF, ["--spec", "2", "--mtp", MTP])
        cli_arm("R_df2", REF, ["--spec", "2", "--mtp", MTP, "--draft", "dflash2", "--draft-model", DF2])
    finally:
        final_restore()
    log(f"done {time.strftime('%T')}")

main()
