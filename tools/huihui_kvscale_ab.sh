#!/bin/bash
# Does the KV cache's format cost this Mac any decode speed? The Metal engine
# always stores K/V int8 (qsa.metal); fp16 storage would be an implementation
# project costing ~2 GB more RAM. This harness puts a BOUND on what fp16 could
# win without implementing it: QSA attention reads a fixed ~2048 cells per
# token whatever the context, so if decode tok/s stays FLAT from 1K to 48K
# context, the KV side is a small slice of the token time and fp16 cannot move
# it much. Second arm: STRATA_KV_ROT=1, the quality knob (Hadamard-rotated
# int8, closer to fp16 quality) - its speed cost and output changes measured.
#
#   tools/huihui_kvscale_ab.sh        # ~30-40 minutes, fully unattended
#
# Refuses to start while ANY strata engine is live (this box cannot hold two).
# Runs its own engines on 8181 from /tmp configs; the repo config is untouched.
# ALWAYS restarts the plain strata server on 8180 at the end
# (run-ud_q4_k_xl-huihui.sh), including on Ctrl-C. Results: stdout +
# /tmp/huihui-kvscale-ab.csv + quality texts /tmp/huihui-kvscale-*.txt
set -u
REPO="/Users/pc/src/Strata-fork"
PY="$REPO/.venv/bin/python"
CFG="$REPO/strata-ud-q4_k_xl-huihui.json"
LOG="$REPO/strata-ud-q4_k_xl-huihui.log"
CSV="/tmp/huihui-kvscale-ab.csv"
PIDF="/tmp/strata-huihui-kvscale.pid"
API="http://127.0.0.1:8181"

say() { printf '%s\n' "$*"; }
live8180() { curl -sf -m 3 http://127.0.0.1:8180/v1/models >/dev/null 2>&1; }
live8181() { curl -sf -m 3 "$API/v1/models" >/dev/null 2>&1; }
engine_pid() { pgrep -f "build-metal/strata" | head -1; }
rss_mib() { local p; p=$(engine_pid); [ -n "$p" ] && ps -o rss= -p "$p" | awk '{printf "%.0f", $1/1024}'; }

hui_stop() {
  [ -f "$PIDF" ] && { kill -TERM "$(cat "$PIDF")" 2>/dev/null; }
  for _ in $(seq 1 30); do
    sleep 2
    live8181 && continue
    [ -n "$(engine_pid)" ] && continue
    break
  done
  sleep 6   # let the wired arena unmap before the next engine maps it
}

restore_strata() {
  hui_stop
  if ! live8180 && ! live8181 && [ -z "$(engine_pid)" ]; then
    say "restarting the strata server on 8180 ..."
    nohup "$REPO/run-ud_q4_k_xl-huihui.sh" >> "$REPO/strata-ud-q4_k_xl-huihui.log" 2>&1 &
    for _ in $(seq 1 120); do
      sleep 4
      live8180 && { say "8180 is live again."; return 0; }
    done
    say "WARNING: 8180 did not come back within 480 s - start it with run-ud_q4_k_xl-huihui.sh"
  fi
}
cleanup() { restore_strata; }
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

probe() {  # $1 = arm tag; $2 = which set ("scale"|"rot")
  "$PY" - "$1" "$2" <<'PYEOF' | tee -a "$CSV"
import json, sys, time, urllib.request
from hashlib import sha256
tag, which = sys.argv[1], sys.argv[2]
API = "http://127.0.0.1:8181"
salt = sha256((tag + str(time.time())).encode()).hexdigest()[:8]
S = ("At 16:40 the arena hit rate fell from 0.86 to 0.41 after another process started reading a 90 GB "
     "file; drive read latency rose from 0.9 ms to 3.1 ms per 2 MB expert; GPU utilization stayed flat; "
     "conversations longer than 32k tokens began re-reading their whole prompt every turn. ")

def prompt(tok_target):
    reps = max(1, round(tok_target / (55 * 1.35)))   # ~55 words, ~1.35 tok/word
    return f"Case {salt}. " + S * reps + " Now answer question 1 only, in one paragraph."

def post(messages, mx):
    req = urllib.request.Request(API + "/v1/chat/completions",
        data=json.dumps({"messages": messages, "temperature": 0, "max_tokens": mx, "stream": False}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=3600) as r:
        body = json.loads(r.read().decode())
    return time.monotonic() - t0, body

def timings():
    try:
        with urllib.request.urlopen(API + "/v1/status", timeout=10) as r:
            return (json.loads(r.read().decode()) or {}).get("last_timings") or {}
    except Exception:
        return {}

def rss():
    import subprocess
    p = subprocess.run(["pgrep", "-f", "build-metal/strata"], capture_output=True, text=True)
    pid = p.stdout.split("\n")[0].strip() if p.stdout.strip() else ""
    if not pid:
        return "?"
    r = subprocess.run(["ps", "-o", "rss=", "-p", pid], capture_output=True, text=True)
    try:
        return str(int(int(r.stdout.strip()) / 1024))
    except Exception:
        return "?"

def scale(name, tok, gen=192):
    w, _ = post([{"role": "user", "content": prompt(tok)}], gen)
    t = timings()
    print(f"{tag},{name},{t.get('prompt_n','?')},{t.get('prompt_per_second','?')},{t.get('predicted_n','?')},{t.get('predicted_per_second','?')},{w:.1f},{rss()}", flush=True)

QUAL = [
    ("q1", "Explain in two sentences why a centrifugal pump cavitates, and name one remedy."),
    ("q2", "Write a Python function running_total(xs) that returns the cumulative sums using itertools.accumulate, with one example."),
    ("q3", "At 14:02:11 the arena eviction rate hit 0.43 while drive latency was 3.1 ms per 2 MB expert. List the three values as key: value lines."),
]

def quality():
    for name, text in QUAL:
        w, body = post([{"role": "user", "content": text}], 96)
        out = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
        with open(f"/tmp/huihui-kvscale-{tag}-{name}.txt", "w") as f:
            f.write(out)
        print(f"{tag},{name},-,-,-,-,{w:.1f},{rss()}", flush=True)

try:
    if which == "scale":
        scale("ctx1k", 1000)
        scale("ctx16k", 16000)
        scale("ctx48k", 48000)
        quality()
    else:
        scale("ctx16k", 16000)
        scale("ctx48k", 48000)
        quality()
except Exception as e:
    print(f"{tag},FAILED,-,-,-,-,-,{e}", flush=True)
PYEOF
}

# ---- refusal guard: two engines never co-reside on this box
if live8180 || live8181 || [ -n "$(engine_pid)" ]; then
  say "a strata engine is still live (8180/8181) - this box cannot hold two. Stop it first (Ctrl-C the run script, or tools/huihui_session.sh stop), then run this again."
  trap - EXIT
  exit 1
fi

say "start: $(memory_pressure -Q 2>/dev/null | grep 'System-wide memory free' || echo 'memory_pressure unavailable'), swap $(sysctl -n vm.swapusage)"

: > "$CSV"
for entry in "int8 0" "rot1 1"; do
  tag="${entry%% *}"; rot="${entry##* }"
  say "=== arm $tag (STRATA_KV_ROT=$rot) ==="
  "$PY" - "$CFG" "$tag" "$rot" <<'PYEOF'
import json, sys
cfg, tag, rot = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(cfg))
d["env"]["STRATA_KV_ROT"] = rot
d["port"] = 8181
open(f"/tmp/huihui-kvscale-{tag}.json", "w").write(json.dumps(d, indent=1))
PYEOF
  nohup "$PY" "$REPO/serve/server.py" --engine strata --config "/tmp/huihui-kvscale-$tag.json" --port 8181 >> "$LOG" 2>&1 &
  echo $! > "$PIDF"
  for _ in $(seq 1 120); do
    sleep 4
    live8181 && break
    kill -0 "$(cat "$PIDF")" 2>/dev/null || { say "engine died at arm=$tag; see $LOG"; tail -8 "$LOG"; break 2; }
  done
  live8181 || { say "engine not ready at arm=$tag"; tail -8 "$LOG"; break; }
  say "engine up (pid $(engine_pid), rss $(rss_mib) MiB); $(memory_pressure -Q 2>/dev/null | grep 'System-wide memory free' || true)"
  if [ "$tag" = "int8" ]; then probe "$tag" scale; else probe "$tag" rot; fi
  say "  arm end: rss $(rss_mib) MiB, swap $(sysctl -n vm.swapusage)"
  hui_stop
done

say "=== summary ==="
"$PY" - "$CSV" <<'PYEOF'
import csv, sys, os

rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) == 8]
fails = [r for r in rows if r[1] == "FAILED"]
d = {(r[0], r[1]): r for r in rows if r[1] != "FAILED"}

print(f"{'arm':5} {'probe':10} {'prompt_n':>9} {'prompt t/s':>11} {'gen_n':>6} {'decode t/s':>11} {'wall_s':>7} {'rss MiB':>8}")
for r in rows:
    if r[1] != "FAILED":
        print(f"{r[0]:5} {r[1]:10} {r[2]:>9} {r[3]:>11} {r[4]:>6} {r[5]:>11} {r[6]:>7} {r[7]:>8}")

def f(arm, probe, col):
    r = d.get((arm, probe))
    try:
        return float(r[col])
    except (TypeError, ValueError):
        return None

print()
if fails:
    print("FAILED rows:", "; ".join(r[7] for r in fails))

got_scale = all(("int8", p) in d for p in ("ctx1k", "ctx16k", "ctx48k"))
if got_scale:
    a, b = f("int8", "ctx1k", 5), f("int8", "ctx48k", 5)
    if a and b:
        ratio = b / a
        print(f"decode flatness 48K/1K: {b:.1f}/{a:.1f} = {ratio:.3f}")
        if ratio >= 0.95:
            print("VERDICT: decode is FLAT vs context - KV reads + dequant are under ~5% of the token time,")
            print("  so fp16 K/V storage (a +2 GB project) could not move decode more than a couple percent here.")
            print("  Recommendation: keep int8; the quality lever is STRATA_KV_ROT, not fp16.")
        elif ratio >= 0.85:
            print(f"VERDICT: decode falls {100 * (1 - ratio):.0f}% at 48K context. Part is indexer scoring (fp16 does NOT fix that),")
            print("  part is the KV side. Implement fp16 only if the drop is bigger at 96K+ and answers must stay exact.")
        else:
            print(f"VERDICT: decode falls {100 * (1 - ratio):.0f}% at 48K - a long-context cost worth fixing;")
            print("  implement fp16 K/V storage (bounded upside = this drop, less the indexer share).")
    r1, r48 = f("int8", "ctx1k", 7), f("int8", "ctx48k", 7)
    if r1 and r48:
        print(f"engine RSS 1K -> 48K: {r1:.0f} -> {r48:.0f} MiB (delta {r48 - r1:.0f} MiB; fp16 KV would add ~2048 B/cell x QSA layers on top)")

if ("rot1", "ctx16k") in d and ("int8", "ctx16k") in d:
    x0, x1 = f("int8", "ctx16k", 5), f("rot1", "ctx16k", 5)
    p0, p1 = f("int8", "ctx16k", 3), f("rot1", "ctx16k", 3)
    if x0 and x1:
        print(f"STRATA_KV_ROT=1 cost at 16K: decode {x0:.1f} -> {x1:.1f} t/s ({100 * (x1 / x0 - 1):+.1f}%), prompt {p0} -> {p1} t/s")

ndiff = 0
for q in ("q1", "q2", "q3"):
    p0, p1 = f"/tmp/huihui-kvscale-int8-{q}.txt", f"/tmp/huihui-kvscale-rot1-{q}.txt"
    if os.path.exists(p0) and os.path.exists(p1):
        same = open(p0).read().strip() == open(p1).read().strip()
        ndiff += not same
        print(f"quality probe {q}: rot1 output {'identical to' if same else 'DIFFERS from'} int8")
if ndiff:
    print(f"(rot1 changed {ndiff}/3 greedy answers - the closer-to-fp16 quality knob has a real effect; see the /tmp/huihui-kvscale-*-q*.txt files)")
print()
print("Reference from earlier runs on this box: reused-turn decode ~18.4 t/s (chunk 2048, arena 33);")
print("decode bandwidth ceiling ~20-21 t/s; expert reads ~1.18 GB/token vs KV int8 ~26 MB/token at topk 2048.")
PYEOF

restore_strata
say "done. CSV: $CSV"
