#!/bin/bash
# Arena A/B for the Huihui UD-Q4_K_XL engine: STRATA_METAL_ARENA_GB=33 +
# STRATA_METAL_LEAVE_SHARE=0.35 (the IQ3 server's proven values) at chunk
# 1024 / 2048 / 4096 + a --suffix-draft 1 round, all at --max-context 162144.
#
#   tools/huihui_arena_ab.sh
#
# - Stops the IQ3_XXS server on 8180 if live; ALWAYS restores it at the end
#   (EXIT trap covers Ctrl+C and crashes too).
# - Baseline for comparison (measured today at --max-context 32768, auto arena ~30 GB):
#     chunk 1024: 4k 110.0 / 12k 122.7 / decode mean 16.6
#     chunk 2048: 4k 128.9 / 12k 114.7 / decode mean 14.3
#     chunk 4096: 4k 125.5 / 12k 109.8 / decode mean 14.1
# - Prints swap + memory-free % after each round; if memory free drops under
#   20% or swap climbs past ~2 GB, report that and stop raising the arena.
# - Picks the winner by estimated agent-turn wall time (8000 prompt + 400
#   generated tokens) and writes chunk/arena/leave-share into
#   strata-ud-q4_k_xl-huihui.json. ~7 minutes total.
set -u
REPO="/Users/pc/src/Strata-fork"
PY="$REPO/.venv/bin/python"
CFG="$REPO/strata-ud-q4_k_xl-huihui.json"
LOG="/tmp/huihui-arena-ab.log"
CSV="/tmp/huihui-arena-ab.csv"
PIDF="/tmp/strata-huihui-ab.pid"
IQ3="serve/server.py.*iq3_xxs"
HUI="strata --serve --pack.*ud-q4_k_xl-huihui"

say() { printf '%s\n' "$*"; }
iq3_live() { curl -sf -m 3 http://127.0.0.1:8180/v1/models >/dev/null 2>&1; }
hui_live() { curl -sf -m 3 http://127.0.0.1:8181/v1/models >/dev/null 2>&1; }

restore_iq3() {
  if ! iq3_live; then
    say "restoring the IQ3_XXS server on 8180 ..."
    nohup "$REPO/run-iq3_xxs.sh" >> "$REPO/strata-iq3_xxs.log" 2>&1 &
    for _ in $(seq 1 90); do
      sleep 2
      iq3_live && { say "8180 is live again (serve pid $(pgrep -f "$IQ3" | head -1))"; return 0; }
    done
    say "WARNING: 8180 did not come back within 180 s - check strata-iq3_xxs.log"
  fi
}
cleanup() { [ -f "$PIDF" ] && kill -TERM "$(cat "$PIDF")" 2>/dev/null; restore_iq3; }
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

hui_stop() {
  [ -f "$PIDF" ] && { kill -TERM "$(cat "$PIDF")" 2>/dev/null; rm -f "$PIDF"; }
  for _ in $(seq 1 30); do
    sleep 2
    hui_live && continue
    pgrep -f "$HUI" >/dev/null && continue
    break
  done
  sleep 4
}

probe() {  # $1 = round label
  "$PY" - "$1" <<'PYEOF' | tee -a "$CSV"
import json, sys, time, urllib.request
from hashlib import sha256
round_ = sys.argv[1]
salt = sha256(str(time.time()).encode()).hexdigest()[:8]
body = {
    "short": f"Explain in two sentences why rivers make good borders. Case {salt}.",
    "4k": ("Summarize the key engineering tradeoffs of hybrid CPU/GPU inference in under 200 words. Case " + salt + ". "
           + ("Streaming expert weights from NVMe turns a fixed GPU budget into a memory hierarchy: every token "
              "that routes to a cold expert pays an SSD read, so routing skew, cache admission policy, and the "
              "size of the resident hot set all change the token rate. Chunked prompt GEMMs amortize each expert "
              "read across many rows, which is why longer prompt chunks help when the drive is the bottleneck. "
              ) * 12),
    "12k": ("Given this engineering log, list the three most likely causes of the token-rate drop. Case " + salt + ". "
            + ("At 16:40 the arena hit rate fell from 0.86 to 0.41 after another process started reading a 90 GB "
               "file; drive read latency rose from 0.9 ms to 3.1 ms per 2 MB expert; GPU utilization stayed flat. "
               ) * 26),
}
def post(payload):
    req = urllib.request.Request(
        "http://127.0.0.1:8181/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read().decode())
def timings():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8181/v1/status", timeout=10) as r:
            return (json.loads(r.read().decode()) or {}).get("last_timings") or {}
    except Exception:
        return {}
for label, mx in (("short", 96), ("short", 96), ("4k", 160), ("12k", 160)):
    payload = {"messages": [{"role": "user", "content": body[label]}],
               "temperature": 0, "max_tokens": mx, "stream": False}
    try:
        post(payload)
    except Exception as e:
        print(f"{round_},{label},FAILED,{e}")
        continue
    t = timings()
    print(f"{round_},{label},{t.get('prompt_n','?')},{t.get('prompt_per_second','?')},{t.get('predicted_per_second','?')}")
PYEOF
}

if hui_live; then
  say "a Huihui server is already live on 8181 - stop it first (tools/huihui_q4k_test.sh stop) and re-run."
  exit 1
fi
if iq3_live; then
  say "stopping the IQ3_XXS server on 8180 (it comes back at the end) ..."
  kill -TERM "$(pgrep -f "$IQ3" | head -1)" 2>/dev/null
  for _ in $(seq 1 30); do sleep 2; if ! iq3_live && ! pgrep -f "build-metal/strata" >/dev/null; then break; fi; done
  sleep 4
fi

: > "$LOG"; : > "$CSV"
for ROUND in "1024" "2048" "4096" "2048sd1"; do
  say "=== arena 33, leave 0.35, max-context 162144, $ROUND ==="
  "$PY" - "$CFG" "$ROUND" <<'PYEOF'
import json, sys
cfg, chunk = sys.argv[1], sys.argv[2]
d = json.load(open(cfg))
real = "2048" if chunk == "2048sd1" else chunk
d["env"]["STRATA_METAL_PROMPT_CHUNK"] = real
d["env"]["STRATA_METAL_ARENA_GB"] = "33"
d["env"]["STRATA_METAL_LEAVE_SHARE"] = "0.35"
a = d["args"]
a[a.index("--max-context") + 1] = "162144"
if chunk == "2048sd1":
    a[a.index("--suffix-draft") + 1] = "1"
open(f"/tmp/huihui-arena-{chunk}.json", "w").write(json.dumps(d, indent=1))
PYEOF
  nohup "$PY" "$REPO/serve/server.py" --engine strata --config "/tmp/huihui-arena-$ROUND.json" --port 8181 >> "$LOG" 2>&1 &
  echo $! > "$PIDF"
  for _ in $(seq 1 90); do
    sleep 4
    hui_live && break
    kill -0 "$(cat "$PIDF")" 2>/dev/null || { say "engine died at chunk $ROUND; see $LOG"; tail -8 "$LOG"; break 2; }
  done
  hui_live || { say "engine not ready at chunk $ROUND"; tail -8 "$LOG"; break; }
  grep -E "planned resident footprint|expert arena|wired \(mlock\)|resident footprint after" "$LOG" | tail -4
  probe "$ROUND"
  say "  swap: $(sysctl -n vm.swapusage)"
  say "  $(memory_pressure -Q 2>/dev/null | grep 'System-wide memory free')"
  hui_stop
done

say "=== summary (arena 33) vs today's baseline (auto arena ~30) ==="
"$PY" - "$CSV" "$CFG" <<'PYEOF'
import csv, json, sys
rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) == 5 and r[2] != "FAILED"]
base = {"1024": (110.0, 122.7, 16.6), "2048": (128.9, 114.7, 14.3), "4096": (125.5, 109.8, 14.1)}
print(f"{'round':8} {'4k p':>7} {'12k p':>7} {'decode':>7}   {'agent turn est_s':>16}   (32k-context baseline est_s)")
best, best_t = None, 1e9
for label in ("1024", "2048", "4096", "2048sd1"):
    big = [r for r in rows if r[0] == label and r[1] in ("4k", "12k")]
    dec = [float(r[4]) for r in rows if r[0] == label]
    if len(big) < 2 or not dec:
        print(f"{label:8}  (incomplete)")
        continue
    p4 = float(next(r[3] for r in big if r[1] == "4k"))
    p12 = float(next(r[3] for r in big if r[1] == "12k"))
    d = sum(dec) / len(dec)
    est = 8000 / ((p4 + p12) / 2) + 400 / d
    tag = "     -"
    if label in base:
        bp, bd = (base[label][0] + base[label][1]) / 2, base[label][2]
        tag = f"({8000 / bp + 400 / bd:.1f})"
    print(f"{label:8} {p4:7.1f} {p12:7.1f} {d:7.1f}   {est:10.1f}          {tag}")
    if est < best_t:
        best, best_t = label, est
if best:
    cfg = sys.argv[2]
    d = json.load(open(cfg))
    real = "2048" if best == "2048sd1" else best
    d["env"]["STRATA_METAL_PROMPT_CHUNK"] = real
    d["env"]["STRATA_METAL_ARENA_GB"] = "33"
    d["env"]["STRATA_METAL_LEAVE_SHARE"] = "0.35"
    a = d["args"]
    a[a.index("--max-context") + 1] = "162144"
    a[a.index("--suffix-draft") + 1] = "1" if best == "2048sd1" else "0"
    open(cfg, "w").write(json.dumps(d, indent=1) + "\n")
    print(f"winner: {best} at arena 33 + max-context 162144 (est agent turn {best_t:.1f} s); written to {cfg}")
PYEOF

restore_iq3
