#!/bin/bash
# Suffix-draft head-to-head for Huihui UD-Q4_K_XL: chunk 2048, arena 33,
# leave-share 0.35, max-context 162144 - suffix-draft 0 vs 1, agent-shaped
# probes (16k cold prompt; then the same conversation extended with a
# 1024-token decode turn, which exercises parking reuse).
#
#   tools/huihui_sd_ab.sh
#
# Stops 8180 if live; ALWAYS restores it (EXIT trap). ~6 minutes. Writes the
# winning --suffix-draft value into strata-ud-q4_k_xl-huihui.json.
set -u
REPO="/Users/pc/src/Strata-fork"
PY="$REPO/.venv/bin/python"
CFG="$REPO/strata-ud-q4_k_xl-huihui.json"
LOG="$REPO/strata-ud-q4_k_xl-huihui.log"
CSV="/tmp/huihui-sd-ab.csv"
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
    say "WARNING: 8180 did not come back within 180 s"
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

probe() {  # $1 = sd0|sd1
  "$PY" - "$1" <<'PYEOF' | tee -a "$CSV"
import json, sys, time, urllib.request
from hashlib import sha256
tag = sys.argv[1]
salt = sha256(str(time.time()).encode()).hexdigest()[:8]
base = ("You are a senior engineer reviewing an incident log. Answer each numbered question in order with a "
        "short paragraph plus a one-line action item. Case " + salt + ". "
        + ("At 16:40 the arena hit rate fell from 0.86 to 0.41 after another process started reading a 90 GB "
           "file; drive read latency rose from 0.9 ms to 3.1 ms per 2 MB expert; GPU utilization stayed flat; "
           "conversations longer than 32k tokens began re-reading their whole prompt every turn. "
           ) * 45)
def post(messages, mx):
    req = urllib.request.Request(
        "http://127.0.0.1:8181/v1/chat/completions",
        data=json.dumps({"messages": messages, "temperature": 0, "max_tokens": mx, "stream": False}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as r:
        r.read()
    return time.monotonic() - t0
def timings():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8181/v1/status", timeout=10) as r:
            return (json.loads(r.read().decode()) or {}).get("last_timings") or {}
    except Exception:
        return {}
# 1) cold 16k prompt, short answer
w = post([{"role": "user", "content": base + " Answer question 1 only."}], 256)
t = timings()
print(f"{tag},cold16k_decode256,{t.get('prompt_n','?')},{t.get('prompt_per_second','?')},{t.get('predicted_n','?')},{t.get('predicted_per_second','?')},{w:.1f}")
# 2) same conversation continues - long decode turn, parking reuse exercised
w = post([{"role": "user", "content": base + " Answer ALL numbered questions, each with its action item."}], 1024)
t = timings()
print(f"{tag},reused_turn_decode1024,{t.get('prompt_n','?')},{t.get('prompt_per_second','?')},{t.get('predicted_n','?')},{t.get('predicted_per_second','?')},{w:.1f}")
PYEOF
}

if hui_live; then say "a Huihui server is already live on 8181 - stop it first."; exit 1; fi
if iq3_live; then
  say "stopping the IQ3_XXS server on 8180 (it comes back at the end) ..."
  kill -TERM "$(pgrep -f "$IQ3" | head -1)" 2>/dev/null
  for _ in $(seq 1 30); do sleep 2; if ! iq3_live && ! pgrep -f "build-metal/strata" >/dev/null; then break; fi; done
  sleep 4
fi

: > "$CSV"
for SD in 0 1; do
  say "=== chunk 2048, arena 33, max-context 162144, suffix-draft $SD ==="
  "$PY" - "$CFG" "$SD" <<'PYEOF'
import json, sys
cfg, sd = sys.argv[1], sys.argv[2]
d = json.load(open(cfg))
d["env"]["STRATA_METAL_PROMPT_CHUNK"] = "2048"
d["env"]["STRATA_METAL_ARENA_GB"] = "33"
d["env"]["STRATA_METAL_LEAVE_SHARE"] = "0.35"
a = d["args"]
a[a.index("--max-context") + 1] = "162144"
a[a.index("--suffix-draft") + 1] = sd
open(f"/tmp/huihui-sd-{sd}.json", "w").write(json.dumps(d, indent=1))
PYEOF
  nohup "$PY" "$REPO/serve/server.py" --engine strata --config "/tmp/huihui-sd-$SD.json" --port 8181 >> "$LOG" 2>&1 &
  echo $! > "$PIDF"
  for _ in $(seq 1 90); do
    sleep 4
    hui_live && break
    kill -0 "$(cat "$PIDF")" 2>/dev/null || { say "engine died at sd=$SD; see $LOG"; tail -8 "$LOG"; break 2; }
  done
  hui_live || { say "engine not ready at sd=$SD"; tail -8 "$LOG"; break; }
  probe "$SD"
  grep "suffix drafts:" "$LOG" | tail -3
  say "  swap: $(sysctl -n vm.swapusage)"
  say "  $(memory_pressure -Q 2>/dev/null | grep 'System-wide memory free')"
  hui_stop
done

say "=== summary ==="
"$PY" - "$CSV" "$CFG" <<'PYEOF'
import csv, json, sys
rows = list(csv.reader(open(sys.argv[1])))
want = {"0": [], "1": []}
for r in rows:
    if len(r) == 7 and r[2] != "FAILED":
        want[r[0]].append(r)
print(f"{'sd':4} {'probe':24} {'prompt_n':>9} {'prompt t/s':>11} {'gen_n':>6} {'decode t/s':>11} {'wall_s':>7}")
est = {}
for tag in ("0", "1"):
    rs = want[tag]
    for r in rs:
        print(f"{tag:4} {r[1]:24} {r[2]:>9} {r[3]:>11} {r[4]:>6} {r[5]:>11} {r[6]:>7}")
    if len(rs) == 2:
        cold_wall, long_wall = float(rs[0][6]), float(rs[1][6])
        est[tag] = cold_wall + long_wall
        print(f"{tag:4} total agent-shaped wall: {est[tag]:.1f} s")
if len(est) == 2:
    win = min(est, key=est.get)
    cfg = sys.argv[2]
    d = json.load(open(cfg))
    d["args"][d["args"].index("--suffix-draft") + 1] = win
    open(cfg, "w").write(json.dumps(d, indent=1) + "\n")
    print(f"winner: --suffix-draft {win} (wall {est[win]:.1f} s); written to {cfg}")
PYEOF

restore_iq3
