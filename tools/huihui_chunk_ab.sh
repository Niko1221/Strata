#!/bin/bash
# Chunk-width A/B for the Huihui UD-Q4_K_XL engine: prompt throughput at
# STRATA_METAL_PROMPT_CHUNK 1024 / 2048 / 4096, one engine at a time.
#
#   tools/huihui_chunk_ab.sh
#
# - Stops the IQ3_XXS server on 8180 if it is live, and ALWAYS brings it back at
#   the end (also on Ctrl+C / error) via ./run-iq3_xxs.sh.
# - Runs the Huihui engine alone at each chunk, probes (short x2, 4k, 12k),
#   stops it, moves on. Nothing else on 8181 but this script.
# - Prints a comparison table, writes the winning chunk into
#   strata-ud-q4_k_xl-huihui.json (env STRATA_METAL_PROMPT_CHUNK).
# - ~6 minutes total. Swap is printed per step; if it grows past ~4 GB say so
#   in your report and stop chasing bigger chunks.
set -u
REPO="/Users/pc/src/Strata-fork"
PY="$REPO/.venv/bin/python"
CFG="$REPO/strata-ud-q4_k_xl-huihui.json"
LOG="/tmp/huihui-ab.log"
CSV="/tmp/huihui-ab-probes.csv"
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
  sleep 4   # let the wired arena actually be released
}

probe() {  # $1 = chunk label
  "$PY" - "$1" <<'PYEOF' | tee -a "$CSV"
import json, sys, time, urllib.request
from hashlib import sha256
chunk = sys.argv[1]
salt = sha256(str(time.time()).encode()).hexdigest()[:8]
body = {
    "short": f"Explain in two sentences why rivers make good borders. Case {salt}.",
    "4k": ("Summarize the key engineering tradeoffs of hybrid CPU/GPU inference in under 200 words. Case " + salt + ". "
           + ("Streaming expert weights from NVMe turns a fixed GPU budget into a memory hierarchy: every token "
              "that routes to a cold expert pays an SSD read, so routing skew, cache admission policy, and the "
              "size of the resident hot set all change the token rate. Chunked prompt GEMMs amortize each expert "
              "read across many rows, which is why longer prompt chunks help when the drive is the bottleneck. "
              ) * 12),
    "12k": ("You are a senior engineer. Diagnose the three most likely causes of a Mac that balloons memory and "
            "shows low GPU use under two inference engines, with fixes. Case " + salt + ". "
            + "Check wired arenas, mlock, the compressor, swap counters, and per-process resident footprints. " * 40),
}
def post(payload):
    req = urllib.request.Request("http://127.0.0.1:8181/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
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
    t0 = time.monotonic()
    try:
        post(payload)
    except Exception as e:
        print(f"{chunk},{label},FAILED,{e}")
        continue
    t = timings()
    print(f"{chunk},{label},{t.get('prompt_n','?')},{t.get('prompt_per_second','?')},{t.get('predicted_per_second','?')}")
PYEOF
}

# the guard also restores 8180 on abnormal exit; start clean
if hui_live; then
  say "a Huihui server is already live on 8181 - stop it first (tools/huihui_q4k_test.sh stop) and re-run."
  exit 1
fi
if iq3_live; then
  say "stopping the IQ3_XXS server on 8180 (it comes back at the end) ..."
  SRV=$(pgrep -f "$IQ3" | head -1)
  kill -TERM "$SRV" 2>/dev/null
  for _ in $(seq 1 30); do sleep 2; if ! iq3_live && ! pgrep -f "build-metal/strata" >/dev/null; then break; fi; done
  sleep 4
fi

: > "$LOG"; : > "$CSV"
for CHUNK in 1024 2048 4096; do
  say "=== chunk $CHUNK ==="
  "$PY" - "$CFG" "$CHUNK" <<'PYEOF'
import json, sys
cfg, chunk = sys.argv[1], sys.argv[2]
d = json.load(open(cfg))
d["env"]["STRATA_METAL_PROMPT_CHUNK"] = chunk
d.pop("open_browser", None)
open(f"/tmp/huihui-chunk-{chunk}.json", "w").write(json.dumps(d, indent=1))
PYEOF
  nohup "$PY" "$REPO/serve/server.py" --engine strata --config "/tmp/huihui-chunk-$CHUNK.json" --port 8181 >> "$LOG" 2>&1 &
  echo $! > "$PIDF"
  for _ in $(seq 1 90); do
    sleep 4
    hui_live && break
    kill -0 "$(cat "$PIDF")" 2>/dev/null || { say "engine died at chunk $CHUNK; see $LOG"; tail -8 "$LOG"; break 2; }
  done
  hui_live || { say "engine not ready at chunk $CHUNK"; tail -8 "$LOG"; break; }
  grep -E "planned resident footprint|expert arena|prompts on the matrix units" "$LOG" | tail -3
  probe "$CHUNK"
  say "  swap: $(sysctl -n vm.swapusage)"
  hui_stop
done

say "=== summary ==="
"$PY" - "$CSV" "$CFG" <<'PYEOF'
import csv, json, sys
rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) == 5 and r[2] != "FAILED"]
print(f"{'chunk':6} {'4k prompt t/s':>14} {'12k prompt t/s':>15} {'decode t/s (mean)':>18}")
best, best_score = None, -1.0
for chunk in ("1024", "2048", "4096"):
    big = [r for r in rows if r[0] == chunk and r[1] in ("4k", "12k")]
    dec = [float(r[4]) for r in rows if r[0] == chunk and r[1] != "FAILED"]
    if len(big) < 2:
        print(f"{chunk:6} {'(incomplete)':>14}")
        continue
    p4 = float(next(r[3] for r in big if r[1] == "4k"))
    p12 = float(next(r[3] for r in big if r[1] == "12k"))
    print(f"{chunk:6} {p4:14.1f} {p12:15.1f} {sum(dec)/len(dec):18.1f}")
    score = p4 + p12
    if score > best_score:
        best, best_score = chunk, score
if best:
    cfg = sys.argv[2]
    d = json.load(open(cfg))
    old = d["env"].get("STRATA_METAL_PROMPT_CHUNK")
    d["env"]["STRATA_METAL_PROMPT_CHUNK"] = best
    open(cfg, "w").write(json.dumps(d, indent=1) + "\n")
    print(f"winner: chunk {best} (was {old}); written to {cfg}")
PYEOF

# trap EXIT also runs restore; run it now so ordering is clear
restore_iq3
