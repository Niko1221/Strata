#!/bin/bash
# End-to-end test of the disk session path on Metal:
#   start -> ~4k-token conversation -> SAVE -> full stop -> start -> RESTORE
#   -> continuation that must reuse the restored state (almost no prefill).
#
#   tools/huihui_session_test.sh
#
# Stops the 8180 server if live and ALWAYS restores it (EXIT trap). ~4 minutes.
# Gates: save/restore succeed with matching token counts, the continuation
# reuses at least (saved - 200) tokens, no "session file:" errors in the log.
set -u
REPO="/Users/pc/src/Strata-fork"
PY="$REPO/.venv/bin/python"
LOG="$REPO/strata-ud-q4_k_xl-huihui.log"
API="http://127.0.0.1:8181"
NAME="itest-$$"
DIR="/Users/pc/src/Strata-data/sessions/ud-q4_k_xl-huihui"
IQ3="serve/server.py.*iq3_xxs"
PIDF="/tmp/strata-huihui-session.pid"

say() { printf '%s\n' "$*"; }
iq3_live() { curl -sf -m 3 http://127.0.0.1:8180/v1/models >/dev/null 2>&1; }
hui_live() { curl -sf -m 3 "$API/v1/models" >/dev/null 2>&1; }
logsize() { [ -f "$LOG" ] && wc -c < "$LOG" || echo 0; }

restore_iq3() {
  if ! iq3_live; then
    say "restoring the IQ3_XXS server on 8180 ..."
    nohup "$REPO/run-iq3_xxs.sh" >> "$REPO/strata-iq3_xxs.log" 2>&1 &
    for _ in $(seq 1 90); do sleep 2; iq3_live && { say "8180 live again"; return 0; }; done
    say "WARNING: 8180 did not come back"
  fi
}
hui_stop() { [ -f "$PIDF" ] && kill -TERM "$(cat "$PIDF")" 2>/dev/null; rm -f "$PIDF"; }
cleanup() { hui_stop; rm -f "$DIR/$NAME"; restore_iq3; }
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

hui_start() {
  nohup "$REPO/run-ud_q4_k_xl-huihui.sh" >> "$LOG" 2>&1 &
  echo $! > "$PIDF"
  for _ in $(seq 1 90); do
    sleep 4
    hui_live && return 0
    kill -0 "$(cat "$PIDF")" 2>/dev/null || { say "engine died at start; see $LOG"; return 1; }
  done
  say "engine not ready in time"; return 1
}
hui_stop_wait() {
  kill -TERM "$(cat "$PIDF")" 2>/dev/null
  for _ in $(seq 1 30); do sleep 2; kill -0 "$(cat "$PIDF")" 2>/dev/null || break; done
  kill -0 "$(cat "$PIDF")" 2>/dev/null && { say "server would not stop"; return 1; }
  rm -f "$PIDF"; sleep 3; return 0
}

chat() {  # $1 max_tokens ; $2.. messages as separate args (json-encoded in python)
  "$PY" - "$API" "$1" "${@:2}" <<'PYEOF'
import json, sys, urllib.request
api, mx = sys.argv[1], int(sys.argv[2])
msgs = [json.loads(m) for m in sys.argv[3:]]
req = urllib.request.Request(api + "/v1/chat/completions",
    data=json.dumps({"messages": msgs, "temperature": 0, "max_tokens": mx}).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=1800) as r:
    d = json.loads(r.read().decode())
msg = d["choices"][0]["message"]
sys.stderr.write(json.dumps(msg) + "\n")
print(json.dumps(msg))
PYEOF
}

corpus() {  # same case text the arena harness uses (~4k tokens with the questions)
  "$PY" - <<'PYEOF'
import json
base = ("You are a senior engineer reviewing an incident log. Answer each numbered question in order with a "
        "short paragraph plus a one-line action item. Case 2147. "
        + ("At 16:40 the arena hit rate fell from 0.86 to 0.41 after another process started reading a 90 GB "
           "file; drive read latency rose from 0.9 ms to 3.1 ms per 2 MB expert; GPU utilization stayed flat; "
           "conversations longer than 32k tokens began re-reading their whole prompt every turn. "
           ) * 38
        + " Questions: 1) What most likely caused the hit-rate fall? 2) What would you change first? "
          "3) What measurement decides it? 4) What would you tell the on-call team tonight? "
          "5) What is the follow-up for next week?")
print(json.dumps({"role": "user", "content": base}))
PYEOF
}

if hui_live; then say "a Huihui server is already live - stop it first"; exit 1; fi
if iq3_live; then
  say "stopping the IQ3_XXS server on 8180 (it comes back at the end) ..."
  kill -TERM "$(pgrep -f "$IQ3" | head -1)" 2>/dev/null
  for _ in $(seq 1 30); do sleep 2; if ! iq3_live && ! pgrep -f "build-metal/strata" >/dev/null; then break; fi; done
  sleep 4
fi

say "=== start ==="
hui_start || exit 1

say "=== conversation ==="
U=$(corpus)
L1=$(logsize)
A=$(chat 128 "$U") || exit 1
L2=$(logsize)
sed -n "/./p" < <(tail -c +$((L1 + 1)) "$LOG") | grep -E "prompt [0-9]+ tokens|generated in" | tail -2

say "=== save ==="
S=$(curl -sS -m 900 -X POST "$API/slots/0?action=save" -H "Content-Type: application/json" -d "{\"filename\": \"$NAME\"}")
say "$S"
saved=$(echo "$S" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print(d.get("n_saved") or 0)')
[ "$saved" -gt 0 ] || { say "SAVE failed"; exit 1; }

say "=== full stop ==="
hui_stop_wait || exit 1

say "=== start again ==="
hui_start || exit 1

say "=== restore ==="
R=$(curl -sS -m 900 -X POST "$API/slots/0?action=restore" -H "Content-Type: application/json" -d "{\"filename\": \"$NAME\"}")
say "$R"
restored=$(echo "$R" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print(d.get("n_restored") or 0)')
[ "$restored" = "$saved" ] || { say "RESTORE token count mismatch ($restored vs $saved)"; exit 1; }

say "=== continuation (must reuse the restored state) ==="
L3=$(logsize)
chat 96 "$U" "$A" "{\"role\": \"user\", \"content\": \"Now rank those five answers by confidence and give the single next action.\"}" >/dev/null || exit 1
L4=$(logsize)
tail -c +$((L3 + 1)) "$LOG" | grep -E "prompt [0-9]+ tokens" | tail -1

say "=== verdict ==="
"$PY" - "$LOG" "$L3" "$L4" "$saved" <<'PYEOF'
import re, sys
log, l3, l4, saved = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
with open(log, "rb") as f:
    f.seek(l3); new = f.read().decode(errors="replace")
line = None
for l in new.splitlines():
    if "prompt " in l and " tokens" in l and "reused" in l: line = l
errs = [l for l in new.splitlines() if "session file:" in l or "FATAL" in l]
m = re.search(r"= (\d+) reused", line or "")
reused = int(m.group(1)) if m else 0
ok = line and reused >= saved - 200 and not errs
print(("PASS" if ok else "FAIL") + f": continuation reused {reused} of {saved} saved tokens")
if line: print("  " + line.strip())
for e in errs: print("  error: " + e.strip())
PYEOF
