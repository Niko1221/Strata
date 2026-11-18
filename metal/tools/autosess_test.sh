#!/bin/bash
# auto_session verification: idle save -> full server restart -> read-back -> conversation still answers.
# Run through the harness so the server is always left healthy:
#   ./metal/tools/serve_huihui.sh run metal/tools/autosess_test.sh
# Needs the port free (the harness stops the managed server) and NO other engine on this Mac.
set -u
cd /Users/pc/src/Strata-fork
H=./metal/tools/serve_huihui.sh
RLOG=/tmp/bw/restart.log
URL=http://127.0.0.1:8180/v1/chat/completions
MODEL=qwen3.8-flash-next-ud-q4_k_xl-huihui
SES=/Users/pc/src/Strata-data/sessions/ud-q4_k_xl-huihui/auto-session
PY=.venv/bin/python

say() { echo "[autosess $(date +%H:%M:%S)] $*"; }

mkdir -p /tmp/bw
$H start || { say "FAIL: first start"; exit 1; }
MARK=$(wc -l < "$RLOG")

msg1="I have three plants in my room: an orchid, a fern, and a cactus. The orchid is on the third shelf, the fern is by the window, and the cactus sits in a blue pot on the desk. Acknowledge that you understood in one short sentence."
t0=$(date +%s)
out1=$(curl -s --max-time 300 "$URL" -H 'Content-Type: application/json' -d "$($PY -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":sys.argv[2]}],"max_tokens":160,"temperature":0}))' "$MODEL" "$msg1")")
rep1=$(printf '%s' "$out1" | $PY -c 'import json,sys
d=json.load(sys.stdin)["choices"][0]["message"]
print((d.get("content") or d.get("reasoning_content") or d.get("reasoning") or "").strip())' 2>/dev/null)
say "turn1 ($(( $(date +%s)-t0 ))s): ${rep1:0:90}"
[ -n "$rep1" ] || { say "FAIL: no turn1 answer ($out1)"; $H stop; exit 1; }

say "waiting 14 s for the idle save ..."
sleep 14
if [ -f "$SES" ]; then say "session file: $(ls -l "$SES" | awk '{print $5" bytes, "$6" "$7" "$8}')"; else say "FAIL: no session file at $SES"; $H stop; exit 1; fi
sed -n "${MARK},\$p" "$RLOG" | grep 'auto-session: saved' || { say "FAIL: no save line in server stdout"; $H stop; exit 1; }

say "restarting the whole server (stop+start) ..."
M2=$(wc -l < "$RLOG")
$H stop && $H start || { say "FAIL: restart"; exit 1; }
sed -n "${M2},\$p" "$RLOG" | grep 'auto-session: restored' || { say "FAIL: nothing restored at the start"; exit 1; }

say "turn2: resend the whole conversation ..."
t0=$(date +%s)
out2=$(curl -s --max-time 600 "$URL" -H 'Content-Type: application/json' -d "$($PY -c 'import json,sys
print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":sys.argv[2]},{"role":"assistant","content":sys.argv[3]},{"role":"user","content":"Which shelf is the orchid on? Answer with just the number."}],"max_tokens":600,"temperature":0}))' "$MODEL" "$msg1" "$rep1")")
rep2=$(printf '%s' "$out2" | $PY -c 'import json,sys
d=json.load(sys.stdin)["choices"][0]["message"]
print((d.get("content") or d.get("reasoning_content") or d.get("reasoning") or "").strip())' 2>/dev/null)
say "turn2 ($(( $(date +%s)-t0 ))s): ${rep2:0:140}"
printf '%s' "$out2" | grep -qE '"content"|"reasoning' || { say "FAIL: turn2 got nothing"; $H stop; exit 1; }

$H stop
say "done (harness will now restart the main server)"
