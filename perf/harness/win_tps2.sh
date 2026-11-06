#!/bin/bash
# Pure decode throughput for one config: bash ab/win_tps2.sh <label> <config.json> [ENV=VAL ...]
set -u
LABEL="$1"; CFG="$2"; shift 2
ENG=/home/des/test/strata; PORT=8089
KEY=$(python3 -c "import json;print(json.load(open('$ENG/strata-sc117-iq3_s.json'))['api_key'])")
LOG=$ENG/ab/arm-tps2-$LABEL.log
su - des -c "XDG_RUNTIME_DIR=/run/user/\$(id -u des) systemctl --user stop strata-sc117" >/dev/null 2>&1
pkill -TERM -u des -f "serve.server --engine" >/dev/null 2>&1
for i in $(seq 1 120); do ss -ltn 2>/dev/null | grep -q ":8089 " || break; sleep 1; done
sleep 2; : > "$LOG"; chmod 666 "$LOG"
: > /tmp/ab_env; for e in "$@"; do echo "$e" >> /tmp/ab_env; done; chmod 644 /tmp/ab_env
su des -c "cd $ENG && AB_CFG=$CFG setsid nohup bash $ENG/ab/launch.sh >> $LOG 2>&1 < /dev/null &" >/dev/null 2>&1
for i in $(seq 1 420); do curl -sf -H "Authorization: Bearer $KEY" http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
echo "### $LABEL ready (cfg $CFG, env: $*)"
cd "$ENG" || exit 1
STRATA_KEY="$KEY" .venv/bin/python ab/win_tps2.py --chars "${CH:-160000}" --max-tokens 192 --port "$PORT" --label "$LABEL" \
    | tee ab/tps2-$LABEL.txt
echo "### $LABEL stalls: $(grep -c 'no progress\|exit code -6' "$LOG")"
pkill -TERM -u des -f "serve.server --engine" >/dev/null 2>&1
for i in $(seq 1 40); do pgrep -u des -f "serve.server --engine" >/dev/null || break; sleep 1; done
