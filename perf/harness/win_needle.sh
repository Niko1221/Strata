#!/bin/bash
# Long-context needles + a decode-speed probe for the oct3 exact window (kv_exact_window.hpp).
#   bash ab/win_needle.sh <label> [ENV=VAL ...]     e.g. ... win1 STRATA_KV_WINDOW=1
# Runs one engine off the production config (launch.sh + /tmp/ab_env), 96k and 128k needles at depth 50, then two
# 400-token generations at a short prompt (the second one warm) so the server's "decode N tok/s" line is comparable.
set -u
LABEL="$1"; shift
ENG=/home/des/test/strata; PORT=8089
KEY=$(python3 -c "import json;print(json.load(open('$ENG/strata-sc117-iq3_s.json'))['api_key'])")
LOG=$ENG/ab/arm-winneedle-$LABEL.log
su - des -c "XDG_RUNTIME_DIR=/run/user/\$(id -u des) systemctl --user stop strata-sc117" >/dev/null 2>&1
pkill -TERM -u des -f "serve.server --engine" >/dev/null 2>&1
for i in $(seq 1 120); do ss -ltn 2>/dev/null | grep -q ":8089 " || break; sleep 1; done
sleep 2; : > "$LOG"; chmod 666 "$LOG"
: > /tmp/ab_env; echo "STRATA_KV_OCT=1" >> /tmp/ab_env
for e in "$@"; do echo "$e" >> /tmp/ab_env; done
chmod 644 /tmp/ab_env
su des -c "cd $ENG && setsid nohup bash $ENG/ab/launch.sh >> $LOG 2>&1 < /dev/null &" >/dev/null 2>&1
for i in $(seq 1 420); do curl -sf -H "Authorization: Bearer $KEY" http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
echo "### $LABEL ready (env: $*)"
cd $ENG || exit 1
timeout 3000 .venv/bin/python tools/needle_bench.py --lengths 96k,128k --depths 50 \
  --url http://127.0.0.1:$PORT --api-key "$KEY" --out ab/needles-win-$LABEL.json 2>&1 | tail -6
BODY='{"model":"strata","max_tokens":400,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"Write a detailed technical explanation of how a paged KV cache works in an LLM inference engine, covering pages, blocks, streaming and eviction."}]}'
for r in 1 2; do
  curl -sf -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d "$BODY" \
    http://127.0.0.1:$PORT/v1/chat/completions > /dev/null || echo "decode probe $r failed"
done
echo "### $LABEL decode:"; grep -o "decode [0-9.]* tok/s" "$LOG" | tail -3
echo "### $LABEL stalls: $(grep -c 'no progress\|exit code -6' "$LOG")"
pkill -TERM -u des -f "serve.server --engine" >/dev/null 2>&1
for i in $(seq 1 40); do pgrep -u des -f "serve.server --engine" >/dev/null || break; sleep 1; done
