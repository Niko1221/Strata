#!/bin/bash
# Prefill + decode curve for one engine config, cold prefixes with unique heads:
#   bash ab/win_perf.sh <label> <config.json> [ENV=VAL ...]
# ctx_curve.py at 32k/96k, 4 reps each: rep 0 warms the expert cache and is dropped in the analysis
# (ab/win_summary.py does that).  GPU clock/power are sampled throughout so a throttled run is visible.
set -u
LABEL="$1"; CFG="$2"; shift 2
ENG=/home/des/test/strata; PORT=8089
KEY=$(python3 -c "import json;print(json.load(open('$ENG/strata-sc117-iq3_s.json'))['api_key'])")
LOG=$ENG/ab/arm-perf-$LABEL.log
su - des -c "XDG_RUNTIME_DIR=/run/user/\$(id -u des) systemctl --user stop strata-sc117" >/dev/null 2>&1
for i in $(seq 1 30); do pgrep -u des -f "serve.server --engine" >/dev/null || break; sleep 1; done
sleep 2; : > "$LOG"; chmod 666 "$LOG"
: > /tmp/ab_env; for e in "$@"; do echo "$e" >> /tmp/ab_env; done; chmod 644 /tmp/ab_env
su des -c "cd $ENG && AB_CFG=$CFG setsid nohup bash $ENG/ab/launch.sh >> $LOG 2>&1 < /dev/null &" >/dev/null 2>&1
for i in $(seq 1 420); do curl -sf -H "Authorization: Bearer $KEY" http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
echo "### $LABEL ready (cfg $CFG, env: $*)"
cd "$ENG" || exit 1
( while :; do
    c=$(cat /sys/class/drm/card0/device/pp_dpm_sclk 2>/dev/null | grep '\*' | awk '{print $2}')
    p=$(cat /sys/class/drm/card0/device/hwmon/hwmon*/power1_average 2>/dev/null | head -1)
    echo "$(date +%s) ${c:-?} ${p:-?}"
    sleep 2
  done ) > ab/gpu-$LABEL.txt 2>/dev/null &
GPU=$!
STRATA_KEY="$KEY" .venv/bin/python ab/ctx_curve.py --sizes 32000,96000 --max-tokens 192 --reps 4 \
    --port "$PORT" 2>/dev/null | tee ab/perf-$LABEL.jsonl
kill $GPU 2>/dev/null
echo "### $LABEL stalls: $(grep -c 'no progress\|exit code -6' "$LOG")"
echo "### $LABEL vram-free: $(grep -o '[0-9]* MiB of VRAM free' "$LOG" | tail -1)"
echo "### $LABEL clocks: $(awk '{print $2}' ab/gpu-$LABEL.txt | sort | uniq -c | tr '\n' ' ')"
pkill -TERM -u des -f "serve.server --engine" >/dev/null 2>&1
for i in $(seq 1 40); do pgrep -u des -f "serve.server --engine" >/dev/null || break; sleep 1; done
