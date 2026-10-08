#!/usr/bin/env bash
# Dedicated-GPU llama.cpp (club-3090) community benchmark: Qwen3.8-27B UD-IQ4_XS on one RTX 3090.
# Stops Strata + the desktop's GPU neighbours, launches the club-3090 container on loopback,
# runs the 4K/32K/128K sweep + needle recall with a monitor, then brings the container down
# and restores Strata + neighbours. The trap restores on any exit.
set -euo pipefail

ROOT=/home/jfd/repositories/Strata
OUT="$ROOT/bench/results/2026-10-07-community-rtx-3090-128gb"
DED="$OUT/club3090"
PY="$ROOT/.venv/bin/python"
CLUB=/home/jfd/repositories/club-3090
URL=http://127.0.0.1:8090
MODEL_DIR=/mnt/nvme1/models/lmstudio
CONTAINER=llama-cpp-qwen38-27b-single
NEIGHBOURS="comfyui openclaw-gateway"
MON_PID=""
STOPPED=0

log(){ printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

restore(){
  local rc=$?
  trap - EXIT INT TERM
  log "restore: container down + Strata + neighbours"
  ( cd "$CLUB" && bash scripts/switch.sh --down ) || log "WARN: switch.sh --down failed"
  systemctl --user is-active --quiet strata || systemctl --user start strata || log "WARN: strata not active"
  if [ "$STOPPED" = 1 ]; then systemctl --user start $NEIGHBOURS || log "WARN: could not start neighbours"; fi
  [ -n "$MON_PID" ] && kill "$MON_PID" 2>/dev/null || true
  "$PY" - "$OUT/state-after.json" <<'PYEOF' || true
import json,subprocess,sys,pathlib,urllib.request
names=["strata","comfyui","openclaw-gateway"]
out=subprocess.run(["systemctl","--user","is-active",*names],capture_output=True,text=True).stdout.split()
d={"units":dict(zip(names,out))}
try: d["status"]=json.load(urllib.request.urlopen("http://127.0.0.1:8080/v1/status",timeout=10))
except Exception as e: d["status_error"]=str(e)
pathlib.Path(sys.argv[1]).write_text(json.dumps(d,indent=2)+"\n")
PYEOF
  log "restore done (exit $rc)"
  exit $rc
}
trap restore EXIT INT TERM

wait_ready(){ for _ in $(seq 1 180); do curl -fsS --max-time 5 "$URL/props" >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }

mkdir -p "$DED"

# 1. state before
log "snapshot state-before"
"$PY" - "$DED/state-before.json" <<'PYEOF'
import json,subprocess,sys,pathlib
names=["strata","comfyui","openclaw-gateway"]
out=subprocess.run(["systemctl","--user","is-active",*names],capture_output=True,text=True).stdout.split()
pathlib.Path(sys.argv[1]).write_text(json.dumps({"units":dict(zip(names,out))},indent=2)+"\n")
PYEOF

# 2. free the GPU
log "stop Strata + neighbours"
systemctl --user stop strata $NEIGHBOURS
STOPPED=1

# 3. launch the club-3090 container on loopback, greedy, thinking off
log "launch llamacpp/qwen38-27b-single-iq4xs on 127.0.0.1:8090"
( cd "$CLUB" && MODEL_DIR="$MODEL_DIR" BIND_HOST=127.0.0.1 PORT=8090 INSTRUCT=1 TEMP=0 PRESENCE_PENALTY=0 \
    bash scripts/switch.sh --force llamacpp/qwen38-27b-single-iq4xs --no-wait )
wait_ready || { log "ERROR: server not ready"; exit 1; }
log "server ready"

# 4. monitor + sweep + recall
"$PY" "$OUT/monitor-club3090.py" "$DED/telemetry.jsonl" "$CONTAINER" "$URL" &
MON_PID=$!
"$PY" "$OUT/benchmark-club-3090.py" --url "$URL" --out "$DED" --targets 4096,32768,128000 --runs 3
if [ "${SKIP_NEEDLES:-0}" != "1" ]; then
  "$PY" "$OUT/needle_bench-club3090.py" --url "$URL" --lengths 32k,128k --depths 10,50,90 --out "$DED/needles.json"
fi
kill "$MON_PID" 2>/dev/null || true; MON_PID=""

# 5. copy the container log for provenance
docker logs "$CONTAINER" > "$DED/container.log" 2>&1 || true

log "benchmark complete"
