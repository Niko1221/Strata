#!/usr/bin/env bash
# Every measurement this report is made of, in one serial background run.
#
#   1  interleave500   10 interleaved pairs of 500-token greedy generations, A vs
#                      B (--pipeline-windows 1), one server restart per request
#   2  arm pipeline-off      --pipeline-windows 1          (ab-run.sh)
#   3  arm stage-trim-off    STRATA_STAGE_TRIM=0           (ab-run.sh)
#   4  arm resident          --resident-budget-gib 30      (ab-run.sh)
#   5  soak            30 minutes with --resident-budget-gib 30, RAM/VRAM every 60 s
#   6  arm batch2       --batch 2
#   7  arm batch4       --batch 4
#
# Everything is serial on purpose: every step starts its own server on port 8080
# and kills any other. Logs go to the report's data/ folder.
set -u
REPORT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
ROOT="$(cd -- "$REPORT/../../.." && pwd)"
DATA=$REPORT/data
S=$REPORT/scripts
LOG=$ROOT/strata-coder-iq1_m.log
PY=$ROOT/.venv/bin/python
MODEL=qwen3.8-flash-next-coder-iq1_m

# no browser tabs from ab-run.sh's --open (webbrowser honours $BROWSER)
export BROWSER=true

step() { echo; echo "[$(date -Is)] === $* ==="; echo; }

kill_server() {
  pkill -f "serve/server.py" 2>/dev/null
  pkill -f "engine/strata --serve" 2>/dev/null
  sleep 5
}

mkdir -p "$DATA"
kill_server

# --- 1. the interleaved 500-token A/B ---------------------------------------
step "interleave500: 10 pairs, A = pipeline 2, B = pipeline 1"
"$PY" "$S/build_cfg.py" "$DATA/cfg-interleave-B.json" --arg --pipeline-windows 1
off=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
"$PY" "$S/interleave500.py" \
  --config-a "$ROOT/strata-coder-iq1_m.json" \
  --config-b "$DATA/cfg-interleave-B.json" \
  --pairs 10 --model "$MODEL" --engine-log "$LOG" \
  --output "$DATA/interleave500.json" \
  --server-out "$DATA/interleave500-server.out"
tail -c +$((off + 1)) "$LOG" > "$DATA/engine-interleave.log"

# --- 2..4, 6..7. the A/B harness arms ---------------------------------------
step "arm pipeline-off"
bash "$S/run-arm.sh" pipeline-off --arg --pipeline-windows 1

step "arm stage-trim-off (the base config already has STRATA_STAGE_TRIM=1)"
bash "$S/run-arm.sh" stage-trim-off --env STRATA_STAGE_TRIM=0

step "arm resident"
bash "$S/run-arm.sh" resident --arg --resident-budget-gib 30

# --- 5. the 30-minute soak ---------------------------------------------------
step "soak: 30 minutes, --resident-budget-gib 30, samples every 60 s"
kill_server
"$PY" "$S/build_cfg.py" "$DATA/cfg-soak.json" --arg --resident-budget-gib 30
"$PY" "$S/soak.py" --config "$DATA/cfg-soak.json" --model "$MODEL" \
  --engine-log "$LOG" --out-dir "$DATA" --minutes 30 --interval 60

step "arm batch2"
bash "$S/run-arm.sh" batch2 --arg --batch 2

step "arm batch4"
bash "$S/run-arm.sh" batch4 --arg --batch 4

step "ALL DONE"
kill_server
