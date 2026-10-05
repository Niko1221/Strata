#!/usr/bin/env bash
# Arm runner for the IQ3_S tuning study: VRAM guard across the arm + the three
# bench phases, engine-log slice saved per arm.  Usage:
#   ./arm-run.sh <arm-label> <extra args passed through to nothing - knobs change via ./run.sh restart>
# The server must already be running on 127.0.0.1:19931.  Excluded prewarm runs first.
set -uo pipefail
LABEL="$1"
ROOT="/mnt/storage/Development/github/dhoard/strata"
D="$ROOT/bench/results/2026-10-04-iq3s-tuning"
LOG=/mnt/storage/Development/strata-work/logs/strata-hip.log
MODEL=qwen3.8-flash-next-iq3_s
URL=http://127.0.0.1:19931
START=$(stat -c %s "$LOG")

echo "== prewarm (excluded) =="
python3 "$ROOT/tools/hip/bench_prefill.py" --url "$URL" --model "$MODEL" --engine-log "$LOG" \
  --output "$D/$LABEL-prewarm.json" --label "$LABEL-prewarm" --trials 3 --start-trial 90 >/dev/null || { echo prewarm failed; exit 1; }

python3 "$ROOT/docker/vram-guard.py" --budget-mib 10240 --interval 0.05 \
  --output "$D/$LABEL-vram.json" > "$D/$LABEL-vram-guard.log" 2>&1 &
GUARD=$!
PID=$(pgrep -x strata | head -1)
GUARD2=""
# Per-process share audit must run INSIDE the container (vram-guard's documented mode):
# it reads the engine's own PID-namespace fds.  Output lands in the mounted /work.
docker exec strata-gfx1101 bash -c 'python3 /opt/strata/docker/vram-guard.py --pid $(pgrep -x strata | head -1) --budget-mib 10240 --interval 0.05 --output /work/logs/ARM-vram-share.json' > "$D/$LABEL-vram-share-guard.log" 2>&1 &
GUARD2=$!
trap 'kill -INT $GUARD $GUARD2 2>/dev/null' EXIT
echo "== guard $GUARD raw + $GUARD2 strata-share (in-container) started =="

echo "== bench_prefill (5 trials) =="
python3 "$ROOT/tools/hip/bench_prefill.py" --url "$URL" --model "$MODEL" --engine-log "$LOG" \
  --output "$D/$LABEL-prefill.json" --label "$LABEL" --trials 5 --start-trial 1 \
  | python3 -c '
import sys, json
for l in sys.stdin:
    d = json.loads(l); m = d["metrics"]
    print(" ", d["trial"], d["kind"], int(m["prompt_tokens"]), "pre", round(m["prefill_tps"],1), "dec", round(m["decode_tps"],1), "wall", round(d["wall_s"],1))' || echo "PREFILL FAILED"

echo "== bench_discover (1024/4096 x5 + followups) =="
python3 "$ROOT/tools/hip/bench_discover.py" --url "$URL" --model "$MODEL" \
  --tokenizer /mnt/storage/Development/strata-work/packs/iq3_s/tokenizer \
  --engine-log "$LOG" --output "$D/$LABEL-stream.json" --label "$LABEL" \
  --sizes 1024,4096 --repetitions 5 --followups --timeout 3600 || echo "STREAM FAILED"

echo "== coding smoke =="
python3 "$ROOT/tools/hip/check_coding_task.py" --url "$URL" --model "$MODEL" \
  --label "$LABEL" --output "$D/$LABEL-smoke.json" 2>&1 | tail -2 || echo "SMOKE FAILED"

kill -INT $GUARD 2>/dev/null; wait $GUARD 2>/dev/null
# SIGINT does not propagate through docker exec; stop the in-container guard there.
docker exec strata-gfx1101 pkill -INT -f vram-guard.py 2>/dev/null || true
wait $GUARD2 2>/dev/null
for i in $(seq 1 20); do [ -f /mnt/storage/Development/strata-work/logs/ARM-vram-share.json ] && break; sleep 1; done
cp /mnt/storage/Development/strata-work/logs/ARM-vram-share.json "$D/$LABEL-vram-share.json" 2>/dev/null && rm -f /mnt/storage/Development/strata-work/logs/ARM-vram-share.json
echo "== raw guard: $(tail -1 "$D/$LABEL-vram-guard.log") =="
[ -n "$GUARD2" ] && echo "== share guard: $(tail -1 "$D/$LABEL-vram-share-guard.log") =="

tail -c +$((START+1)) "$LOG" > "$D/$LABEL-engine.log"
python3 - "$D" "$LABEL" <<'PY'
import re, sys, pathlib
d, label = pathlib.Path(sys.argv[1]), sys.argv[2]
log = (d / (label + '-engine.log')).read_text()
for pat in [r'expert cache auto: [^\n]*', 'expert cache \\d+ slots, [^\n]*',
            'prompt chunk[^\n]*', 'the prompt path borrows [^\n]*',
            '[0-9]+ expert-pool workers [^\n]*', '[0-9.]+ MiB of VRAM free [^\n]*',
            'hit rate: [^\n]*']:
    for m in re.findall(pat, log)[:3]:
        print(' ', m.strip())
PY
