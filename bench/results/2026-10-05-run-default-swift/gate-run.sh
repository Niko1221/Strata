#!/usr/bin/env bash
# Final gates for the shipping config: long-context ladder (32768/65536/130944 with
# follow-ups) + cancel/recovery, under both VRAM guards sampling the whole window.
set -uo pipefail
LABEL="$1"
ROOT="/mnt/storage/Development/github/dhoard/strata"
D="$ROOT/bench/results/2026-10-05-run-default-swift"
LOG=/mnt/storage/Development/strata-work/logs/strata-hip.log
MODEL="${ARM_MODEL:-qwen3.8-flash-next-iq3_s}"
TOKENIZER="${ARM_TOKENIZER:-/mnt/storage/Development/strata-work/packs/iq3_s/tokenizer}"
URL=http://127.0.0.1:19931
START=$(stat -c %s "$LOG")

python3 "$ROOT/docker/vram-guard.py" --budget-mib 10240 --interval 0.05 \
  --output "$D/$LABEL-gate-vram.json" > "$D/$LABEL-gate-vram-guard.log" 2>&1 &
GUARD=$!
docker exec strata-gfx1101 bash -c 'python3 /opt/strata/docker/vram-guard.py --pid $(pgrep -x strata | head -1) --budget-mib 10240 --interval 0.05 --output /work/logs/gate-share.json' > "$D/$LABEL-gate-share-guard.log" 2>&1 &
GUARD2=$!
trap 'kill -INT $GUARD 2>/dev/null; docker exec strata-gfx1101 pkill -INT -f vram-guard.py 2>/dev/null' EXIT

echo "== long-context ladder =="
python3 "$ROOT/tools/hip/bench_discover.py" --url "$URL" --model "$MODEL" \
  --tokenizer "$TOKENIZER" --engine-log "$LOG" --output "$D/$LABEL-gate-long.json" \
  --label "$LABEL-gate" --sizes 32768,65536,130944 --repetitions 1 \
  --followups --timeout 7000 || echo "LADDER FAILED"

echo "== cancel/recovery =="
python3 "$D/cancel-probe.py" --url "$URL" --model "$MODEL" \
  --output "$D/$LABEL-gate-cancel.json" || echo "CANCEL FAILED"

echo "== coding smoke =="
python3 "$ROOT/tools/hip/check_coding_task.py" --url "$URL" --model "$MODEL" \
  --label "$LABEL-gate" --output "$D/$LABEL-gate-smoke.json" 2>&1 | tail -2 || echo "SMOKE FAILED"

kill -INT $GUARD 2>/dev/null; wait $GUARD 2>/dev/null
docker exec strata-gfx1101 pkill -INT -f vram-guard.py 2>/dev/null || true
wait $GUARD2 2>/dev/null
for i in $(seq 1 20); do [ -f /mnt/storage/Development/strata-work/logs/gate-share.json ] && break; sleep 1; done
cp /mnt/storage/Development/strata-work/logs/gate-share.json "$D/$LABEL-gate-vram-share.json" 2>/dev/null && rm -f /mnt/storage/Development/strata-work/logs/gate-share.json
echo "== raw guard: $(tail -1 "$D/$LABEL-gate-vram-guard.log") =="
[ -s "$D/$LABEL-gate-vram-share.json" ] && python3 -c "import json; v=json.load(open('$D/$LABEL-gate-vram-share.json')); print('== share guard: peak_strata_share', v['peak_strata_share_mib'], 'verdict', v['verdict'], 'samples', v['samples'])"
tail -c +$((START+1)) "$LOG" > "$D/$LABEL-gate-engine.log"
grep -E "prompt 130944|prompt 65[0-9]+ tokens|prompt 32768|reused|cancel" "$D/$LABEL-gate-engine.log" | tail -8
