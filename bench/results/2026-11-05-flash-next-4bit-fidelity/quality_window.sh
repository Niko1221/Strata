#!/bin/bash
# Quality window for the two 4-bit arms: production brain stopped BY NAME, three scored legs,
# brain restored on ANY exit path (trap), watchdog stays paused outside this script.
# Never select containers by image here — production runs the same image as these tool containers.
set -u
IMG=strata-local:0.2.10-cuda13
PYIN=/opt/strata/terminal/python/bin/python3
Q=/mnt/cache/strata/quality-run
S=/mnt/data/strata-models
TS=20261105-2014   # reuse the cfgs derived at 20:14 (substitution-only, port 9001, own log paths)
IDS=/data/build-q36/Strata/data/quant_quality/v1/set.ids.json
log(){ echo "[$(date +%H:%M:%S)] $*"; }

restore(){
  log "RESTORE: starting strata-0210"
  docker start strata-0210 >/dev/null 2>&1
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $(cat /mnt/cache/strata/studio.key)" http://localhost:9000/v1/models)
    [ "$c" = "200" ] && { log "RESTORE: brain HTTP 200 after $((i*10))s"; break; }
    sleep 10
  done
  docker ps --filter name=strata-0210 --format "RESTORE: {{.Names}} {{.Status}}"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/RESTORE: gpu MiB /'
  ls -la $Q/*-$TS.tsv 2>/dev/null | sed 's/^/ARTIFACT: /'
}
trap restore EXIT

log "1/4 stop production BY NAME and confirm the GPU is free"
docker stop strata-0210
sleep 8
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/   gpu: /'
free -g | sed -n '2p;3p' | sed 's/^/   /'
docker ps --format "{{.Names}}" | grep -c strata | sed 's/^/   strata containers still up: /'

run_leg(){
  log "   leg: $1"
  ionice -c2 -n7 nice -n15 docker run --rm --gpus all \
    -v /mnt/cache/strata:/data -v $S:/data/Strata-models -v /mnt/cache/tools:/tools:ro \
    --entrypoint $PYIN $IMG \
    /data/quality-run/quant_quality.py run --config /data/quality-run/cfg-$1-$TS.json \
    --out /data/quality-run/$1-$TS.tsv --ids $IDS 2>&1 | tail -4 | sed 's/^/     /'
  if [ -f $Q/$1-$TS.tsv ]; then
    log "     OK rows=$(wc -l < $Q/$1-$TS.tsv) size=$(du -h $Q/$1-$TS.tsv | cut -f1)"
  else
    log "     NO TSV for $1 — this leg failed; stderr tail:"
    tail -5 $Q/$1-$TS.tsv.stderr.txt 2>/dev/null | sed 's/^/       /'
  fi
}

log "2/4 legs (cfgs derived earlier, substitution-only, port 9001)"
for a in ref iq4xs q4kxl; do
  [ -f $Q/cfg-$a-$TS.json ] || { log "   cfg-$a-$TS.json missing — re-derive needed"; continue; }
  run_leg $a
done

log "3/4 compare (per segment + json)"
docker run --rm -v /mnt/cache/strata:/data --entrypoint $PYIN $IMG \
  /data/quality-run/quant_quality.py compare \
  $(for a in ref iq4xs q4kxl; do [ -f $Q/$a-$TS.tsv ] && printf '/data/quality-run/%s-%s.tsv ' $a $TS; done) \
  --segments --json /data/quality-run/compare-4bit-$TS.json 2>&1 | sed 's/^/   /'

log "4/4 artefacts for the ledger"
ls -la $Q/*-$TS.* 2>/dev/null | sed 's/^/   /'
echo "TS=$TS"
log "QUALITYWINDOW DONE"
