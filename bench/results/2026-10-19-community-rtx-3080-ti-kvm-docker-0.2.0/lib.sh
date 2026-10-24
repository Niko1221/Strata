#!/bin/bash
# Round 2 bench helpers on the KVM guest: one Strata container at a time on the RTX 3080 Ti, port 127.0.0.1:8090.
set -u
R2=$HOME/strata-trial/r2
GPU=<GPU UUID>
PORT=8090
URL=http://127.0.0.1:$PORT
IMG_OLD=strata:v0.1.43
IMG_NEW=strata:0.2.0-beta-cuda13
PROD_MODELS=/var/lib/docker/volumes/strata-data/_data/models
CORPUS_DIR=$HOME/strata-trial/Strata-0.2.0
export STRATA_API_KEY
STRATA_API_KEY=$(cat "$R2/key")

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$R2/progress.log"; }

stop_bench() {
  docker stop -t 120 strata-bench >/dev/null 2>&1
  docker rm -f strata-bench >/dev/null 2>&1
  return 0
}

wait_ready() {  # wait_ready MAX_SECONDS
  local max=$1 t=0
  while [ "$t" -lt "$max" ]; do
    if ! docker ps --format '{{.Names}}' | grep -qx strata-bench; then
      log "container exited"; docker logs --tail 40 strata-bench 2>&1 | tee -a "$R2/progress.log"; return 1
    fi
    if curl -fsS -m 10 -H "Authorization: Bearer $STRATA_API_KEY" "$URL/v1/models" >/dev/null 2>&1; then
      curl -fsS -m 300 -H "Authorization: Bearer $STRATA_API_KEY" -H 'Content-Type: application/json' \
        -d '{"model":"strata","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"reasoning_effort":"none"}' \
        "$URL/v1/chat/completions" >/dev/null 2>&1 && return 0
    fi
    sleep 15; t=$((t + 15))
  done
  log "not ready after $max s"; return 1
}

# start_old VOLUME FAMILY MODEL GGUF_HOST_DIR
start_old() {
  stop_bench
  docker run -d --name strata-bench --gpus "device=$GPU" -p 127.0.0.1:$PORT:8080 --ulimit memlock=-1 \
    --shm-size 2g --stop-timeout 120 -v "$1":/data -v "$4":/gguf:ro \
    -e FAMILY="$2" -e MODEL="$3" -e CONTEXT=262144 -e VISION=no -e HOST=0.0.0.0 -e GGUF_DIR=/gguf \
    -e API_KEY="$STRATA_API_KEY" "$IMG_OLD" >/dev/null
}

# start_new VOLUME FAMILY MODEL MODEL_HOST_DIR|- [extra -e args...]
start_new() {
  local vol=$1 fam=$2 model=$3 dir=$4; shift 4
  local mnt=()
  [ "$dir" != "-" ] && mnt=(-v "$dir":/models:ro -e MODEL_PATH=/models)
  stop_bench
  docker run -d --name strata-bench --gpus "device=$GPU" -p 127.0.0.1:$PORT:8080 --ulimit memlock=-1:-1 \
    --shm-size 2g --stop-timeout 120 -v "$vol":/data "${mnt[@]}" \
    -e FAMILY="$fam" -e MODEL="$model" -e CONTEXT=262144 -e VISION=off \
    -e API_KEY="$STRATA_API_KEY" "$@" "$IMG_NEW" >/dev/null
}

# sweep LABEL OUTFILE
sweep() {
  log "sweep $1"
  python3 "$R2/sweep.py" --url "$URL" --label "$1" --sizes short,4k,32k,128k --runs 3 --out "$R2/out/$2" \
    "$CORPUS_DIR"/docs/*.md "$CORPUS_DIR"/README*.md >>"$R2/progress.log" 2>&1
}

snap() {  # snap LABEL: memory and log snapshot after load
  {
    echo "== $1 $(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=name,memory.used,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader
    free -g | head -2
  } >>"$R2/out/snapshots.txt"
  docker logs strata-bench >"$R2/logs/$1.log" 2>&1
}
