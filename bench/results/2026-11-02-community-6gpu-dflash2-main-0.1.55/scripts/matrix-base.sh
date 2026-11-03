#!/bin/bash
# Four configurations, each loaded fresh, in an order and its reverse: Swift or original IQ3_S, MTP or DFlash2. Same image, flags and files on disk.
# Then the control without saved rings, and the draft standard benchmark on the original weights. Production comes back at the end.
set -uo pipefail
root=/tmp/strata-matrix
image=local/strata:0.2.8-cuda12-dflash2-bench
key=$(python3 -c "import secrets;print(secrets.token_hex(24))")
mkdir -p "$root/out" "$root/cfg" && chmod 777 "$root/out"
restore() {
  docker rm -f strata-matrix >/dev/null 2>&1 || true
  [ -n "${SAMPLER:-}" ] && kill "$SAMPLER" 2>/dev/null
  docker start strata
  echo RESTORED "$(date)"
}
trap restore EXIT
trap 'exit 130' INT TERM HUP
d=/opt/strata/components/engine/0.1.54-linux-x86_64-cuda-cuda12
python3 - "$d" "$root" <<'PY'
import json, sys
d, root = sys.argv[1:]
base = json.load(open('/models/strata-docker/config/swift-iq3-agent-v028.json'))
M = '/models/models/Qwen/3.8FlashNext/'
models = {
    'swift': ('/models/packs/swift-iq3_s', M + 'Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf', M + 'Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf', 'Swift-IQ3-S'),
    'orig': ('/models/packs/iq3_s_orig', M + 'Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf', M + 'Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf', 'qwen3.8-flash-next-iq3_s'),
}
for name, (pack, native, ple, model_name) in models.items():
    c = json.loads(json.dumps(base))
    a = c['args']
    a[a.index('--pack') + 1], a[a.index('--native') + 1], a[a.index('--ple-gguf') + 1] = pack, native, ple
    c['args'] = [x for x in a if x != '--vision']
    c.pop('vision', None)
    c.update(tokenizer=pack + '/tokenizer', model_name=model_name, host='127.0.0.1', port=8181, parallel=2, exe=d + '/strata', lib_dirs=[d], log=f'/out/{name}.current.engine.log', drafting='mtp')
    json.dump(c, open(f'{root}/cfg/{name}.json', 'w'), indent=1)
PY
start() { # run-name model drafter [extra docker args]
  local run=$1 model=$2 drafter=$3; shift 3
  sed "s#/out/$model.current.engine.log#/out/$run.engine.log#" "$root/cfg/$model.json" > "$root/cfg/current.json"
  docker run -d --name strata-matrix --init --user 1000:1000 --gpus all --network none --ulimit memlock=-1 --shm-size 2g \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HOME=/data/home \
    -e DRAFTING="$drafter" -e STRATA_API_KEY="$key" -e BENCH_KEY="$key" "$@" \
    -v /models/models:/models/models:ro -v /models/packs:/models/packs:ro -v "$root/cfg:/cfg:ro" -v "$root/client:/client:ro" -v "$root/out:/out" \
    -v /models/strata-docker/data:/data --entrypoint /opt/strata/terminal/bin/strata "$image" serve --profile /cfg/current.json --output jsonl >/dev/null
}
wait_ready() {
  for attempt in $(seq 1 120); do
    if docker exec -e K="$key" strata-matrix /opt/strata/terminal/python/bin/python3 -c 'import json,os,urllib.request; h=json.load(urllib.request.urlopen("http://127.0.0.1:8181/health",timeout=5)); assert h["loaded"]' >/dev/null 2>&1; then return 0; fi
    [ "$(docker inspect -f '{{.State.Running}}' strata-matrix)" = true ] || return 1
    sleep 5
  done
  return 1
}
one() { # run-name model drafter sections [extra docker args]
  local run=$1 model=$2 drafter=$3 sections=$4; shift 4
  echo "=== $run $(date -u +%FT%TZ)"
  local t0=$(date +%s)
  start "$run" "$model" "$drafter" "$@"
  if ! wait_ready; then
    echo "FAILED TO LOAD $run"
    docker logs strata-matrix 2>&1 | tail -30 | cut -c1-400
    docker rm -f strata-matrix >/dev/null
    return
  fi
  echo "loaded in $(( $(date +%s) - t0 )) s"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu,clocks.sm,power.draw --format=csv,noheader > "$root/out/$run.vram.csv"
  ( while :; do nvidia-smi --query-gpu=timestamp,index,temperature.gpu,utilization.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader; sleep 30; done > "$root/out/$run.gpu.csv" 2>/dev/null ) &
  SAMPLER=$!
  timeout 2400 docker exec -e BENCH_KEY="$key" strata-matrix /opt/strata/terminal/python/bin/python3 -I /client/matrix_client.py "$sections" "/out/$run.jsonl" > "$root/out/$run.client.log" 2>&1
  echo "client rc=$? $(tail -1 "$root/out/$run.client.log" | cut -c1-200)"
  kill "$SAMPLER" 2>/dev/null; SAMPLER=
  docker logs strata-matrix > "$root/out/$run.server.log" 2>&1
  [ "$(docker inspect -f '{{.State.Running}}' strata-matrix)" = true ] || echo "CONTAINER DIED $run"
  docker stop -t 90 strata-matrix >/dev/null
  docker rm strata-matrix >/dev/null
  echo "done in $(( $(date +%s) - t0 )) s"
}
docker stop -t 120 strata
ALL=prefill,decode,ring,session
one r1-swift-mtp     swift mtp     "$ALL"
one r1-swift-dflash2 swift dflash2 "$ALL"
one r1-orig-mtp      orig  mtp     "$ALL"
one r1-orig-dflash2  orig  dflash2 "$ALL"
one r2-orig-dflash2  orig  dflash2 "$ALL"
one r2-orig-mtp      orig  mtp     "$ALL"
one r2-swift-dflash2 swift dflash2 "$ALL"
one r2-swift-mtp     swift mtp     "$ALL"
one c1-swift-dflash2-noring swift dflash2 ring,session -e STRATA_DF2_RING_CACHE=0
one c1-orig-dflash2-noring  orig  dflash2 ring,session -e STRATA_DF2_RING_CACHE=0
# the draft standard benchmark, original weights only (its profile); best effort, its folders say what they are
for drafter in mtp dflash2; do
  echo "=== standard $drafter $(date -u +%FT%TZ)"
  mkdir -p "$root/out/std-$drafter" && chmod 777 "$root/out/std-$drafter"
  sed "s#/out/orig.current.engine.log#/out/std-$drafter.engine.log#" "$root/cfg/orig.json" > "$root/cfg/current.json"
  timeout 2400 docker run --rm --name strata-matrix --init --user 1000:1000 --gpus all --network none --ulimit memlock=-1 --shm-size 2g \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HOME=/data/home \
    -e DRAFTING="$drafter" -e STRATA_API_KEY="$key" \
    -v /models/models:/models/models:ro -v /models/packs:/models/packs:ro -v "$root/cfg:/cfg:ro" -v "$root/out:/out" -v /models/strata-docker/data:/data \
    -w "/out/std-$drafter" --entrypoint /opt/strata/terminal/bin/strata "$image" bench run --profile /cfg/current.json --track custom \
    --track-notes "branch dflash2, DRAFTING=$drafter, parallel 2, 16 GB conversation cache" --background "none" > "$root/out/std-$drafter.log" 2>&1
  echo "standard $drafter rc=$? $(tail -1 "$root/out/std-$drafter.log" | cut -c1-200)"
done
