#!/bin/bash
# The standard benchmark again with nothing else on the machine: a compile overlapped the earlier DFlash2 run.
set -uo pipefail
root=/tmp/strata-matrix2
mv "$root/out/std-dflash2" "$root/out/contaminated-std-dflash2"; mv "$root/out/std-dflash2.log" "$root/out/contaminated-std-dflash2.log"; mv "$root/out/std-dflash2.engine.log" "$root/out/contaminated-std-dflash2.engine.log"
mv "$root/out/std-mtp" "$root/out/first-std-mtp"; mv "$root/out/std-mtp.log" "$root/out/first-std-mtp.log"; mv "$root/out/std-mtp.engine.log" "$root/out/first-std-mtp.engine.log"
. "$root/lib.sh"
image=local/strata:0.2.9-cuda12-dflash2
docker stop -t 120 strata
uptime
for drafter in mtp dflash2; do
  echo "=== standard $drafter $(date -u +%FT%TZ)"
  mkdir -p "$root/out/std-$drafter" && chmod 777 "$root/out/std-$drafter"
  sed "s#/out/orig.current.engine.log#/out/std-$drafter.engine.log#" "$root/cfg/orig.json" > "$root/cfg/current.json"
  timeout 2400 docker run --rm --name strata-matrix --init --user 1000:1000 --gpus all --network none --ulimit memlock=-1 --shm-size 2g \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HOME=/data/home \
    -e DRAFTING="$drafter" -e STRATA_API_KEY="$key" \
    -v /models/models:/models/models:ro -v /models/packs:/models/packs:ro -v "$root/cfg:/cfg:ro" -v "$root/out:/out" -v /models/strata-docker/data:/data \
    -w "/out/std-$drafter" --entrypoint /opt/strata/terminal/bin/strata "$image" bench run --profile /cfg/current.json --track custom \
    --track-notes "branch dflash2 merged with main, DRAFTING=$drafter, parallel 2, 16 GB conversation cache" --background "none" > "$root/out/std-$drafter.log" 2>&1
  echo "standard $drafter rc=$? $(tail -1 "$root/out/std-$drafter.log" | cut -c1-200)"
  uptime
done
