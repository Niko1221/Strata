#!/bin/bash
# The revision a PR would contain (merge of upstream main), against upstream main itself (plus the P100 fix these cards need).
# Builds both images, then measures. Production comes back at the end.
set -uo pipefail
M=cc75837cdea224683bf5507296dc7c3876358565
BASE=c62d5e3a15b40ea51c96aae9e8d36928965f53ab
branch_image=local/strata:0.2.9-cuda12-dflash2
main_image=local/strata:0.2.9-cuda12-main-p100fix
build() { # tag dir commit vision
  ( cd "$2" && docker build -t "$1" --build-arg STRATA_COMMIT="$3" --build-arg CUDA_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu24.04 \
      --build-arg "CUDA_ARCHITECTURES=60;75;86" --build-arg BUILD_VISION="$4" . ) > "/tmp/build-$(basename "$2").log" 2>&1
}
echo "build branch $(date -u +%T)"; build "$branch_image" /tmp/strata-dflash2-src "$M" 1 || { echo BUILD-FAILED branch; tail -5 /tmp/build-strata-dflash2-src.log | cut -c1-200; echo RESTORED-NOT-NEEDED; exit 1; }
echo "build main $(date -u +%T)";   build "$main_image"   /tmp/strata-main-src    "$BASE" 0 || { echo BUILD-FAILED main; tail -5 /tmp/build-strata-main-src.log | cut -c1-200; echo RESTORED-NOT-NEEDED; exit 1; }
for i in $main_image $branch_image; do docker run --rm --entrypoint /bin/sh "$i" -c 'ls /opt/strata/components/engine; env | grep -c DFLASH2_SERVE'; done

root=/tmp/strata-matrix2
mkdir -p "$root/client" "$root/cfg" "$root/out"; chmod 777 "$root/out"
cp /tmp/strata-matrix/client/matrix_client.py "$root/client/"; chmod -R a+rX "$root/client"
sed -n '1,/^docker stop -t 120 strata$/p' /tmp/strata-matrix/matrix.sh | sed '$d' \
  | sed -e 's#^root=.*#root=/tmp/strata-matrix2#' -e "s#^image=.*#image=$branch_image#" -e 's#0.1.54-linux#0.1.55-linux#' > "$root/lib.sh"
. "$root/lib.sh"
variant() { # new-name spec
  python3 - "$root/cfg/swift.json" "$root/cfg/$1.json" "$2" "$1" <<'PY'
import json, sys
src, dst, spec, name = sys.argv[1:]
c = json.load(open(src))
a = c['args']
a[a.index('--spec') + 1] = spec
c['log'] = f'/out/{name}.current.engine.log'
json.dump(c, open(dst, 'w'), indent=1)
PY
}
docker stop -t 120 strata
ALL=prefill,decode,ring,session
image=$main_image;   one b1-swift-mtp-main swift mtp "$ALL"
image=$branch_image
one r1-swift-mtp     swift mtp     "$ALL"
one r1-swift-dflash2 swift dflash2 "$ALL"
one r1-orig-mtp      orig  mtp     "$ALL"
one r1-orig-dflash2  orig  dflash2 "$ALL"
one r2-orig-dflash2  orig  dflash2 "$ALL"
one r2-orig-mtp      orig  mtp     "$ALL"
one r2-swift-dflash2 swift dflash2 "$ALL"
one r2-swift-mtp     swift mtp     "$ALL"
image=$main_image;   one b2-swift-mtp-main swift mtp "$ALL"
image=$branch_image
one c1-swift-dflash2-noring swift dflash2 ring,session -e STRATA_DF2_RING_CACHE=0
one c1-orig-dflash2-noring  orig  dflash2 ring,session -e STRATA_DF2_RING_CACHE=0
variant swifts6 6; variant swifts8 8
one s6-swift-mtp     swifts6 mtp     decode
one s6-swift-dflash2 swifts6 dflash2 decode
one s8-swift-dflash2 swifts8 dflash2 decode
one s8-swift-mtp     swifts8 mtp     decode
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
done
