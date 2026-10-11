#!/bin/bash
# Round 2: engine 0.1.43 (setup.py image) against 0.2.0-beta (engine 0.1.45.3) on the RTX 3080 Ti, context 262,144.
# Usage: run.sh PHASE...   phases: stopprod iq3s ud swift
. "$HOME/strata-trial/r2/lib.sh"
mkdir -p "$R2/out" "$R2/logs"

vol_path() {  # vol_path VOLUME GLOB -> host dir holding the first match
  docker run --rm -v "$1":/d public.ecr.aws/docker/library/alpine:3 sh -c "find /d -name '$2' | head -1" |
    sed "s#^/d#/var/lib/docker/volumes/$1/_data#; s#/[^/]*\$##"
}

pair() {  # pair TAG FAMILY MODEL_OLD MODEL_NEW GGUF_DIR VOLTAG
  local tag=$1 fam=$2 mo=$3 mn=$4 dir=$5 v=$6 i
  for i in 1 2; do
    start_new "b020-$v" "$fam" "$mn" "$dir" && wait_ready 10800 && snap "$tag-new-$i" && sweep "$tag-new" "sweep-$tag-new.jsonl"
    start_old "b0143-$v" "$fam" "$mo" "$dir" && wait_ready 10800 && snap "$tag-old-$i" && sweep "$tag-old" "sweep-$tag-old.jsonl"
    if [ "$tag" = iq3s ]; then
      start_new "b020-$v-sm86off" "$fam" "$mn" "$dir" -e STRATA_SET="STRATA_SM86_PERF=0" && wait_ready 3600 &&
        snap "$tag-sm86off-$i" && sweep "$tag-sm86off" "sweep-$tag-sm86off.jsonl"
    fi
  done
  stop_bench
}

tuned() {  # tuned TAG FAMILY MODEL GGUF_DIR VOLTAG
  log "tune $1"
  start_new "b020-$5-tuned" "$2" "$3" "$4" -e TUNE=quick && wait_ready 10800 && snap "$1-tuned" &&
    sweep "$1-tuned" "sweep-$1-tuned.jsonl"
  docker run --rm -v "b020-$5-tuned":/d public.ecr.aws/docker/library/alpine:3 sh -c \
    'find /d/state /d/install -maxdepth 4 \( -name "*tun*" -o -name "*.json" \) -newer /d/state -type f 2>/dev/null' \
    >"$R2/out/tuned-$1-files.txt" 2>&1
  stop_bench
}

for phase in "$@"; do
  case $phase in
    stopprod)
      log "stopping the production container strata (restart policy unless-stopped keeps it stopped)"
      docker stop -t 120 strata ;;
    iq3s)
      d=$PROD_MODELS/IQ3_S
      pair iq3s qwen IQ3_S IQ3_S "$d" iq3s
      log "needles iq3s-new"
      start_new b020-iq3s qwen IQ3_S "$d" && wait_ready 3600
      python3 "$CORPUS_DIR/tools/needle_bench.py" --url "$URL" --api-key "$STRATA_API_KEY" --lengths 32k,128k,262k \
        --depths 10,50,90 --out "$R2/out/needles-iq3s-new.json" >>"$R2/progress.log" 2>&1 || log "needle_bench failed"
      stop_bench
      tuned iq3s qwen IQ3_S "$d" iq3s ;;
    ud)
      log "first start ud (download)"
      start_new b020-ud unsloth UD-IQ4_XS - && wait_ready 14400 && snap ud-download
      stop_bench
      d=$(vol_path b020-ud '*UD-IQ4_XS-00001-of-*.gguf'); log "ud files in $d"
      pair ud unsloth UD-IQ4_XS UD-IQ4_XS "$d" ud
      tuned ud unsloth UD-IQ4_XS "$d" ud ;;
    swift)
      log "first start swift (download)"
      start_new b020-swift swift IQ3_S - && wait_ready 14400 && snap swift-download
      stop_bench
      d=$(vol_path b020-swift 'Swift-*IQ3_S-00001-of-*.gguf'); log "swift files in $d"
      pair swift swift IQ3_S IQ3_S "$d" swift ;;
  esac
done
log "done: $*"
