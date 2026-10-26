#!/bin/bash
# gap.sh <arms-file>: stops the local serve process (binary name in untracked gap/.server), runs each
# measurement arm, ALWAYS restores it (trap on any exit).
# arm line: name|binary|extra-env|tokens-file|max-new|ctx|extra-args   (# lines ignored)
# The window must stay short (<~10 min): that server hosts this agent's own model while stopped = no inference.
set -u
FORK=/Users/pc/src/Strata-fork
M=/Users/pc/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS
G1=$M/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
G2=$M/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
PACK=/Users/pc/src/Strata-data/packs/iq3_xxs
MTP=/Users/pc/src/Strata-data/mtp/rt
ARMS=${1:?usage: gap.sh arms-file}
SRV=$(cat "$FORK/gap/.server" 2>/dev/null) || true
[[ -z ${SRV:-} ]] && { echo "put the local serve binary name in $FORK/gap/.server" >&2; exit 1; }
SRV_BIN=$(command -v "$SRV") || { echo "serve binary '$SRV' not on PATH" >&2; exit 1; }
OUT=$FORK/gap/results/$(date +%m%d-%H%M); mkdir -p "$OUT"
SUM=$OUT/summary.txt
RESTORED=0
trap 'restore' EXIT INT TERM

stop_server() {
  [[ ${GAP_NO_SERVER:-0} == 1 ]] && { echo "server left alone (GAP_NO_SERVER=1)"; return; }
  local pid i
  pid=$(pgrep -x "$SRV" | head -1) || true
  [[ -z $pid ]] && { echo "server not running at start"; return; }
  ps -o command= -p "$pid" | sed "s|^$SRV_BIN ||; s|^$SRV ||" > "$OUT/server-argv.txt"
  kill "$pid"
  for i in {1..30}; do pgrep -x "$SRV" >/dev/null || break; sleep 1; done
  pgrep -x "$SRV" >/dev/null && pkill -9 -x "$SRV"
  echo "server stopped $(date +%T)"
}

restore() {
  local i
  [[ $RESTORED == 1 ]] && return
  [[ ${GAP_NO_SERVER:-0} == 1 ]] && { RESTORED=1; return; }
  if ! pgrep -x "$SRV" >/dev/null; then
    nohup "$SRV_BIN" $(cat "$OUT/server-argv.txt") >"$OUT/server-restore.log" 2>&1 &
  fi
  for i in {1..150}; do
    sleep 2
    curl -s -m 3 http://127.0.0.1:12345/v1/models 2>/dev/null | grep -q '"state":"ready"' && {
      RESTORED=1; echo "server restored after ~$((i*2))s $(date +%T)"; return; }
  done
  echo "SERVER NOT RESTORED after 300s - check $OUT/server-restore.log" >&2
}

run() { # name bin env tok mn ctx extra
  local name=$1 bin=$2 envs=$3 tok=$4 mn=$5 ctx=$6 extra=$7
  echo "=== $name $(date +%T)" | tee -a "$SUM"
  local fc=$(sysctl -n vm.vm_page_filecache_min) fpg=$(vm_stat | awk '/Pages free/{gsub(/\./,"",$3); print $3}')
  if [[ $bin == *dispatch_lat || $bin == *alu_bench ]]; then
    "$bin" > "$OUT/$name.txt" 2>&1
  else
    env STRATA_METAL_STREAM=1 STRATA_METAL_PROMPT_CHUNK=4096 STRATA_METAL_LEAVE_SHARE=0.35 STRATA_METAL_ARENA_GB=33 $envs \
      "$bin" --pack "$PACK" --native "$G1" --ple-gguf "$G2" --expert-profile "$FORK/data/expert-profile.bin" \
      --expert-cache 64 --prefill auto --spec 2 --suffix-draft 0 --mtp "$MTP" \
      --max-context "$ctx" --mmap-experts --tokens-file "$FORK/gap/tokens/$tok" --max-new "$mn" $extra \
      > "$OUT/$name.txt" 2>&1
  fi
  echo "arm-start fc_min_pages $fc free_pages $fpg" >> "$OUT/$name.txt"
}

stop_server
while IFS='|' read -r name bin envs tok mn ctx extra; do
  name=${name#"${name%%[![:space:]]*}"}
  [[ -z $name || $name == \#* ]] && continue
  run "$name" "$bin" "$envs" "$tok" "$mn" "$ctx" "$extra"
done < "$ARMS"
restore
echo "done: $OUT"
