#!/bin/bash
# round 2: is the metal-refresh engine (HEAD+579: PR #3541 park-with-MTP + DFlash2) better for Huihui?
# arms: CLI decode t357@40K for refresh-noMTP / refresh-MTP(spec2,3) / refresh-MTP+dflash2,
#       then serve arms (refresh-MTP, and refresh-MTP while a second parked conversation turns) vs the
#       metal-live serve (current engine). OUT set by the caller.
set -u
FORK=/Users/pc/src/Strata-fork
PACK=/Users/pc/src/Strata-data/packs/ud-q4_k_xl-huihui
GGUF=/Users/pc/src/Strata-data/metal/ud-q4_k_xl-huihui-metal.gguf
PLE=/Users/pc/models/Huihui-Qwen3.8-Flash-Next-abliterated-v2-UD-Q4_K_XL-V2/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf
MTP=/Users/pc/src/Strata-data/mtp/rt
DF2=/Users/pc/src/Strata-data/dflash2/flash-next-df2.pack
R=/tmp/wtR/build-metal/strata
OUT=${OUT:?caller sets OUT}
SUM=$OUT/summary.txt
export STRATA_METAL_STREAM=1 STRATA_METAL_PROMPT_CHUNK=2048 STRATA_METAL_PROMPT_MOE=mpp \
       STRATA_METAL_GDN_SCAN=scan_v STRATA_METAL_ARENA_GB=33 STRATA_METAL_LEAVE_SHARE=0.35

cli() { # name bin extra-args...
  local name=$1 bin=$2; shift 2
  echo "=== $name $(date +%T)" | tee -a "$SUM"
  local fc fpg; fc=$(sysctl -n vm.vm_page_filecache_min); fpg=$(vm_stat | awk '/Pages free/{gsub(/\./,"",$3); print $3}')
  "$bin" --pack "$PACK" --native "$GGUF" --ple-gguf "$PLE" --native-dense-gguf "$GGUF" \
    --expert-profile "$FORK/data/expert-profile.bin" --expert-cache 64 --prefill auto --suffix-draft 1 \
    --max-context 40960 --mmap-experts --tokens-file "$FORK/gap/tokens/t357.txt" --max-new 192 \
    "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "arm-start fc_min_pages $fc free_pages $fpg rc $rc" >> "$OUT/$name.log"
  grep -E "decode |draft|mtp|dflash|speculation|KERR|error|off:" "$OUT/$name.log" | tail -8 | tee -a "$SUM"
}

serve_arm() { # name cfgfile
  local name=$1 cfg=$2
  echo "=== $name $(date +%T)" | tee -a "$SUM"
  pkill -f "serve/server.py.*ud-q4_k_xl-huihui" 2>/dev/null; sleep 2
  pkill -9 -f "build-metal/strata --serve.*ud-q4_k_xl-huihui" 2>/dev/null; sleep 3
  STRATA_HUIHUI_CFG="$cfg" nohup "$R" --serve --pack "$PACK" --native "$GGUF" --ple-gguf "$PLE" \
    --native-dense-gguf "$GGUF" --expert-profile "$FORK/data/expert-profile.bin" --expert-cache 64 \
    --prefill auto --spec 2 --suffix-draft 1 --mtp "$MTP" --max-context 204800 --mmap-experts \
    --conversation-cache-mib 8192 --conversation-cache-slots 6 --conversation-cache-min-tokens 2048 \
    --conversation-cache-min-free-mib 1024 --port 8180 --log "$OUT/$name.englog" \
    > "$OUT/$name.eng.out" 2>&1 &
  for _ in $(seq 1 60); do sleep 2; curl -sf -m 3 http://127.0.0.1:8180/v1/models 2>/dev/null | grep -q . && break; done
  curl -sf -m 3 http://127.0.0.1:8180/v1/models >/dev/null || { echo "engine up FAILED" | tee -a "$SUM"; return; }
  nohup "$FORK/.venv/bin/python" "$FORK/serve/server.py" --engine external --port 8180 --engine-port \
    8187 --config "$cfg" > "$OUT/$name.srv.log" 2>&1 &
  for _ in $(seq 1 30); do sleep 2; curl -sf -m 3 http://127.0.0.1:8180/v1/models 2>/dev/null | grep -q ud-q4_k_xl-huihui && break; done
  "$FORK/.venv/bin/python" "$FORK/gap/mtp-serve-probe.py" "$name" 2>&1 | tee -a "$SUM"
}

pkill -f "serve/server.py.*ud-q4_k_xl-huihui" 2>/dev/null
"$FORK/tools/huihui_session.sh" stop pre-mtp-ab 2>&1 | tee -a "$SUM"
trap '"$FORK/tools/huihui_session.sh" start pre-mtp-ab >> "$OUT/restore.log" 2>&1' EXIT

cli R_nmtp  "$R" --spec 2
cli R_mtp2  "$R" --spec 2 --mtp "$MTP"
cli R_mtp3  "$R" --spec 3 --mtp "$MTP"
cli R_df2   "$R" --spec 2 --mtp "$MTP" --draft dflash2 --draft-model "$DF2"

serve_arm S_refresh "$FORK/strata-ud-q4_k_xl-huihui.json"
# serve_arm S_park  "$FORK/strata-ud_q4k_test.json"   # parked-two-conversations probe, run by hand after
echo "done $(date +%T)" | tee -a "$SUM"
