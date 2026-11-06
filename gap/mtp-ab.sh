#!/bin/bash
# MTP decode A/B on the Huihui UD-Q4_K_XL (host-stop window: this agent's own backend goes down).
# Arms interleave the baseline before and after the ladder (anchor drift). OUT set by the caller.
# arm line: --spec N [--mtp rt]; everything else matches the serve config (suffix-draft 1, chunk 2048, mpp, scan_v, arena 33).
set -u
FORK=/Users/pc/src/Strata-fork
PACK=/Users/pc/src/Strata-data/packs/ud-q4_k_xl-huihui
GGUF=/Users/pc/src/Strata-data/metal/ud-q4_k_xl-huihui-metal.gguf
PLE=/Users/pc/models/Huihui-Qwen3.8-Flash-Next-abliterated-v2-UD-Q4_K_XL-V2/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf
MTP=/Users/pc/src/Strata-data/mtp/rt
OUT=${OUT:?caller sets OUT}
SUM=$OUT/summary.txt

export STRATA_METAL_STREAM=1 STRATA_METAL_PROMPT_CHUNK=2048 STRATA_METAL_PROMPT_MOE=mpp \
       STRATA_METAL_GDN_SCAN=scan_v STRATA_METAL_ARENA_GB=33 STRATA_METAL_LEAVE_SHARE=0.35

arm() { # name extra-args...
  local name=$1; shift
  echo "=== $name $(date +%T)" | tee -a "$SUM"
  local fc fpg
  fc=$(sysctl -n vm.vm_page_filecache_min); fpg=$(vm_stat | awk '/Pages free/{gsub(/\./,"",$3); print $3}')
  "$FORK/build-metal/strata" --pack "$PACK" --native "$GGUF" --ple-gguf "$PLE" --native-dense-gguf "$GGUF" \
    --expert-profile "$FORK/data/expert-profile.bin" --expert-cache 64 --prefill auto --suffix-draft 1 \
    --max-context 40960 --mmap-experts --tokens-file "$FORK/gap/tokens/t357.txt" --max-new 192 \
    "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "arm-start fc_min_pages $fc free_pages $fpg rc $rc" >> "$OUT/$name.log"
  grep -E "generated|draft|mtp|KERR|error" "$OUT/$name.log" | tail -8 | tee -a "$SUM"
}

# save + stop the host (this agent's model); a trap below always brings it back
"$FORK/tools/huihui_session.sh" stop pre-mtp-ab 2>&1 | tee -a "$SUM"
trap '"$FORK/tools/huihui_session.sh" start pre-mtp-ab >> "$OUT/restore.log" 2>&1' EXIT

arm F_base  --spec 2
arm F_mtp2  --spec 2 --mtp "$MTP"
arm F_mtp3  --spec 3 --mtp "$MTP"
arm F_mtp4  --spec 4 --mtp "$MTP"
arm F_base2 --spec 2

trap - EXIT
"$FORK/tools/huihui_session.sh" start pre-mtp-ab 2>&1 | tee -a "$SUM"
echo "done $(date +%T)" | tee -a "$SUM"
