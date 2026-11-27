#!/bin/bash
# Inkling 2 Mini 0.2.23 correctness + C1 / C32 + STRATA_INK_INTELARC (off/exact/fast) + DFlash 2 gate on Arc Pro B70 (SYCL).
#   b70_gate.sh <label> <model> <exe> <tokdir> <py> <repo-root> <out-dir>
# Note: DSpark is retired; DFlash 2 (dflash2.safetensors beside the model) is the sole drafter.
# Prints one summary line per step; exit status 1 when a check failed (the later steps still run).
set -u
L=$1; MODEL=$2; EXE=$3; TOK=$4; PY=$5; ROOT=$6; O=$7
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$O"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export STRATA_INK_DRAFTER=dflash2
bad=0
echo "== $(date +%T) gate $L: $MODEL"
echo "free RAM $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) GiB"

# 1. Selfcheck (.strata only): OFF, EXACT, FAST
if [ "${MODEL##*.}" = "strata" ]; then
  for mode in off exact fast; do
    if [ "$mode" = "off" ]; then
      env UR_LOADER_USE_LEVEL_ZERO_V2=0 STRATA_INK_INTELARC=0 STRATA_INK_INTELARC_EXACT=0 \
        timeout 900 "$EXE" --selfcheck --native "$MODEL" --gpus 1 --max-context 4096 > "$O/selfcheck-$mode.log" 2>&1
    elif [ "$mode" = "exact" ]; then
      env -u UR_LOADER_USE_LEVEL_ZERO_V2 STRATA_INK_INTELARC=1 STRATA_INK_INTELARC_EXACT=1 \
        timeout 900 "$EXE" --selfcheck --native "$MODEL" --gpus 1 --max-context 4096 > "$O/selfcheck-$mode.log" 2>&1
    else
      env -u UR_LOADER_USE_LEVEL_ZERO_V2 STRATA_INK_INTELARC=1 STRATA_INK_INTELARC_EXACT=0 \
        timeout 900 "$EXE" --selfcheck --native "$MODEL" --gpus 1 --max-context 4096 > "$O/selfcheck-$mode.log" 2>&1
    fi
    rc=$?; echo "selfcheck[$mode] rc=$rc $(grep -E 'selfcheck|SELFCHECK|PASS|FAIL' "$O/selfcheck-$mode.log" | head -2 | cut -c1-200)"; [ $rc -eq 0 ] || bad=1
  done
fi

# 2. STRATA_INK_INTELARC (off / exact / fast) bench + EXACT == OFF greedy identity + KLD
I64=$(seq -s, 101 164)
env UR_LOADER_USE_LEVEL_ZERO_V2=0 STRATA_INK_INTELARC=0 STRATA_INK_INTELARC_EXACT=0 \
  timeout 1800 "$EXE" --native "$MODEL" --gpus 1 --max-context 4096 --draft off --prompt-ids "$I64" --gen 16 --dump-ref "$O/ref-off.bin" > "$O/p64-off.log" 2>&1
rc_off=$?; sha_off=$(sha256sum "$O/ref-off.bin" 2>/dev/null | cut -c1-16)
# --dump-ref only writes the logits file; the greedy ids of the OFF run come from a second, generating run (the first B70 run compared the dump-ref message with the EXACT ids)
env UR_LOADER_USE_LEVEL_ZERO_V2=0 STRATA_INK_INTELARC=0 STRATA_INK_INTELARC_EXACT=0 \
  timeout 1800 "$EXE" --native "$MODEL" --gpus 1 --max-context 4096 --draft off --prompt-ids "$I64" --gen 16 > "$O/p64-off-gen.log" 2>&1
ids_off=$(tail -n 1 "$O/p64-off-gen.log")
echo "p64[off]   rc=$rc_off ids=$(echo "$ids_off" | cut -c1-70) sha=$sha_off"; [ $rc_off -eq 0 ] || bad=1

env -u UR_LOADER_USE_LEVEL_ZERO_V2 STRATA_INK_INTELARC=1 STRATA_INK_INTELARC_EXACT=1 \
  timeout 1800 "$EXE" --native "$MODEL" --gpus 1 --max-context 4096 --draft off --prompt-ids "$I64" --gen 16 > "$O/p64-exact.log" 2>&1
rc_ex=$?; ids_ex=$(tail -n 1 "$O/p64-exact.log")
echo "p64[exact] rc=$rc_ex ids=$(echo "$ids_ex" | cut -c1-70)"; [ $rc_ex -eq 0 ] || bad=1
if [ "$ids_off" != "$ids_ex" ]; then
  echo "FAIL: EXACT greedy tokens diverged from OFF! off='$ids_off' exact='$ids_ex'"
  bad=1
else
  echo "EXACT==OFF greedy parity: PASS"
fi

if [ -f "$O/ref-off.bin" ]; then
  for mode in exact fast; do
    ex_flag=0; [ "$mode" = "exact" ] && ex_flag=1
    env -u UR_LOADER_USE_LEVEL_ZERO_V2 STRATA_INK_INTELARC=1 STRATA_INK_INTELARC_EXACT=$ex_flag \
      timeout 1800 "$EXE" --kld-eval "$O/ref-off.bin" --native "$MODEL" --gpus 1 --max-context 4096 --draft off > "$O/kld-$mode.log" 2>&1
    rc=$?; echo "kld[$mode] rc=$rc $(grep -E 'KLD|kld|top1|Top-1|mean' "$O/kld-$mode.log" | tail -2 | tr '\n' ' ' | cut -c1-200)"; [ $rc -eq 0 ] || bad=1
  done
  rm -f "$O/ref-off.bin"
fi

# 3. 5-run deterministic logits fingerprint in STRATA_INK_INTELARC=1 FAST mode
shas=""
for n in 1 2 3 4 5; do
  env -u UR_LOADER_USE_LEVEL_ZERO_V2 STRATA_INK_INTELARC=1 STRATA_INK_INTELARC_EXACT=0 \
    timeout 1800 "$EXE" --native "$MODEL" --gpus 1 --max-context 4096 --draft off --prompt-ids "$I64" --gen 16 --dump-ref "$O/l64-$n.bin" > "$O/p64-$n.log" 2>&1
  s=$(sha256sum "$O/l64-$n.bin" 2>/dev/null | cut -c1-16); shas="$shas $s"
  echo "p64[fast] run $n rc=$? ids=$(tail -n 1 "$O/p64-$n.log" | cut -c1-70) sha=$s"
  rm -f "$O/l64-$n.bin"
done
nd=$(echo $shas | tr ' ' '\n' | sort -u | wc -l)
echo "p64[fast] distinct logits hashes in 5 runs: $nd"; [ "$nd" = "1" ] || bad=1

# 4. Bench matrix: off / exact / fast (C1 plain + C1 dflash2 + C32 fast)
for mode in off exact fast; do
  for draft in off dflash2; do
    if [ "$mode" = "off" ]; then
      env UR_LOADER_USE_LEVEL_ZERO_V2=0 STRATA_INK_INTELARC=0 STRATA_INK_INTELARC_EXACT=0 \
        timeout 3000 "$EXE" --bench --native "$MODEL" --gpus 1 --parallel 1 --max-context 4096 --prompt-len 512 --gen 64 --draft "$draft" > "$O/bench-c1-$mode-$draft.log" 2>&1
    elif [ "$mode" = "exact" ]; then
      env -u UR_LOADER_USE_LEVEL_ZERO_V2 STRATA_INK_INTELARC=1 STRATA_INK_INTELARC_EXACT=1 \
        timeout 3000 "$EXE" --bench --native "$MODEL" --gpus 1 --parallel 1 --max-context 4096 --prompt-len 512 --gen 64 --draft "$draft" > "$O/bench-c1-$mode-$draft.log" 2>&1
    else
      env -u UR_LOADER_USE_LEVEL_ZERO_V2 STRATA_INK_INTELARC=1 STRATA_INK_INTELARC_EXACT=0 \
        timeout 3000 "$EXE" --bench --native "$MODEL" --gpus 1 --parallel 1 --max-context 4096 --prompt-len 512 --gen 64 --draft "$draft" > "$O/bench-c1-$mode-$draft.log" 2>&1
    fi
    rc=$?
    echo "bench C1[$mode/$draft] rc=$rc $(grep -E '^\[bench\]' "$O/bench-c1-$mode-$draft.log" | cut -c1-240)"
    if [ $rc -ne 0 ]; then bad=1; grep -E "init:|sycl:|dalloc|out of device" "$O/bench-c1-$mode-$draft.log" | head -3 | cut -c1-400; fi
  done
done
env -u UR_LOADER_USE_LEVEL_ZERO_V2 STRATA_INK_INTELARC=1 STRATA_INK_INTELARC_EXACT=0 \
  timeout 3000 "$EXE" --bench --native "$MODEL" --gpus 1 --parallel 32 --max-context 4096 --prompt-len 512 --gen 64 --draft off > "$O/bench-c32-fast.log" 2>&1
rc=$?; echo "bench C32[fast] rc=$rc $(grep -E '^\[bench\]' "$O/bench-c32-fast.log" | cut -c1-240)"
if [ $rc -ne 0 ]; then bad=1; grep -E "init:|sycl:|dalloc|out of device" "$O/bench-c32-fast.log" | head -3 | cut -c1-400; fi
grep -E "loaded|runner ready|VRAM experts|intelarc" "$O/bench-c1-fast-dflash2.log" | head -5 | cut -c1-300

# 5. Live server checks (chat + 4K/32K needles + C32) under STRATA_INK_INTELARC=1 with DFlash 2
env -u UR_LOADER_USE_LEVEL_ZERO_V2 STRATA_INK_INTELARC=1 STRATA_INK_INTELARC_EXACT=0 \
  "$PY" "$HERE/serve_chat.py" --out "$O/chat" --src "$ROOT" --exe "$EXE" --tok "$TOK" --port 18781 --model-name "$L" --wait 1800 \
  --post "$HERE/longctx_chat.py {port} 4096" --post "$HERE/longctx_chat.py {port} 32000" -- --native "$MODEL" --max-context 40960 --draft dflash2
rc=$?; echo "chat+needles[dflash2] rc=$rc"; [ $rc -eq 0 ] || bad=1

env -u UR_LOADER_USE_LEVEL_ZERO_V2 STRATA_INK_INTELARC=1 STRATA_INK_INTELARC_EXACT=0 \
  "$PY" "$HERE/serve_chat.py" --out "$O/c32" --src "$ROOT" --exe "$EXE" --tok "$TOK" --port 18782 --model-name "$L" --wait 1800 --no-chat \
  --post "$HERE/c32_chat.py {port} 32 24" -- --native "$MODEL" --max-context 4096 --parallel 32 --draft off
rc=$?; echo "c32 server rc=$rc"; [ $rc -eq 0 ] || bad=1

echo "== $(date +%T) gate $L done, failures: $bad"
exit $bad
