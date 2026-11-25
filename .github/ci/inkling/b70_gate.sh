#!/bin/bash
# Inkling 2 Mini correctness + C1 / C32 gate for one model on the Arc Pro B70 runner (SYCL).
#   b70_gate.sh <label> <model> <exe> <tokdir> <py> <repo-root> <out-dir> [dspark-gguf]
# 1 selfcheck (a .strata only); 2 bench at --parallel 1 and 32 (prefill / decode tok/s); 3 five greedy `--draft off` 64-token runs (logits sha, the number of
# distinct hashes must be 1); 4 server: first chat (answer text checked), needle prompts at about 4K and 32K tokens; 5 server with --parallel 32 and 32 clients.
# Prints one summary line per step; exit status 1 when a check failed (the later steps still run).
set -u
L=$1; MODEL=$2; EXE=$3; TOK=$4; PY=$5; ROOT=$6; O=$7; DSP=${8:-}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$O"
DS=(); [ -n "$DSP" ] && DS=(--dspark "$DSP")
export UR_LOADER_USE_LEVEL_ZERO_V2=0 ONEAPI_DEVICE_SELECTOR=level_zero:gpu
bad=0
echo "== $(date +%T) gate $L: $MODEL"
echo "free RAM $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) GiB"

if [ "${MODEL##*.}" = "strata" ]; then
  timeout 900 "$EXE" --selfcheck --native "$MODEL" --gpus 1 --max-context 4096 > "$O/selfcheck.log" 2>&1
  rc=$?; echo "selfcheck rc=$rc $(grep -E 'selfcheck|SELFCHECK|PASS|FAIL' "$O/selfcheck.log" | head -2 | cut -c1-200)"; [ $rc -eq 0 ] || bad=1
fi

for par in 1 32; do
  timeout 3000 "$EXE" --bench --native "$MODEL" ${DS[@]+"${DS[@]}"} --gpus 1 --parallel $par --max-context 4096 --prompt-len 512 --gen 64 > "$O/bench-c$par.log" 2>&1
  rc=$?
  echo "bench C$par rc=$rc $(grep -E '^\[bench\]' "$O/bench-c$par.log" | cut -c1-240)"
  if [ $rc -ne 0 ]; then bad=1; grep -E "init:|sycl:|dalloc|out of device" "$O/bench-c$par.log" | head -3 | cut -c1-400; fi
done
grep -E "loaded|runner ready|VRAM experts" "$O/bench-c1.log" | head -3 | cut -c1-300

I64=$(seq -s, 101 164); shas=""
for n in 1 2 3 4 5; do
  timeout 1800 "$EXE" --native "$MODEL" ${DS[@]+"${DS[@]}"} --gpus 1 --max-context 4096 --draft off --prompt-ids "$I64" --gen 16 --dump-ref "$O/l64-$n.bin" > "$O/p64-$n.log" 2>&1
  s=$(sha256sum "$O/l64-$n.bin" 2>/dev/null | cut -c1-16); shas="$shas $s"
  echo "p64 run $n rc=$? ids=$(tail -n 1 "$O/p64-$n.log" | cut -c1-70) sha=$s"
  rm -f "$O/l64-$n.bin"
done
nd=$(echo $shas | tr ' ' '\n' | sort -u | wc -l)
echo "p64 distinct logits hashes in 5 runs: $nd"; [ "$nd" = "1" ] || bad=1

"$PY" "$HERE/serve_chat.py" --out "$O/chat" --src "$ROOT" --exe "$EXE" --tok "$TOK" --port 18781 --model-name "$L" --wait 1800 \
  --post "$HERE/longctx_chat.py {port} 4096" --post "$HERE/longctx_chat.py {port} 32000" -- --native "$MODEL" ${DS[@]+"${DS[@]}"} --max-context 40960 --draft dspark
rc=$?; echo "chat+needles rc=$rc"; [ $rc -eq 0 ] || bad=1
"$PY" "$HERE/serve_chat.py" --out "$O/c32" --src "$ROOT" --exe "$EXE" --tok "$TOK" --port 18782 --model-name "$L" --wait 1800 --no-chat \
  --post "$HERE/c32_chat.py {port} 32 24" -- --native "$MODEL" ${DS[@]+"${DS[@]}"} --max-context 4096 --parallel 32 --draft dspark
rc=$?; echo "c32 server rc=$rc"; [ $rc -eq 0 ] || bad=1
echo "== $(date +%T) gate $L done, failures: $bad"
exit $bad
