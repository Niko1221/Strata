#!/bin/bash
# run_dq_test.sh <dq_test binary> [device] [extra dq_test args, e.g. --quick]
# One process per (type, kernel): a kernel that kills the device context (shared-memory overflow on NVIDIA) ends only its own case.
# Prints every RESULT / FAIL / note line, and "CRASH" with the exit code when a case dies; ends with a table of the RESULT lines.
BIN=${1:?binary}; DEV=${2:-0}; shift 2 2>/dev/null || shift $#
export DS4_LAUNCH_CHECK=1
TYPES="Q6_K Q5_K Q4_K Q3_K Q2_K IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_XS IQ1_S IQ1_M IQ4_NL Q8_0 BF16"
ALL=$(mktemp)
echo "binary $BIN device $DEV args: $*"
for t in $TYPES; do
  for k in new old default; do
    for P in rows gguf,gemm; do   # (rows alone: a sticky device error in it must not hide the gg::rows and GEMM cases)
      echo "=== $t kernel=$k path=$P"
      timeout 1500 "$BIN" --device "$DEV" --type "$t" --kernel "$k" --path "$P" "$@" > "$ALL.one" 2>&1
      rc=$?
      grep -E "^(DEVICE|RESULT|  FAIL|  first|  note|STICKY|DEVICE ERROR|SUMMARY)" "$ALL.one" | head -60
      [ $rc -ge 2 ] && echo "CRASH $t kernel=$k exit=$rc: $(tail -n 3 "$ALL.one" | tr '
' '|')"
      grep -E "^RESULT" "$ALL.one" >> "$ALL"
    done
  done
  echo "=== $t mv/f16"
  timeout 1500 "$BIN" --device "$DEV" --type "$t" --kernel default --path mv,f16 "$@" > "$ALL.one" 2>&1
  rc=$?
  grep -E "^(RESULT|  FAIL|  first|  note|STICKY|DEVICE ERROR|SUMMARY)" "$ALL.one" | head -60
  [ $rc -ge 2 ] && echo "CRASH $t mv/f16 exit=$rc: $(tail -n 3 "$ALL.one" | tr '\n' '|')"
  grep -E "^RESULT" "$ALL.one" >> "$ALL"
done
echo "================ TABLE"
sort "$ALL"
rm -f "$ALL" "$ALL.one"
echo "ALLDONE"
