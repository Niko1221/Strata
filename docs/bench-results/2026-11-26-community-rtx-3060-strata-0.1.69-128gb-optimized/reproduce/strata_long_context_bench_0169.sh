#!/usr/bin/env bash
# Custom NON-STANDARD Strata long-context extension,
# using exact THINKING xhigh and sampling settings from standard draft.1.
# Usage: bash strata_long_context_bench.sh 32k|64k|128k|262k
# Run AFTER a fresh official benchmark of the SAME configured context.
# GPU Unlock remains ON. No grub/driver/kernel modifications.
set -Eeuo pipefail

CTX="${1:-}"
case "$CTX" in
  32k) TOKENS=32000; MAXCTX=32768 ;;
  64k) TOKENS=64000; MAXCTX=65536 ;;
  128k) TOKENS=128000; MAXCTX=131072 ;;
  262k) TOKENS=260000; MAXCTX=262144 ;;
  *) echo "Usage: bash $0 {32k|64k|128k|262k}" >&2; exit 2 ;;
esac

export PATH="$HOME/.local/bin:$PATH"
BASE="$HOME/strata-test/community-bench"
SCRIPT="$HOME/strata-test/strata_long_context_bench.py"
URL="http://127.0.0.1:18080"

for binary in strata python3 curl nvidia-smi; do
  command -v "$binary" >/dev/null || { echo "STOP: missing $binary" >&2; exit 3; }
done
test -s "$SCRIPT" || { echo "STOP: missing $SCRIPT" >&2; exit 3; }
python3 -m py_compile "$SCRIPT"

# Strata's Terminal distribution includes its own exact BPE tokenizer and
# standalone Python runtime (including the regex dependency).
# Search local user installation only; no download and no installation.
echo "===== LOCATE OFFICIAL LOCAL TOKENIZER ====="
BUNDLED_SOURCE="$(
  printf "%s\n" "$HOME/.local/share/strata/terminal/0.1.63/tools/strata_tokenizer.py"
)"
if [[ -z "$BUNDLED_SOURCE" ]]; then
  echo "STOP: bundled Strata tokenizer not found under ~/.local" >&2
  echo "Try: find ~/.local -name strata_tokenizer.py -print" >&2
  exit 3
fi
BUNDLE_ROOT="$(dirname "$(dirname "$BUNDLED_SOURCE")")"
BUNDLED_PYTHON="$BUNDLE_ROOT/python/bin/python3"
if [[ ! -x "$BUNDLED_PYTHON" ]]; then
  if python3 -c 'import regex' 2>/dev/null; then
    BUNDLED_PYTHON="$(command -v python3)"
  else
    echo "STOP: Strata bundled Python not found and system Python lacks regex" >&2
    echo "Tokenizer source: $BUNDLED_SOURCE" >&2
    exit 3
  fi
fi

mapfile -t TOKENIZER_PACKS < <(
  find "$HOME/strata-data/packs" -maxdepth 8 -type f \
    -path '*/tokenizer/vocab.json' -print 2>/dev/null \
  | grep -i 'q2_0' || true
)
if (( ${#TOKENIZER_PACKS[@]} != 1 )); then
  echo "STOP: expected exactly one Q2_0 prepared tokenizer pack; found ${#TOKENIZER_PACKS[@]}" >&2
  printf 'Candidates: %s\n' "${TOKENIZER_PACKS[@]}" >&2
  echo "Available: find ~/strata-data -name vocab.json -print" >&2
  exit 3
fi
TOKENIZER_DIR="$(dirname "${TOKENIZER_PACKS[0]}")"
export STRATA_TOKENIZER_SOURCE="$BUNDLED_SOURCE"
export STRATA_TOKENIZER_DIR="$TOKENIZER_DIR"
echo "Tokenizer source: $BUNDLED_SOURCE"
echo "Tokenizer pack:   $TOKENIZER_DIR"
echo "Tokenizer Python: $BUNDLED_PYTHON"

"$BUNDLED_PYTHON" -c '
import importlib.util, os, regex
p=os.environ["STRATA_TOKENIZER_SOURCE"]
s=importlib.util.spec_from_file_location("strata_tokenizer",p)
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)
print("LOCAL TOKENIZER IMPORT: OK")
'

sudo -v
UNLOCK_LOG="$(sudo journalctl -k -b --no-pager | grep -F 'gpu_unlock: unlocked' || true)"
if [[ -z "$UNLOCK_LOG" ]]; then
  echo "STOP: GPU Unlock must be active; no OFF tests" >&2
  exit 3
fi

echo "===== MODEL / CONTEXT / DRAFTING CHECK ====="
curl -fsS "$URL/metrics" |
python3 -c '
import json, sys
d = (json.load(sys.stdin).get("engine") or {})
want = int(sys.argv[1])
assert d.get("version") == "0.1.69", "STOP: unexpected engine version"
assert d.get("max_context") == want, "STOP: expected max_context %d; actual %r" % (want, d.get("max_context"))
assert d.get("model") == "qwen3.8-flash-next-q2_0", "STOP: wrong model"
assert d.get("drafting") == "off", "STOP: drafting must be off"
print("VERIFIED context=%s drafting=%s model=%s" % (want,d["drafting"],d["model"]))
' "$MAXCTX"

echo "===== AVAILABLE MEMORY ====="
free -h
# /proc/meminfo is keyed and reliable on Ubuntu.
AVAILABLE_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
if (( AVAILABLE_KIB < 6 * 1024 * 1024 )); then
  echo "STOP: under 6 GiB memory available before long-context benchmark" >&2
  exit 3
fi

mkdir -p "$BASE"
RUN="$BASE/$(date -u +%Y%m%dT%H%M%SZ)-${CTX}-engine-0169-long-context-thinking-xhigh-unlock-on"
[[ ! -e "$RUN" ]] || { echo "STOP: experiment directory exists: $RUN" >&2; exit 3; }
mkdir -p "$RUN/evidence"

cat > "$RUN/README.txt" <<EOF
Strata non-standard synthetic long-context experiment
UTC: $(date -u --iso-8601=seconds)
GPU: NVIDIA RTX 3060 12 GiB (Minisforum DEG1 OCuLink), GPU Unlock ON
Model: Qwen3.8-Flash-Next Q2_0 (ISTA-DASLab)
Context configured: $MAXCTX
Nominal input target: $TOKENS tokens
Speed runs: 3 repetitions, output cap 512, needle depth 50 percent
Recall runs: needle depth 10 and 90 percent, output cap 2048
Effective recall input can be shorter than nominal target to reserve generation tokens.
Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0,
  presence_penalty 0.0, repetition_penalty 1.0,
  reasoning_effort xhigh, reasoning_budget_tokens 0,
  chat_template_kwargs.enable_thinking true,
  chat_template_kwargs.preserve_thinking true.
Drafting OFF; GPU Unlock ON. One request at a time.
IMPORTANT: Same sampling as strata-qwen38-thinking-v1-draft.1,
  but the synthetic long-context workloads are NOT the standard protocol.
Run after standard benchmark on same Strata instance; caches may have warmed.
Public release requires inspection for hostname/path/IP and secrets.
EOF

nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv \
  > "$RUN/gpu-details.csv"

printf 'utc,temp_c,gpu_util_pct,vram_mib,power_w,fan_pct,pcie_gen,pcie_width,ram_available_kib,swap_used_kib\n' \
 > "$RUN/telemetry.csv"

echo "===== THINKING xhigh LONG CONTEXT: $CTX / INPUT TARGET $TOKENS TOKENS ====="
echo "RESULT DIRECTORY: $RUN"

"$BUNDLED_PYTHON" -u "$SCRIPT" --target "$TOKENS" --out "$RUN/evidence" \
  > "$RUN/console.log" 2>&1 &
PID=$!

trap 'if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then kill -TERM "$PID" || true; fi' EXIT

INTERRUPTED=0
TICKS=0
while kill -0 "$PID" 2>/dev/null; do
  GPU="$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw,fan.speed --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
  LINK="$(nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
  AVAIL="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  SWAP_TOTAL="$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)"
  SWAP_FREE="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
  SWAP_USED="$(( SWAP_TOTAL - SWAP_FREE ))"
  [[ -n "$LINK" ]] || LINK="n/a,n/a"
  if [[ -z "$GPU" ]]; then
    echo "STOP: GPU telemetry unavailable" >&2
    kill -TERM "$PID" 2>/dev/null || true
    INTERRUPTED=1
    break
  fi
  printf '%s,%s,%s,%s,%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$GPU" "$LINK" "$AVAIL" "$SWAP_USED" >> "$RUN/telemetry.csv"
  TEMP="${GPU%%,*}"; TEMP="${TEMP//[[:space:]]/}"
  if [[ ! "$TEMP" =~ ^[0-9]+$ ]] || (( TEMP >= 75 )); then
    echo "STOP: GPU temperature unavailable or reached >=75 C ($TEMP)" >&2
    kill -TERM "$PID" 2>/dev/null || true
    INTERRUPTED=1
    break
  fi
  if (( AVAIL < 1 * 1024 * 1024 || SWAP_USED > 4 * 1024 * 1024 )); then
    echo "STOP: memory pressure (available=${AVAIL} KiB, swap_used=${SWAP_USED} KiB)" >&2
    kill -TERM "$PID" 2>/dev/null || true
    INTERRUPTED=1
    break
  fi
  TICKS=$((TICKS + 1))
  if (( TICKS % 15 == 0 )); then
    tail -n 1 "$RUN/console.log" || true
  fi
  sleep 2
done

RC=0
wait "$PID" || RC=$?
PID=''
trap - EXIT
tail -n 35 "$RUN/console.log" || true
if (( INTERRUPTED == 1 || RC != 0 )); then
  echo "STOP: experiment interrupted/failed. Preserve partial evidence: $RUN (exit=$RC)" >&2
  exit 4
fi

test -s "$RUN/evidence/summary.json"
test -s "$RUN/evidence/long-results.jsonl"
echo "===== DONE ====="
cat "$RUN/evidence/summary.json"
echo
echo "GPU TELEMETRY: $RUN/telemetry.csv"
echo "FULL EVIDENCE: $RUN/evidence"
echo "LONG CONTEXT TEST COMPLETE"
