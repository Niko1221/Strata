#!/usr/bin/env bash
# jevport-arm.sh - switch the jevport A/B preset between readout arms and calibrators, then VERIFY.
#
#   jevport-arm.sh c            # M3.1 default: per-question prompts + state parking (no extra env)
#   jevport-arm.sh b            # one consolidated prompt, jevpos row capture (STRATA_JEV_ROWS=1)
#   jevport-arm.sh h            # pin + capture hybrid (STRATA_JEV_HYBRID=1)
#   jevport-arm.sh c --calib none            # revert calib-L.json to the .prev backup
#   jevport-arm.sh b --calib calib-L-mf.json # install a fitted calibrator (hot-applied at next start)
#   jevport-arm.sh c --labels digit-lower    # #3643 label alphabet 0-9/a-z (default is upper A-Z)
#
# Why a script: every arm is an env var on the llama-swap preset, and the preset only restarts when
# config.yaml changes - so hand-editing the env list and "just retrying" silently measures the OLD arm.
# This flips the env, waits for llama-swap's config watcher, cold-starts the preset and smoke-tests it.
set -euo pipefail

CONFIG=/opt/data/llm-swap/config.yaml
BASE=http://10.89.0.57:8080
PRESET=p-jevport-vision-3090
CALIB_DIR=/cfg/strata/Jev-port-cuda/calib
CALIB_HOST=/opt/data/llm-swap/strata/Jev-port-cuda/calib
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
DOCKER=$HOME/.local/bin/docker
ARM=${1:-c}
shift || true
CALIB=""
LABELS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --calib) CALIB=$2; shift 2 ;;
    --labels) LABELS=$2; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

case "$ARM" in
  c) EXTRA=() ;;
  b) EXTRA=("STRATA_JEV_ROWS=1") ;;
  h) EXTRA=("STRATA_JEV_HYBRID=1") ;;
  *) echo "arm must be c|b|h (got '$ARM')" >&2; exit 2 ;;
esac
case "${LABELS:-upper}" in
  upper) : ;;                      # the default - keep the env byte-identical to the release
  digit-lower) EXTRA+=("STRATA_JEV_LABELS=digit-lower") ;;
  *) echo "labels must be upper|digit-lower (got '$LABELS')" >&2; exit 2 ;;
esac

python3 - "$CONFIG" "$ARM" "${EXTRA[@]:-}" <<'PY'
import re, sys
cfg, arm = sys.argv[1], sys.argv[2]
extra = [a for a in sys.argv[3:] if a]
base = ['LD_LIBRARY_PATH=/cfg/toolchain/cuda-13.0/lib64', 'STRATA_JEV_MODES=chat,decision']
keep = ['STRATA_JEV_ADMIN_KEY=17bfd6d3ae9e19320896e69c860f19ec']   # localhost admin surface (/admin/calib)
want = '    env: [' + ', '.join('"%s"' % e for e in base + extra + keep) + ']'

lines = open(cfg).read().splitlines()
# Anchor on the preset KEY, then take the first `    env:` line before the next `  "key":` line.
# Block-splitting on blank lines is WRONG here: several presets share a blank-line block, so the
# first env: in the block belongs to whatever preset came first (this clobbered kvmem-b once).
start = None
for i, ln in enumerate(lines):
    if re.match(r'^  "p-jevport-vision-3090":\s*$', ln):
        start = i
        break
if start is None:
    sys.exit('preset p-jevport-vision-3090 not found')
env_i = None
for j in range(start + 1, len(lines)):
    if re.match(r'^  "', lines[j]) and j > start:
        break
    if lines[j].startswith('    env: '):
        env_i = j
        break
if env_i is None:
    sys.exit('env: line not found inside the jevport preset block')
old = lines[env_i]
lines[env_i] = want
open(cfg, 'w').write('\n'.join(lines) + '\n')
print(f'arm={arm}')
print(f'  - {old.strip()}')
print(f'  + {want.strip()}')
PY

if [ -n "$CALIB" ]; then
  # calib/ is root-owned inside the container (the model writes its own logs there), so every write
  # goes through `docker exec` from a staging file in /cfg/strata - a host-side cp gets EACCES.
  if [ "$CALIB" = "identity" ]; then
    # no calibrator file = identity, per tuple.  Use this for any raw->fit measurement: with a
    # calibrator installed the "raw" column is already f_old(raw) and a new fit COMPOUNDS it,
    # so the fitted params are not installable and the before/after numbers are not honest.
    # L = text, D = image direct, S = image summary.  V has no shipped file (never fitted).
    $DOCKER exec llama-swap sh -c "cd $CALIB_DIR && for f in calib-L.json calib-D.json calib-S.json; do
      if [ -f \$f ]; then mv -f \$f \$f.hold; echo \"  aside: \$f\"; fi
    done; true"
    echo "calibrator: identity (L/D/S files moved to .hold)"
  elif [ "$CALIB" = "none" ]; then
    # restore whatever identity moved aside; fall back to the archived Oct-15 L fit if .hold is gone
    $DOCKER exec llama-swap sh -c "cd $CALIB_DIR && for f in calib-L.json calib-D.json calib-S.json; do
      if [ -f \$f.hold ]; then mv -f \$f.hold \$f; echo \"  restored: \$f\";
      elif [ \"\$f\" = calib-L.json ] && [ -f \$f.prev-20261022 ]; then cp \$f.prev-20261022 \$f; echo \"  restored \$f from .prev-20261022\"; fi
    done; true"
    echo "calibrator: restored from .hold (or the .prev-20261022 archive)"
  else
    [ -f "$SCRIPTS/$CALIB" ] || { echo "no such calibrator: $SCRIPTS/$CALIB" >&2; exit 1; }
    STAGE=/opt/data/llm-swap/strata/calib-incoming.json
    cp "$SCRIPTS/$CALIB" "$STAGE"
    $DOCKER exec llama-swap sh -c "cp $CALIB_DIR/calib-L.json $CALIB_DIR/calib-L.json.prev-$(date +%Y%m%d-%H%M%S) 2>/dev/null || true; cp /cfg/strata/calib-incoming.json $CALIB_DIR/calib-L.json"
    echo "calibrator: installed $CALIB"
  fi
  $DOCKER exec llama-swap sh -c "cd $CALIB_DIR && for f in calib-L.json calib-D.json calib-S.json; do if test -f \$f; then printf '%s: ' \$f; grep -E 'a_noul|c_noul|T_choice' \$f | tr -d ' \n'; echo; fi; done; echo '(absent = identity)'"
fi

echo "restarting llama-swap (a config reload alone does NOT restart an already-running preset,"
echo "so the smoke below would silently measure the OLD arm - that trap is why this script restarts)"
$DOCKER restart llama-swap >/dev/null

KEY=$(grep -m1 '^LLM_SWAP_API_KEY=' /opt/data/llm-swap/.container-env | cut -d= -f2-)
for i in $(seq 1 30); do
  code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' -H "Authorization: Bearer $KEY" "$BASE/v1/models" || true)
  [ "$code" = "200" ] && break
  sleep 2
done
echo "llama-swap /v1/models -> $code"

# cold start of the preset + a 1-item smoke through the real decision surface
head -1 /opt/data/eval/banking/mf-200.jsonl > /tmp/arm-smoke.jsonl
cd "$SCRIPTS"
KEY="$KEY" python3 eval_mf_set.py run --base "$BASE/upstream/$PRESET" --key "$KEY" \
  --set /tmp/arm-smoke.jsonl --out /tmp/arm-smoke.json --label "arm=$ARM" 2>&1 | tail -1

echo "--- live env on the serve process:"
LIVE_ALL=$($DOCKER exec llama-swap sh -c 'for p in /proc/[0-9]*; do
  e=$(tr "\0" "\n" < $p/environ 2>/dev/null || true)
  case "$e" in *STRATA_JEV_MODES*) echo "$e" | grep -E "^STRATA_JEV_(ROWS|HYBRID|LABELS)=" ;; esac
done' 2>/dev/null || true)
LIVE=$(echo "$LIVE_ALL" | grep -E "^STRATA_JEV_(ROWS|HYBRID)=" || true)   # arm flags only
LIVE_LABELS=$(echo "$LIVE_ALL" | grep -E "^STRATA_JEV_LABELS=" || true)   # label flag only: the arm filter above
# dropped LABELS, so asking for digit-lower always ended in a false "MISMATCH" after setting it correctly
echo "  ${LIVE_ALL:-<no arm/label flags>}"
case "$ARM" in
  c) [ -z "$LIVE" ] || { echo "MISMATCH: asked for arm c, live has: $LIVE" >&2; exit 1; } ;;
  b) echo "$LIVE" | grep -q 'ROWS=1' || { echo "MISMATCH: asked for arm b, live has: $LIVE" >&2; exit 1; } ;;
  h) echo "$LIVE" | grep -q 'HYBRID=1' || { echo "MISMATCH: asked for arm h, live has: $LIVE" >&2; exit 1; } ;;
esac
case "${LABELS:-upper}" in
  digit-lower) echo "$LIVE_LABELS" | grep -q 'LABELS=digit-lower' || { echo "MISMATCH: asked for digit-lower labels, live has: $LIVE_LABELS" >&2; exit 1; } ;;
  upper) echo "$LIVE_LABELS" | grep -q 'LABELS' && { echo "MISMATCH: asked for upper labels, live has: $LIVE_LABELS" >&2; exit 1; } ;;
esac
echo "arm $ARM labels=${LABELS:-upper} confirmed live"
