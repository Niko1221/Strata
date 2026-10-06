#!/usr/bin/env bash
# The A/B harness this report's short arms ran through, vendored from this
# machine's bench/ab/run.sh so the report folder is enough to repeat them.
#
# Arm A is the install config as it stands; arm B is that config plus the extra
# engine args/env given on the command line. Each arm starts its own server,
# waits for {"loaded": true}, runs tools/hip/bench_prefill.py (one warm-up, 4
# fresh prompts, 4 follow-ups) and kills the server again.
#
# Usage: ab-run.sh <label> [FLAG VALUE ...] [--env KEY=VALUE ...]
#   AB_OUT  configs, arm JSON and server output go here (default <report>/data/ab-work)
#   AB_CFG  base config            (default <root>/strata-coder-iq1_m.json)
#   AB_LOG  engine log to append to (default <root>/strata-coder-iq1_m.log)
#
# The B config takes bare (FLAG, VALUE) pairs - no "--arg" marker, which the
# original harness raises IndexError on before arm B's config is written.
#
# Like the original, this leaves a server running on the base config at the
# end; run-arm.sh kills it again.
set -u
REPORT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
ROOT="$(cd -- "$REPORT/../../.." && pwd)"
OUT="${AB_OUT:-$REPORT/data/ab-work}"
CFG="${AB_CFG:-$ROOT/strata-coder-iq1_m.json}"
LOG="${AB_LOG:-$ROOT/strata-coder-iq1_m.log}"
MODEL=qwen3.8-flash-next-coder-iq1_m
PY="$ROOT/.venv/bin/python"
LABEL=$1; shift
mkdir -p "$OUT"

arm() { # $1=label $2=config path
  local label=$1 cfg=$2 out=$OUT/server-$1.out
  pkill -f "serve/server.py" 2>/dev/null
  pkill -f "engine/strata --serve" 2>/dev/null
  sleep 5
  : > "$out"
  nohup "$PY" "$ROOT/serve/server.py" --engine strata --config "$cfg" --port 8080 --open \
    > "$out" 2>&1 &
  local srv=$! ok=0 i=0
  while [ $i -lt 600 ]; do
    curl -s http://127.0.0.1:8080/health | grep -q '"loaded": true' && { ok=1; break; }
    kill -0 "$srv" 2>/dev/null || break
    sleep 2; i=$((i+2))
  done
  if [ "$ok" != 1 ]; then echo "arm $label FAILED to reach READY"; kill "$srv" 2>/dev/null; return 1; fi
  "$PY" "$ROOT/tools/hip/bench_prefill.py" --model "$MODEL" \
    --url http://127.0.0.1:8080 --engine-log "$LOG" --label "$label" \
    --output "$OUT/$label.json" || echo "bench $label FAILED"
  kill "$srv" 2>/dev/null
  pkill -f "engine/strata --serve" 2>/dev/null
  sleep 3
}

# build the B config: append extra args / env to a copy
CFG_B=$OUT/cfg-$LABEL-B.json
"$PY" - "$CFG" "$CFG_B" "$@" <<'EOF'
import json, sys
cfg_b, extras = sys.argv[2], sys.argv[3:]
d = json.load(open(sys.argv[1]))
args, env = list(d["args"]), dict(d.get("env") or {})
i = 0
while i < len(extras):
    if extras[i].startswith("--env"):
        k, _, v = extras[i+1].partition("=")
        env[k] = v; i += 2
    else:
        args += [extras[i], extras[i+1]]; i += 2
d["args"] = args
if env: d["env"] = env
json.dump(d, open(cfg_b, "w"), indent=2)
EOF

echo "waiting 30 s before the arms..."
sleep 30
arm A "$CFG"
arm B "$CFG_B"

# leave a server running on the base config, as the original harness does
if [ -x "$ROOT/run-coder-iq1_m.sh" ]; then
  nohup "$ROOT/run-coder-iq1_m.sh" > "$OUT/server-after-arm.out" 2>&1 &
fi

"$PY" - "$LABEL" "$OUT/A.json" "$OUT/B.json" <<'EOF'
import json, statistics, sys
lab, path_a, path_b = sys.argv[1], sys.argv[2], sys.argv[3]
a = json.load(open(path_a))
b = json.load(open(path_b))
def key(rs): return {(r['kind'], r['trial']): r['metrics'] for r in rs}
A, B = key(a), key(b)
print(f"A/B [{lab}]: A = current config, B = current + variant")
for k in sorted(A):
    if k in B:
        da, db = A[k]['decode_tps'], B[k]['decode_tps']
        pa, pb = A[k]['prefill_tps'], B[k]['prefill_tps']
        print(f"{k}: decode {da:6.1f} -> {db:6.1f} ({100*(db-da)/da:+5.1f}%) | prefill {pa:7.1f} -> {pb:7.1f}")
for name, rs in (("A", a), ("B", b)):
    print(f"mean wall {name}: {statistics.mean(r['wall_s'] for r in rs):.2f} s")
EOF
echo "done" > "$OUT/DONE-$LABEL"
