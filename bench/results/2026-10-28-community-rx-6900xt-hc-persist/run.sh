#!/bin/sh
# run.sh: HC_PERSIST / adapt-swaps / vram-reserve A/B on the three Qwen3.8 configs.
# Each arm = copy of the setup config (Strata/strata-*.json) with one change, own server on :8097, tools/strata_bench.py (3 runs).
# Three passes, arms interleaved, so drift hits every arm alike. Live configs are not touched.
set -u
D=${BENCH_DIR:-$PWD}; S=${STRATA_DIR:?set STRATA_DIR to the Strata folder}; PORT=8097

ARMS="coder:base coder:hc coder:swaps160
iq2_xs:base iq2_xs:hc iq2_xs:swaps160 iq2_xs:res2048 iq2_xs:res1536
swift:base swift:hc"

base_of() { case $1 in coder) echo strata-coder-iq1_m.json;; iq2_xs) echo strata-iq2_xs.json;; swift) echo strata-swift-iq3_s.json;; esac; }

arm() {  # arm MODEL VARIANT PASS
    tag=$1-$2-p$3; cfg=$D/config-$tag.json
    python3 -I - "$S/$(base_of $1)" "$cfg" "$2" "$D/engine-$tag.log" "$PORT" <<'P'
import json, sys
src, out, var, log, port = sys.argv[1:]
c = json.load(open(src))
c.pop("args_written", None); c.pop("user_args", None)
a = c["args"]
def put(flag, val):
    if flag in a: a[a.index(flag) + 1] = val
    else: a.extend([flag, val])
if var == "hc": c["env"]["STRATA_HC_PERSIST"] = "1"
elif var == "swaps160": put("--adapt-swaps", "160")
elif var.startswith("res"): put("--vram-reserve-mib", var[3:])
c["log"] = log; c["port"] = int(port); c["open_browser"] = False
json.dump(c, open(out, "w"), indent=1)
P
    cd "$S"
    .venv/bin/python serve/server.py --engine strata --config "$cfg" --port $PORT > "$D/server-$tag.log" 2>&1 &
    srv=$!
    ok=0
    for _ in $(seq 1 200); do
        curl -s -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { ok=1; break; }
        kill -0 $srv 2>/dev/null || break
        sleep 3
    done
    if [ $ok = 1 ]; then
        .venv/bin/python tools/strata_bench.py --url http://127.0.0.1:$PORT --out "$D/bench-$tag" > "$D/bench-$tag.log" 2>&1
        echo "$(date +%T) $tag done" >> "$D/progress.log"
    else
        echo "$(date +%T) $tag FAILED to start (see server-$tag.log)" >> "$D/progress.log"
    fi
    kill $srv 2>/dev/null; wait $srv 2>/dev/null
    sleep 5
}

echo "$(date +%T) start" >> "$D/progress.log"
for p in 1 2 3; do
    echo "$ARMS" | tr ' ' '\n' | while IFS=: read -r m v; do [ -n "$m" ] && arm "$m" "$v" $p; done
done
echo "$(date +%T) all done" >> "$D/progress.log"
