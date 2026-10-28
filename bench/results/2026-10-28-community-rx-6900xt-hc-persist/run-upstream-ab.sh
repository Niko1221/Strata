#!/bin/sh
# run.sh: IQ2_XS on the upstream-main engine (Strata-upstream, no gfx1030 rocBLAS 5.2.0 table) and on the fork engine,
# each with and without STRATA_HC_PERSIST=1. Base config: the setup config Strata/strata-iq2_xs.json (reserve 3072).
# Own server on :8097, tools/strata_bench.py (3 runs); three passes, arms interleaved. Live configs are not touched.
set -u
D=${BENCH_DIR:-$PWD}; FORK=${FORK_DIR:?the fork checkout with its engine}; UP=${MAIN_DIR:?the upstream-main worktree with build-hip/strata}; PORT=8097
ARMS="up-base up-hc fork-base fork-hc"

arm() {  # arm VARIANT PASS
    tag=iq2_xs-$1-p$2; cfg=$D/config-$tag.json
    case $1 in up-*) tree=$UP;; *) tree=$FORK;; esac
    python3 -I - "$FORK/strata-iq2_xs.json" "$cfg" "$1" "$D/engine-$tag.log" "$PORT" "$UP" <<'P'
import json, sys
src, out, var, log, port, up = sys.argv[1:]
c = json.load(open(src))
c.pop("args_written", None); c.pop("user_args", None)
if var.startswith("up-"):
    c["exe"] = f"{up}/build-hip/strata"; c["cwd"] = up
    c["env"]["STRATA_ROCBLAS_TUNING"] = f"{up}/tools/hip"
if var.endswith("-hc"): c["env"]["STRATA_HC_PERSIST"] = "1"
c["log"] = log; c["port"] = int(port); c["open_browser"] = False
json.dump(c, open(out, "w"), indent=1)
P
    cd "$tree"
    $FORK/.venv/bin/python serve/server.py --engine strata --config "$cfg" --port $PORT > "$D/server-$tag.log" 2>&1 &
    srv=$!
    ok=0
    for _ in $(seq 1 200); do
        curl -s -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { ok=1; break; }
        kill -0 $srv 2>/dev/null || break
        sleep 3
    done
    if [ $ok = 1 ]; then
        $FORK/.venv/bin/python "$tree/tools/strata_bench.py" --url http://127.0.0.1:$PORT --out "$D/bench-$tag" > "$D/bench-$tag.log" 2>&1
        echo "$(date +%T) $tag done" >> "$D/progress.log"
    else
        echo "$(date +%T) $tag FAILED to start (see server-$tag.log)" >> "$D/progress.log"
    fi
    kill $srv 2>/dev/null; wait $srv 2>/dev/null
    sleep 5
}

echo "$(date +%T) ab start" >> "$D/progress.log"
for p in 1 2 3; do
    for v in $ARMS; do arm "$v" $p; done
done
echo "$(date +%T) ab done" >> "$D/progress.log"
