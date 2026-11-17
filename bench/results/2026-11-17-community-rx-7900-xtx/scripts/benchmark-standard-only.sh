#!/bin/bash
# benchmark-standard-only.sh - the standard-bench phases that the first window missed
# (its serves died on a stale vision path in serve-ud-smoke.json; vision now re-pointed
# to the UD mmproj). Two arms, identical config: vanilla v0.2.15 build and the fork
# build, fresh serve each, strata.py bench run per docs/BENCHMARK_PROTOCOL.md.
# Prod 8081 stopped for the window; trap-restored. Serve errors live in
# `journalctl --user -u bench-serve` (the redirect only catches the launch line).
set -u
EXP=~/strata-rocm-experiments/20261003-231142
ROOT=~/projects/strata-amd
TREE=~/projects/strata-r160
PY=$EXP/venv/bin/python
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/artifacts/rocm/logs/bench-standard-$STAMP.log
OUTD=$ROOT/artifacts/rocm/results/bench-rx7900xtx-20261117T011142Z
mkdir -p "$OUTD"; exec > >(tee -a "$LOG") 2>&1
restore_8081() {
    systemctl --user stop bench-serve 2>/dev/null
    systemctl --user reset-failed strata-8081.service 2>/dev/null
    systemctl --user start strata-8081.service 2>/dev/null
    for i in $(seq 1 45); do
        curl -s -m 3 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status": "ok"' && { echo "RESTORE_8081 ok"; return 0; }; sleep 4
    done
    echo "RESTORE_8081 FAILED"
}
trap restore_8081 EXIT
vram() { local v; v=$(cat /sys/class/drm/card1/device/mem_info_vram_used 2>/dev/null || echo 999999999999); echo $((v / 1048576)); }

make_cfg() {  # make_cfg <tag> <exe>
    python3 - "$EXP/serve-ud-smoke.json" "$EXP/serve-bench-$1.json" "$2" <<'PY'
import json, sys
src, dst, exe = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open(src)); c["exe"] = exe
c["note"] = "standard-benchmark arm " + dst
json.dump(c, open(dst, "w"), indent=1)
print("cfg written:", dst, "exe:", exe)
PY
}

serve_fresh() {  # serve_fresh <tag>
    systemctl --user stop bench-serve 2>/dev/null; sleep 4
    for i in $(seq 1 30); do [ "$(vram)" -le 2600 ] && break; sleep 4; done
    systemctl --user reset-failed bench-serve 2>/dev/null
    systemd-run --user --collect --unit=bench-serve -p OOMScoreAdjust=-100 \
        env LD_LIBRARY_PATH=$EXP/sdk/lib STRATA_HIPBLASLT_TUNING=$EXP/gfx1100-hipblaslt-100500.txt \
        STRATA_PREFILL_RING=96 STRATA_SELECT_WMMA=1 STRATA_HIP_WMMA=1 STRATA_SH_STREAM=1 \
        STRATA_ROUTE_TAIL_SKIP=7 STRATA_ROUTE_PRIOR=0.5 STRATA_GDN_CHUNKED=1 STRATA_PF_GEMM=1 \
        STRATA_PF_PAD=1 STRATA_KV_RING_ADAPT=1 \
        $PY "$TREE/serve/server.py" --engine strata --config "$EXP/serve-bench-$1.json" \
        --port 8084 --host 127.0.0.1 > /dev/null 2>&1
    for i in $(seq 1 50); do
        curl -s -m 3 http://127.0.0.1:8084/health 2>/dev/null | grep -q '"loaded": true' && { echo "serve $1 up (fresh, zero prior requests)"; return 0; }
        sleep 6
    done
    echo "serve $1 FAILED - see journalctl --user -u bench-serve"; return 1
}

run_standard() {  # run_standard <tag> <exe> <source-rev>
    make_cfg "$1" "$2" || return 1
    serve_fresh "$1" || return 1
    ( cd "$TREE" && PYTHONPATH=~/bench-venv-lib $PY strata.py bench run \
        --url http://127.0.0.1:8084 \
        --track default \
        --track-notes "engine $3; production env verbatim (gfx1100 knob set, see github.com/xyzzing/qwen-rdna3-optimization-ledger); UD-IQ3_XXS pack (compat-bf16); fresh server, zero prior requests" \
        --source-revision "$3" \
        --quantization "UD-IQ3_XXS (iq_pack, HC tensors compat-bf16)" \
        --optional-concurrency ) > "$OUTD/standard-$1.log" 2>&1
    echo "standard bench $1 rc=$?"
    tail -6 "$OUTD/standard-$1.log"
    systemctl --user stop bench-serve 2>/dev/null; sleep 5
}

echo "== standard-bench window $STAMP =="
[ -x ~/projects/strata-v0215/build-0161-vanilla/strata ] || { echo "vanilla binary missing"; exit 3; }
git -C "$TREE" log --oneline -1
systemctl --user stop strata-8081.service
for i in $(seq 1 45); do [ "$(vram)" -le 2600 ] && break; sleep 4; done

run_standard vanilla ~/projects/strata-v0215/build-0161-vanilla/strata "v0.2.15 (c77572806, engine 0.1.61, no fork commits)"
run_standard fork    ~/projects/strata-r160/build-0161-hip/strata "v0.2.15 + carried fork line (addf31dca; github.com/xyzzing/qwen-rdna3-optimization-ledger)"

echo "== standard results locations =="
ls -d "$TREE"/bench/results/*standard* 2>/dev/null | tail -4
echo "== done (trap restores 8081) =="
