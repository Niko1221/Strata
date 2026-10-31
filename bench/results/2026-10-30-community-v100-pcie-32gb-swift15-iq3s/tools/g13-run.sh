#!/usr/bin/env bash
# RES-20261029-01 G1.3 (TASK-20261029-04): baseline profile of Strata on v100host's V100, as tasks/TASK-20261029-04-*/brief.md
# says, on Swift 1.5 IQ3_S (operator 2026-10-29). It takes the model-run lock, stops production serving (the production clients
# lose v100host for the run, ~50 min) and on exit, whatever happens, restarts the config production was serving when the run
# began. Arms serve on 127.0.0.1:8081, never the tailnet port. GPU clocks, power and throttle reasons, and CPU clocks, are
# logged throughout.
# Usage: g13-run.sh [PHASES]   default: "smoke stream parity repro serve profile"; smoke failing ends the run
set -u
exec 9><stage>/model-run.lock; flock 9
PH=${1:-"smoke stream parity repro serve profile"}
A=<archive>/TASK-20261029-04; R=$A/raw; C=$A/inputs/configs; G=<harness>/g13
BASE=http://127.0.0.1:8081/v1; DMON=; FREQ=; EPID=
mkdir -p $R/bench $C $A/derived $A/code
log() { echo "# $(date -u +%FT%TZ) $*" | tee -a $R/run.log; }
cp -a $G/. $A/code/ && (cd $A/code && sha256sum -- * > SHA256SUMS)
python3 $G/make-configs.py $C $R > $R/configs.txt
model_of() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['model_name'])" $C/$1.json; }
# what production serves now (the unit's last argument), so the exit trap puts that back
PRODCFG=$(systemctl show -p ExecStart v100host-strata-serve 2>/dev/null | grep -o '<root>/[^ ;]*\.json' | tail -1)
PRODCFG=${PRODCFG:-<strata>/strata-iq3_s-serve.json}

stop_arm() {
  sudo systemctl stop res-g13-arm 2>/dev/null
  for i in $(seq 1 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] && break; sleep 2; done
}
restore() {
  stop_arm; [ -n "$DMON" ] && kill $DMON $FREQ 2>/dev/null
  sudo systemctl reset-failed v100host-strata-serve 2>/dev/null
  sudo systemd-run --unit=v100host-strata-serve --uid=<user> --gid=<user> --setenv=HOME=<home> \
    -p StandardOutput=truncate:<stage>/strata-serve.log -p StandardError=inherit \
    /usr/bin/bash <stage>/strata-serve.sh "$PRODCFG" >/dev/null 2>&1
  log "production serving restarted ($PRODCFG); run ends"
}
start_arm() {   # NAME REP
  local m; m=$(model_of $1); EPID=
  sudo systemctl reset-failed res-g13-arm 2>/dev/null
  sudo systemd-run --unit=res-g13-arm --uid=<user> --gid=<user> --setenv=HOME=<home> \
    -p StandardOutput=truncate:$R/server-$1-r$2.log -p StandardError=inherit \
    <strata-dev>/.venv/bin/python <strata-dev>/serve/server.py --engine strata --config $C/$1.json --port 8081 >/dev/null 2>&1
  for i in $(seq 1 120); do curl -s -m 5 $BASE/models 2>/dev/null | grep -q "$m" && break; sleep 5; done
  curl -s -m 5 $BASE/models | grep -q "$m" || { log "$1 r$2: not ready after 600 s"; return 1; }
  EPID=$(pgrep -f "^<root>/[^ ]*/strata --serve" | head -1)     # the engine; under nsys too (its child)
  [ -n "$EPID" ] || { log "$1 r$2: ready but no engine process found"; return 1; }
  numastat -p "$EPID" > $R/numastat-$1-r$2.txt 2>&1
  grep -a -i -E "pcie|numa|worker|expert cache|avx|vram|slots|switch|huge|mlock" $R/engine-$1.log | tail -40 > $R/startfacts-$1-r$2.txt
  log "$1 r$2: ready, engine pid $EPID"
}
bench() {       # NAME REP CTX [extra decode-bench args]
  local n=$1 r=$2 x=$3; shift 3
  python3 $G/decode-bench.py --base $BASE --model "$(model_of $n)" --log $R/engine-$n.log --ctx $x --arm $n --rep $r \
    --out $R/bench "$@" >> $R/bench-stdout.log 2>&1 || { log "$n r$r: bench failed"; return 1; }
}
arm() { start_arm $1 $2 && bench $1 $2 $3; stop_arm; }

trap restore EXIT
trap 'log "signal: ending"; exit 143' TERM INT HUP
sudo systemctl stop v100host-strata-serve; stop_arm
log "production serving stopped (was $PRODCFG); phases: $PH"
nvidia-smi dmon -s pucvt -d 1 -o DT > $R/dmon.log 2>&1 & DMON=$!
( while :; do echo "$(date -u +%T) cpu0 $(( $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) / 1000 )) cpu20 $(( $(cat /sys/devices/system/cpu/cpu20/cpufreq/scaling_cur_freq) / 1000 )) max $(( $(sort -n /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq | tail -1) / 1000 )) MHz"; sleep 2; done ) > $R/cpufreq.log 2>&1 & FREQ=$!

for p in $PH; do case $p in
  smoke)    # the rel build (no BUILD.json beside it) answers; nsys wraps the engine and a session starts and stops; perf runs
    sudo perf --version > $R/smoke.txt 2>&1 || { log "smoke: perf missing"; exit 3; }
    start_arm bench-rel 0 && bench bench-rel 0 4096 --tasks code --max-tokens 32 --out $R/smoke \
      || { stop_arm; log "smoke: rel arm failed"; exit 3; }
    stop_arm
    if start_arm bench-nsys 0; then
      /usr/local/cuda-12.8/bin/nsys start --session=g13 --output=$R/smoke/nsys-smoke --force-overwrite=true >> $R/nsys.log 2>&1
      bench bench-nsys 0 4096 --tasks code --max-tokens 32 --out $R/smoke
      /usr/local/cuda-12.8/bin/nsys stop --session=g13 >> $R/nsys.log 2>&1
    fi; stop_arm; /usr/local/cuda-12.8/bin/nsys shutdown --session=g13 >> $R/nsys.log 2>&1   # frees the name for later
    ls $R/smoke/nsys-smoke.nsys-rep >/dev/null 2>&1 && log "smoke done (nsys report written)" \
      || log "smoke done; nsys gave no report: the profile phase's nsys step will be skipped"
    ;;
  stream)   # memory bandwidth per socket, local and across UPI, engine stopped
    for s in "0 0" "1 1" "0 1" "1 0"; do set -- $s
      OMP_NUM_THREADS=20 OMP_PROC_BIND=spread numactl --cpunodebind=$1 --membind=$2 <tools>/stream/stream \
        | grep -E "^(Copy|Scale|Add|Triad):" | sed "s/^/cpu$1-mem$2 /" >> $R/stream.txt; done
    OMP_NUM_THREADS=40 OMP_PROC_BIND=spread numactl --interleave=all <tools>/stream/stream \
      | grep -E "^(Copy|Scale|Add|Triad):" | sed "s/^/both-interleave /" >> $R/stream.txt
    log "stream done" ;;
  parity)   # dev rel build vs production binary, bench shape, interleaved; also the noise floor
    for r in 1 2 3; do arm bench-prod $r 4096,16384; arm bench-rel $r 4096,16384; done; log "parity done" ;;
  repro)    # does the reproducible recipe give byte-identical outputs across restarts and across the two binaries?
    arm bench-prod-repro 1 4096,16384; arm bench-rel-repro 1 4096,16384; arm bench-rel-repro 2 4096,16384
    log "repro done" ;;
  serve)    # the serve shape (262K x 2) that the production clients get, one request at a time
    for r in 1 2 3; do arm serve-rel $r 4096,16384,131072; done; log "serve done" ;;
  profile)  # where the time goes: verify-profile stamps, perf on the engine, pcm memory bandwidth, nsys graph trace
    sudo modprobe msr 2>/dev/null
    if start_arm bench-prof 1; then
      (sleep 1; sudo perf record -F 499 -g -p $EPID -o $R/perf-bench-prof.data -- sleep 22 >/dev/null 2>&1) & P1=$!
      (sleep 1; sudo pcm-memory 1 -csv=$R/pcm-bench-prof.csv -i=22 >/dev/null 2>&1) & P2=$!
      bench bench-prof 1 4096; wait $P1 $P2; fi; stop_arm
    if start_arm serve-prof 1; then
      (sleep 1; sudo perf record -F 499 -g -p $EPID -o $R/perf-serve-prof.data -- sleep 45 >/dev/null 2>&1) & P1=$!
      (sleep 1; sudo pcm-memory 1 -csv=$R/pcm-serve-prof.csv -i=45 >/dev/null 2>&1) & P2=$!
      bench serve-prof 1 16384; wait $P1 $P2; fi; stop_arm
    if ls $R/smoke/nsys-smoke.nsys-rep >/dev/null 2>&1 && start_arm bench-nsys 1; then
      /usr/local/cuda-12.8/bin/nsys start --session=g13 --output=$R/nsys-bench --force-overwrite=true >> $R/nsys.log 2>&1
      bench bench-nsys 1 4096
      /usr/local/cuda-12.8/bin/nsys stop --session=g13 >> $R/nsys.log 2>&1; fi; stop_arm
    /usr/local/cuda-12.8/bin/nsys shutdown --session=g13 >> $R/nsys.log 2>&1
    sudo chown <user>:<user> $R/perf-*.data $R/pcm-*.csv 2>/dev/null
    log "profile done" ;;
esac; done

python3 - "$A" "$PRODCFG" <<'PY'
import hashlib, json, os, subprocess, sys, time
a = sys.argv[1]
def sha(p):
    try: return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except OSError: return None
m = {"task": "TASK-20261029-04", "research": "RES-20261029-01", "host": os.uname().nodename, "device": "Tesla V100-PCIE-32GB",
     "model": "Swift 1.5 IQ3_S (ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), <data>/models/swift-IQ3_S",
     "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
     "binaries": {p: sha(p) for p in ["<strata>/engine-cuda12/strata", "<strata-dev>/build-v100-rel/strata",
                                      "<strata-dev>/build-v100-prof/strata"]},
     "source": {"production": "5e8a4cca", "dev": subprocess.run(["git", "-C", "<strata-dev>", "rev-parse", "HEAD"],
                                                                capture_output=True, text=True).stdout.strip()},
     "configs": {f: sha(os.path.join(a, "inputs/configs", f)) for f in sorted(os.listdir(os.path.join(a, "inputs/configs")))},
     "env_set_by_arms": ["STRATA_IQ_MT_MIN", "STRATA_VERIFY_PROFILE", "STRATA_DECODE_TIMING", "NSYS_SESSION"],
     "production_restored": sys.argv[2],
     "start_and_end": "see raw/run.log (UTC)"}
json.dump(m, open(os.path.join(a, "manifest.json"), "w"), indent=1)
PY
log "manifest written"
