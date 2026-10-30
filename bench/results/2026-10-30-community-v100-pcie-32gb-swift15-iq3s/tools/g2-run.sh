#!/usr/bin/env bash
# RES-20261029-01 G2 flag trials (TASK-20261029-06) on v100host's V100, Swift 1.5 IQ3_S, as tasks/TASK-20261029-06-*/brief.md
# says. Also the G1.3 nsys capture that G1.3's smoke step could not make (nsys 2024.6 refused --cpuctxsw on launch). It
# takes the model-run lock, stops production serving (the clients lose v100host for the run) and on exit, whatever
# happens, restarts the config production was serving when the run began. Arms serve on 127.0.0.1:8081. Arms are
# interleaved within each repeat (repeat 2 in reverse order) and every repeat is a fresh engine start.
# Usage: g2-run.sh [PHASES]   default: "nsys bench repro serve"; REPS=N (default 3)
set -u
exec 9><stage>/model-run.lock; flock 9
PH=${1:-"nsys bench repro serve"}; REPS=${REPS:-3}
A=<archive>/TASK-20261029-06; R=$A/raw; C=$A/inputs/configs; G=<harness>/g2
BASE=http://127.0.0.1:8081/v1; DMON=; FREQ=; EPID=
NSYS=/usr/local/cuda-12.8/bin/nsys
BENCH_ARMS="bench-ref bench-v1 bench-v1x"
SERVE_ARMS="serve-ref serve-v1 serve-v4 serve-v2 serve-v3 serve-pc0 serve-pf32k"
LONG_ARMS=" serve-ref serve-v2 serve-v3 serve-pf32k "          # these also get one 128K request (code)
mkdir -p $R/bench $C $A/derived $A/code
log() { echo "# $(date -u +%FT%TZ) $*" | tee -a $R/run.log; }
cp -a $G/. $A/code/ && (cd $A/code && sha256sum -- *.sh *.py > SHA256SUMS)
python3 $G/make-arms.py $C $R > $R/configs.txt
model_of() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['model_name'])" $C/$1.json; }
PRODCFG=$(systemctl show -p ExecStart v100host-strata-serve 2>/dev/null | grep -o '<root>/[^ ;]*\.json' | tail -1)
PRODCFG=${PRODCFG:-<strata>/strata-iq3_s-serve.json}

stop_arm() {
  sudo systemctl stop res-g2-arm 2>/dev/null
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
  sudo systemctl reset-failed res-g2-arm 2>/dev/null
  sudo systemd-run --unit=res-g2-arm --uid=<user> --gid=<user> --setenv=HOME=<home> \
    -p StandardOutput=truncate:$R/server-$1-r$2.log -p StandardError=inherit \
    <strata-dev>/.venv/bin/python <strata-dev>/serve/server.py --engine strata --config $C/$1.json --port 8081 >/dev/null 2>&1
  for i in $(seq 1 120); do
    curl -s -m 5 $BASE/models 2>/dev/null | grep -q "$m" && break
    systemctl is-active -q res-g2-arm || { log "$1 r$2: server exited before ready"; return 1; }
    sleep 5; done
  curl -s -m 5 $BASE/models | grep -q "$m" || { log "$1 r$2: not ready after 600 s"; return 1; }
  EPID=$(pgrep -f "^<root>/[^ ]*/strata --serve" | head -1)
  [ -n "$EPID" ] || { log "$1 r$2: ready but no engine process found"; return 1; }
  numastat -p "$EPID" > $R/numastat-$1-r$2.txt 2>&1
  grep -a -i -E "pcie|numa|worker|expert cache|avx|vram|slots|switch|huge|mlock|kv|chunk|STRATA_" $R/engine-$1.log | tail -60 > $R/startfacts-$1-r$2.txt
  log "$1 r$2: ready, engine pid $EPID"
}
bench() {       # NAME REP CTX [extra decode-bench args]
  local n=$1 r=$2 x=$3; shift 3
  python3 $G/decode-bench.py --base $BASE --model "$(model_of $n)" --log $R/engine-$n.log --ctx $x --arm $n --rep $r \
    --out $R/bench "$@" >> $R/bench-stdout.log 2>&1 || { log "$n r$r: bench failed"; return 1; }
}
arm() {         # NAME REP CTX: one engine start; the long arms add one 128K request after the short ones
  if start_arm $1 $2; then
    bench $1 $2 $3
    case "$LONG_ARMS" in *" $1 "*) [ "${1#serve}" != "$1" ] && bench $1 $2 131072 --tasks code ;; esac
  fi; stop_arm
}
order() { if [ $(( $1 % 2 )) -eq 0 ]; then echo "$2" | tr ' ' '\n' | tac | tr '\n' ' '; else echo "$2"; fi; }

trap restore EXIT
trap 'log "signal: ending"; exit 143' TERM INT HUP
sudo systemctl stop v100host-strata-serve; stop_arm
log "production serving stopped (was $PRODCFG); phases: $PH; reps $REPS"
nvidia-smi dmon -s pucvt -d 1 -o DT > $R/dmon.log 2>&1 & DMON=$!
( while :; do echo "$(date -u +%T) cpu0 $(( $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) / 1000 )) cpu20 $(( $(cat /sys/devices/system/cpu/cpu20/cpufreq/scaling_cur_freq) / 1000 )) max $(( $(sort -n /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq | tail -1) / 1000 )) MHz"; sleep 2; done ) > $R/cpufreq.log 2>&1 & FREQ=$!

for p in $PH; do case $p in
  nsys)     # G1.3's missing trace: the prof engine under nsys, one 4K decode request captured (CUDA graph node trace)
    if start_arm bench-nsys 1; then
      $NSYS start --session=g13 --sample=none --cpuctxsw=none --output=$R/nsys-bench --force-overwrite=true >> $R/nsys.log 2>&1
      bench bench-nsys 1 4096 --tasks code
      $NSYS stop --session=g13 >> $R/nsys.log 2>&1
    fi; stop_arm; $NSYS shutdown --session=g13 >> $R/nsys.log 2>&1
    ls $R/nsys-bench.nsys-rep >/dev/null 2>&1 && log "nsys done (report written)" || log "nsys: no report" ;;
  bench)    # V1 / V1x against the reference, bench shape (32K x 1), decode speed
    for r in $(seq 1 $REPS); do for a in $(order $r "$BENCH_ARMS"); do arm $a $r 4096,16384; done; done; log "bench done" ;;
  repro)    # exactness of V1 / V1x: under the reproducible recipe their outputs must hash equal to the reference's
    # two repeats: in G1.3 1 of 12 recipe outputs flipped a near-tie word, so one hash per arm cannot judge exactness
    for r in 1 2; do for a in bench-ref-repro bench-v1-repro bench-v1x-repro; do arm $a $r 4096,16384; done; done
    log "repro done" ;;
  serve)    # the serve shape (262K x 2), one request at a time: V1, V4, V2 (KV in RAM), V3 (one slot), node-0 pool, 32K chunks
    for r in $(seq 1 $REPS); do for a in $(order $r "$SERVE_ARMS"); do arm $a $r 4096,16384; done; done; log "serve done" ;;
esac; done

python3 - "$A" "$PRODCFG" <<'PY'
import hashlib, json, os, subprocess, sys, time
a = sys.argv[1]
def sha(p):
    try: return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except OSError: return None
m = {"task": "TASK-20261029-06", "research": "RES-20261029-01", "host": os.uname().nodename, "device": "Tesla V100-PCIE-32GB",
     "model": "Swift 1.5 IQ3_S (ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), <data>/models/swift-IQ3_S",
     "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
     "binaries": {p: sha(p) for p in ["<strata-dev>/build-v100-rel/strata", "<strata-dev>/build-v100-prof/strata"]},
     "source": {"dev": subprocess.run(["git", "-C", "<strata-dev>", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()},
     "configs": {f: sha(os.path.join(a, "inputs/configs", f)) for f in sorted(os.listdir(os.path.join(a, "inputs/configs")))},
     "production_restored": sys.argv[2], "start_and_end": "see raw/run.log (UTC)"}
json.dump(m, open(os.path.join(a, "manifest.json"), "w"), indent=1)
PY
log "manifest written"
