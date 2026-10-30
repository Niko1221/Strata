#!/usr/bin/env bash
# RES-20261029-01 (TASK-20261029-07): the combined serve stacks on v100host's V100, Swift 1.5 IQ3_S, as
# tasks/TASK-20261029-07-*/brief.md says. It takes the model-run lock, stops production serving (the clients lose v100host
# for the run) and on exit, whatever happens, restarts the config production was serving when the run began. Arms serve
# on 127.0.0.1:8081. A throwaway start first (G2's first repeats ran 2-4% slow), then:
#   decode: ref, s2, s1 x REPS, interleaved (repeat 2 reversed), 4K/16K code+prose + 128K code; repeat 1 also runs the
#           twelve sanity checks in the same engine start
#   conc:   two 128K requests at once on ref and s2 (s1 has one slot: they queue, which is itself measured)
#   ladder: the DGX ladder (v41bench prompt set v1, Qwen copy) at levels 1,2,4 on s2 and s1, no prefill cells
# Usage: g4-run.sh [PHASES]   default: "warm decode conc ladder"; REPS=N (default 2)
set -u
exec 9><stage>/model-run.lock; flock 9
PH=${1:-"warm decode conc ladder"}; REPS=${REPS:-2}
A=<archive>/TASK-20261029-07; R=$A/raw; C=$A/inputs/configs; G=<harness>/g4
BASE=http://127.0.0.1:8081/v1; DMON=; FREQ=; EPID=
ARMS="serve-ref serve-s2 serve-s1"
mkdir -p $R/bench $R/ladder $C $A/derived $A/code
log() { echo "# $(date -u +%FT%TZ) $*" | tee -a $R/run.log; }
cp -a $G/. $A/code/ && (cd $A/code && sha256sum -- *.sh *.py > SHA256SUMS)
python3 $G/make-stacks.py $C $R > $R/configs.txt
model_of() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['model_name'])" $C/$1.json; }
PRODCFG=$(systemctl show -p ExecStart v100host-strata-serve 2>/dev/null | grep -o '<root>/[^ ;]*\.json' | tail -1)
PRODCFG=${PRODCFG:-<strata>/strata-iq3_s-serve.json}

stop_arm() {
  sudo systemctl stop res-g4-arm 2>/dev/null
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
  sudo systemctl reset-failed res-g4-arm 2>/dev/null
  sudo systemd-run --unit=res-g4-arm --uid=<user> --gid=<user> --setenv=HOME=<home> \
    -p StandardOutput=truncate:$R/server-$1-r$2.log -p StandardError=inherit \
    <strata-dev>/.venv/bin/python <strata-dev>/serve/server.py --engine strata --config $C/$1.json --port 8081 >/dev/null 2>&1
  for i in $(seq 1 120); do
    curl -s -m 5 $BASE/models 2>/dev/null | grep -q "$m" && break
    systemctl is-active -q res-g4-arm || { log "$1 r$2: server exited before ready"; return 1; }
    sleep 5; done
  curl -s -m 5 $BASE/models | grep -q "$m" || { log "$1 r$2: not ready after 600 s"; return 1; }
  EPID=$(pgrep -f "^<root>/[^ ]*/strata --serve" | head -1)
  [ -n "$EPID" ] || { log "$1 r$2: ready but no engine process found"; return 1; }
  grep -a -i -E "expert cache [0-9]+ slots|slot sessions|KV streaming|VRAM free|reserve" $R/engine-$1.log | tail -12 > $R/startfacts-$1-r$2.txt
  log "$1 r$2: ready, engine pid $EPID"
}
bench() {       # NAME REP CTX [extra decode-bench args]
  local n=$1 r=$2 x=$3; shift 3
  python3 $G/decode-bench.py --base $BASE --model "$(model_of $n)" --log $R/engine-$n.log --ctx $x --arm $n --rep $r \
    --out $R/bench "$@" >> $R/bench-stdout.log 2>&1 || { log "$n r$r: bench failed"; return 1; }
}
order() { if [ $(( $1 % 2 )) -eq 0 ]; then echo "$2" | tr ' ' '\n' | tac | tr '\n' ' '; else echo "$2"; fi; }

trap restore EXIT
trap 'log "signal: ending"; exit 143' TERM INT HUP
sudo systemctl stop v100host-strata-serve; stop_arm
log "production serving stopped (was $PRODCFG); phases: $PH; reps $REPS"
nvidia-smi dmon -s pucvt -d 1 -o DT > $R/dmon.log 2>&1 & DMON=$!
( while :; do echo "$(date -u +%T) cpu0 $(( $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) / 1000 )) cpu20 $(( $(cat /sys/devices/system/cpu/cpu20/cpufreq/scaling_cur_freq) / 1000 )) MHz"; sleep 2; done ) > $R/cpufreq.log 2>&1 & FREQ=$!

for p in $PH; do case $p in
  warm)     # throwaway: one start and one request, not analysed
    start_arm serve-ref 0 && bench serve-ref 0 4096 --tasks code --out $R/warm; stop_arm; log "warm done" ;;
  decode)
    for r in $(seq 1 $REPS); do for a in $(order $r "$ARMS"); do
      if start_arm $a $r; then
        bench $a $r 4096,16384 && bench $a $r 131072 --tasks code
        [ $r -eq 1 ] && { python3 $G/sanity12.py --base $BASE --model "$(model_of $a)" --out $R/sanity --arm $a \
                            >> $R/sanity-stdout.log 2>&1 || log "$a: sanity run failed"; }
      fi; stop_arm
    done; done; log "decode done" ;;
  conc)     # two agents with long transcripts at once
    for a in serve-ref serve-s2 serve-s1; do
      if start_arm $a c; then
        python3 $G/conc2.py --base $BASE --model "$(model_of $a)" --ctx 131072 --arm $a --rep 1 --out $R/conc \
          >> $R/conc-stdout.log 2>&1 || log "$a: conc2 failed"
      fi; stop_arm
    done; log "conc done" ;;
  ladder)   # the DGX ladder, comparable with the z6 session's Swift run (bench-v100host-swift-iq3_s, levels 1-16)
    for a in serve-s2 serve-s1; do
      if start_arm $a l; then
        python3 <stage>/v41bench-qwen.py --base $BASE --model "$(model_of $a)" --label "v100host-swift-$a" --out $R/ladder \
          --levels 1,2,4 --prefill "" --notes "v100host V100 Strata dev rel build, $a ($(basename $C/$a.json)), scaling_max_freq $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq) kHz; v41bench prompt set v1 (Qwen copy, enable_thinking false); 127.0.0.1:8081" \
          >> $R/ladder-stdout.log 2>&1 || log "$a: ladder failed"
      fi; stop_arm
    done; log "ladder done" ;;
esac; done

python3 - "$A" "$PRODCFG" <<'PY'
import hashlib, json, os, subprocess, sys, time
a = sys.argv[1]
def sha(p):
    try: return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except OSError: return None
m = {"task": "TASK-20261029-07", "research": "RES-20261029-01", "host": os.uname().nodename, "device": "Tesla V100-PCIE-32GB",
     "model": "Swift 1.5 IQ3_S (ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), <data>/models/swift-IQ3_S",
     "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
     "binaries": {p: sha(p) for p in ["<strata-dev>/build-v100-rel/strata", "<stage>/v41bench-qwen.py"]},
     "source": {"dev": subprocess.run(["git", "-C", "<strata-dev>", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()},
     "configs": {f: sha(os.path.join(a, "inputs/configs", f)) for f in sorted(os.listdir(os.path.join(a, "inputs/configs")))},
     "production_restored": sys.argv[2], "start_and_end": "see raw/run.log (UTC)"}
json.dump(m, open(os.path.join(a, "manifest.json"), "w"), indent=1)
PY
log "manifest written"
