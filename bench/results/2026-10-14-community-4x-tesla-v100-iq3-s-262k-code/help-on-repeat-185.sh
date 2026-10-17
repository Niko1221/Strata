#!/bin/bash
# Повтор фазы Ф3 (STRATA_PREFILL_HELP=1) на 4xV100: в первом прогоне на 125K два повтора
# из трёх дали wall 65.4 с и 72.5 с против 51 с в базе при неизменной скорости чтения.
# Нужно знать, воспроизводится ли это. Выход пишется в nvlink-full/help-on-2/.
# Рабочее состояние восстанавливает сам.
set -u
D=$HOME/strata185
PY=$D/Strata/.venv/bin/python
BENCH=$D/bin/strata-bench-community.py
PROBE=$D/bin/nvlink-probe-variant.py
OUT=$D/bench-community/nvlink-full
TMP=$D/bench-community/tmp
CFG=strata-185-iq3_s.json
LOGF=$D/Strata/strata-185-iq3_s.log
KEY="sk-9f3a2b8c7d1e4f6a8b0c2d5e7f4a1b8c"
MODEL="qwen3.8-flash-next-iq3_s"
PAT='strata185/Strata/(.venv/bin/python|engine-cuda12/)'
LEN="2048,3072,4096,32768,131072"
say(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
stop_srv(){
    for p in $(pgrep -f "$PAT" 2>/dev/null); do kill -INT "$p" 2>/dev/null; done
    rm -f "$D/run/server-iq3_s.pid"
    local born=$(date +%s)
    while pgrep -f "$PAT" >/dev/null 2>&1; do
        if [ $(( $(date +%s) - born )) -gt 240 ]; then say "через 240 с живы - добиваю TERM"
            for p in $(pgrep -f "$PAT" 2>/dev/null); do kill -TERM "$p" 2>/dev/null; done; sleep 20; break; fi
        sleep 5
    done
}
start_srv(){
    local cfg="$1" tmo="${2:-900}"; cd "$D/Strata" || return 1
    export LD_LIBRARY_PATH="$D/cuda-12.8/lib64"
    nohup "$D/Strata/.venv/bin/python" serve/server.py --engine strata --config "$cfg" --port 8080 \
        > "$D/run/server-iq3_s.out" 2>&1 &
    echo $! > "$D/run/server-iq3_s.pid"
    local born=$(date +%s)
    while [ $(( $(date +%s) - born )) -lt "$tmo" ]; do
        [ "$(curl -s -m 10 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health -H "Authorization: Bearer $KEY")" = "200" ] \
            && { say "health 200 через $(( $(date +%s) - born )) с"; return 0; }
        sleep 5
    done
    say "сервер не поднялся за ${tmo} с"; return 1
}
props(){ curl -s -m 10 http://127.0.0.1:8080/props -H "Authorization: Bearer $KEY" | python3 -c "import json,sys;print(json.load(sys.stdin).get('build_info'))" 2>/dev/null; }
restore(){ stop_srv; start_srv "$CFG" || { say "НЕ УДАЛОСЯ вернуть рабочий конфиг"; return 1; }; say "итоговый движок: $(props)"; }
trap 'say "аварийное восстановление"; restore' EXIT

python3 - "$D/Strata/$CFG" "$D/Strata/$CFG-help-on.json" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
cfg = json.loads(open(src, encoding='utf-8-sig').read())
cfg['env'] = dict(cfg.get('env') or {}); cfg['env']['STRATA_PREFILL_HELP'] = '1'
open(dst, 'w', encoding='utf-8').write(json.dumps(cfg, indent=1, ensure_ascii=False) + "\n")
print("создан", dst, "env=", cfg['env'])
PY

stop_srv
start_srv "$CFG-help-on.json" || { restore; exit 1; }
say "движок: $(props)"
mkdir -p "$OUT/help-on-2"
LOGLINES=$(wc -l < "$LOGF" 2>/dev/null || echo 0)
say "прогрев"; "$PY" "$BENCH" warm-help-on-2 --style code --lengths "$LEN" --out "$TMP/warm-help-on-2" 2>&1 | tail -2
say "фазы PCIe"; "$PY" "$PROBE" "$OUT/help-on-2" > "$OUT/help-on-2/probe.log" 2>&1
say "замер"; export STRATA_COMMIT=296b59d2f73c8f04a805df29c2fa97655adace78 STRATA_BRANCH=v0.1.44.1
"$PY" "$BENCH" help-on-2 --style code --lengths "$LEN" --out "$TMP/help-on-2" 2>&1 | tail -3
cp "$TMP/help-on-2/runs.json" "$OUT/help-on-2/runs.json" 2>/dev/null
tail -n +$((LOGLINES + 1)) "$LOGF" 2>/dev/null | grep -iE "helper|peer|P2P|CPU share|tail skip|layer split|stage|resident|prompt path" | tail -40 > "$OUT/help-on-2/engine-log.txt"
say "новых строк в логе движка: $(( $(wc -l < "$LOGF" 2>/dev/null || echo 0) - LOGLINES ))"
say "замер повторён"
rm -f "$D/Strata/$CFG-help-on.json"
trap - EXIT
restore
echo "ПОВТОР_ЗАВЕРШЁН"
