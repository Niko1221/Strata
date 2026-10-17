#!/bin/bash
# Полная проба межсоединения и путей префила на 4xV100 (dgbox-ai-server) - приложение к
# отчёту 2026-10-14-community-4x-tesla-v100. Задачи:
#   Ф0  при остановленном сервере: потолок P2P по всем шести парам и потолок хост-пути
#       (каждой карты поодиночке и всех четырёх сразу) - то, чего при работающем сервере
#       измерить нельзя (не хватает свободной VRAM);
#   Ф1  документально подтвердить, что --peer-device несовместим с --layer-split;
#   Ф2  базовая конфигурация: замер + фазы PCIe;
#   Ф3  STRATA_PREFILL_HELP=1 (помощник стадии, как его выбирает движок: доли только для T<~3.3K);
#   Ф4  STRATA_PREFILL_HELP=1 + STRATA_PREFILL_HELP_FRAC=0.45 (принудительно на длинных промптах -
#       это и есть цена передачи активаций/строк через хост, которую и хочет знать автор);
#   Ф5  контроль на базовой конфигурации (та же бинарная сборка) - полоса шума.
# Сервер обслуживает агента: первая же остановка обрывает сессию Pi. Скрипт сам возвращает
# рабочее состояние (дерево 0.1.44.1 и рабочий конфиг) даже при аварийном выходе.
#   setsid nohup ~/strata185/bin/nvlink-full-185.sh >> ~/strata185/run/nvlink-full-$(date +%H%M).log 2>&1 &
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
mkdir -p "$OUT" "$TMP"
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
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
}
start_srv(){   # $1 = конфиг, $2 = таймаут подъёма (по умолчанию 900)
    local cfg="$1" tmo="${2:-900}"
    cd "$D/Strata" || return 1
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
probe(){ curl -s -m 120 http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Ответь ровно одно слово: готово\"}],\"max_tokens\":64,\"temperature\":0,\"reasoning_effort\":\"none\"}" | head -c 160; echo; }
englog(){ grep -iE "helper|peer|P2P|CPU share|tail skip|layer split|stage|resident" "$LOGF" 2>/dev/null | tail -25; }

# конфиг-вариант: env и/или аргументы поверх рабочего
mkvariant(){   # $1 = имя; дальше пары env K=V; после -- аргументы
    local name="$1"; shift
    local envjson='{}' argjson='[]' part
    local mode=env; local envs=() args=()
    for part in "$@"; do [ "$part" = "--" ] && { mode=args; continue; }; [ "$mode" = env ] && envs+=("$part") || args+=("$part"); done
    envjson=$(python3 -c "import json,sys;print(json.dumps(dict(p.split('=',1) for p in sys.argv[1:])))" "${envs[@]}")
    argjson=$(python3 -c "import json,sys;print(json.dumps(sys.argv[1:]))" "${args[@]}")
    python3 - "$D/Strata/$CFG" "$D/Strata/$CFG-$name.json" "$envjson" "$argjson" <<'PY'
import json, sys
src, dst, env, args = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), json.loads(sys.argv[4])
cfg = json.loads(open(src, encoding='utf-8-sig').read())
cfg['env'] = dict(cfg.get('env') or {}); cfg['env'].update(env)
cfg['args'] = list(cfg.get('args') or []) + args
open(dst, 'w', encoding='utf-8').write(json.dumps(cfg, indent=1, ensure_ascii=False) + "\n")
print("создан", dst, "env=", env, "args+", args)
PY
}
measure(){   # $1 = имя фазы
    local name="$1"; mkdir -p "$OUT/$name"
    say "$name: прогрев"; "$PY" "$BENCH" "warm-$name" --style code --lengths "$LEN" --out "$TMP/warm-$name" 2>&1 | tail -2
    say "$name: фазы PCIe"; "$PY" "$PROBE" "$OUT/$name" > "$OUT/$name/probe.log" 2>&1
    say "$name: замер"; export STRATA_COMMIT=296b59d2f73c8f04a805df29c2fa97655adace78 STRATA_BRANCH=v0.1.44.1
    "$PY" "$BENCH" "$name" --style code --lengths "$LEN" --out "$TMP/$name" 2>&1 | tail -3
    cp "$TMP/$name/runs.json" "$OUT/$name/runs.json" 2>/dev/null
    englog > "$OUT/$name/engine-log.txt"
    say "$name: готово"
}
restore(){   # рабочий конфиг
    stop_srv
    start_srv "$CFG" || { say "НЕ УДАЛОСЯ поднять рабочий конфиг"; return 1; }
    say "итоговый движок: $(props)"; probe
}
trap 'say "аварийное восстановление"; restore' EXIT

sleep 150   # время агенту договорить: первая остановка сервера оборвёт сессию Pi

# ---------------------------------------------------------------- Ф0: потолок P2P и хост-пути
say "Ф0: остановка сервера для чистых мер"
stop_srv
nvidia-smi nvlink -s > "$OUT/nvlink-s.txt" 2>&1
nvidia-smi topo -m > "$OUT/topo-m.txt" 2>&1
say "Ф0: P2P по шести парам, буфер 256 МиБ"
"$D/bin/p2p-full-probe" 256 > "$OUT/p2p-full.txt" 2>&1; cat "$OUT/p2p-full.txt"
say "Ф0: хост-путь (поодиночке и все четыре сразу)"
"$D/bin/p2p-full-probe" 256 --host > "$OUT/host-bw.txt" 2>&1; cat "$OUT/host-bw.txt"
echo "Ф0_ЗАВЕРШЕНА"

# ---------------------------------------------------------------- Ф1: --peer-device с --layer-split
say "Ф1: проверка совместимости --peer-device и --layer-split"
mkvariant peerdev -- --peer-device 1 --peer-reserve-mib 1024
if start_srv "$CFG-peerdev.json" 120; then
    say "ОЖИДАНИЕ: движок принял --peer-device вместе с --layer-split"; probe
    stop_srv
else
    say "движок отказал (как и предсказывает generate.cpp:2716):"
    grep -iE "peer-device|cannot be combined" "$D/run/server-iq3_s.out" | tail -3 | tee "$OUT/peer-device-refusal.txt"
fi
rm -f "$D/Strata/$CFG-peerdev.json"
echo "Ф1_ЗАВЕРШЕНА"

# ---------------------------------------------------------------- Ф2: базовая конфигурация
start_srv "$CFG" || exit 1
say "движок: $(props)"; probe
measure baseline
echo "Ф2_ЗАВЕРШЕНА"

# ---------------------------------------------------------------- Ф3: помощник стадии (правило движка)
mkvariant help-on STRATA_PREFILL_HELP=1 --
stop_srv; start_srv "$CFG-help-on.json" || { say "не поднялся с STRATA_PREFILL_HELP"; }
say "движок: $(props)"; englog | tail -6; probe
measure help-on
echo "Ф3_ЗАВЕРШЕНА"

# ---------------------------------------------------------------- Ф4: помощник принудительно на длинных промптах
mkvariant help-forced STRATA_PREFILL_HELP=1 STRATA_PREFILL_HELP_FRAC=0.45 --
stop_srv; start_srv "$CFG-help-forced.json" || { say "не поднялся с HELP_FRAC"; }
say "движок: $(props)"; englog | tail -6; probe
measure help-forced
echo "Ф4_ЗАВЕРШЕНА"

# ---------------------------------------------------------------- Ф5: контроль на базовой конфигурации
rm -f "$D/Strata/$CFG-help-on.json" "$D/Strata/$CFG-help-forced.json"
stop_srv; start_srv "$CFG" || exit 1
say "движок (контроль): $(props)"; probe
measure control
echo "Ф5_ЗАВЕРШЕНА"

trap - EXIT
restore
echo "ВСЁ_ЗАВЕРШЕНО"
