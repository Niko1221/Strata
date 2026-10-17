#!/usr/bin/env bash
# run-p2p-full-detached.sh — Фаза B Этапа III: полный ряд размеров и конкуренция пар.
#
# Требует остановки Strata: с загруженной моделью на трёх картах свободно меньше 400 MiB,
# и пары с GPU3 не измерить вовсе (на последнем снимке free = 0 MiB).
#
# ВАЖНО: Strata на этой машине обслуживает и сеанс агента. Скрипт останавливает сервер,
# выполняет замер и ОБЯЗАТЕЛЬНО возвращает его в рабочее состояние (trap + явный пуск
# штатным скриптом + ожидание /health). Работает отсоединённо (setsid nohup), поэтому
# завершится корректно даже если сессия, которая его запустила, прервётся.
#
# Запуск:
#   setsid nohup ~/strata185/'Strata+4xV100+NVlink'/scripts/run-p2p-full-detached.sh \
#       >> ~/strata185/'Strata+4xV100+NVlink'/raw/p2p/detached.log 2>&1 &
set -u

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
D="$HOME/strata185"
PAT='strata185/Strata/(.venv/bin/python|engine-cuda12/)'
STAMP="$(date -u +%Y-%m-%dT%H%M%SZ)"
OUT="$PROJ/raw/p2p/$STAMP"
mkdir -p "$OUT"
say(){ echo "[$(date -u +%FT%TZ)] $*"; }

restore() {
    say "восстановление рабочего запуска"
    if ! pgrep -f "$PAT" >/dev/null 2>&1; then
        # порт должен быть свободен: иначе сервер умрёт с «port 8080 is already in use»,
        # а /health продолжит отвечать от старой обёртки без движка
        HOLD=$(ss -ltnp 2>/dev/null | grep ':8080 ' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
        if [ -n "${HOLD:-}" ]; then say "порт 8080 удерживает pid $HOLD — снимаю"; kill -TERM "$HOLD" 2>/dev/null; sleep 10; fi
        rm -f "$D/run/server-iq3_s.pid"
        "$D/bin/start-strata-iq3_s.sh"
    fi
    for i in $(seq 1 90); do
        code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health 2>/dev/null)
        [ "$code" = "200" ] && break
        sleep 5
    done
    # Проверка по трём признакам (R6): /health сам по себе не отличает рабочий сервер от
    # обёртки с мёртвым движком. Нужны: живые процессы движка, загруженная модель (VRAM) и
    # успешный реальный запрос.
    ok_eng=0; pgrep -f 'engine-cuda12/strata' >/dev/null 2>&1 && ok_eng=1
    MINUSED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | head -1)
    ok_vram=0; [ "${MINUSED:-0}" -ge 14000 ] && ok_vram=1
    KEY=$(python3 -c "import json;print(json.load(open('$D/Strata/strata-185-iq3_s.json')).get('api_key',''))" 2>/dev/null)
    reply=$(curl -s -m 40 -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
        -d '{"model":"qwen3.8-flash-next-iq3_s","messages":[{"role":"user","content":"1"}],"max_tokens":1,"temperature":0,"reasoning_effort":"none"}' \
        http://127.0.0.1:8080/v1/chat/completions 2>/dev/null)
    ok_api=0; case "$reply" in *'"choices"'*) ok_api=1;; esac
    say "проверка возврата: health=$code, процессы движка=$ok_eng, минимальная VRAM на карту=${MINUSED:-?} MiB (нужно ≥14000)=$ok_vram, реальный запрос=$ok_api"
    if [ "$code" = "200" ] && [ "$ok_eng" = 1 ] && [ "$ok_vram" = 1 ] && [ "$ok_api" = 1 ]; then
        say "сервер снова готов полностью"; return 0
    fi
    say "ВНИМАНИЕ: сервер восстановлен НЕ полностью. Проверить вручную: journalctl / $D/Strata/strata-185-iq3_s.log, $D/run/server-iq3_s.out"
    return 1
}
trap 'ec=$?; say "выход, код=$ec"; restore' EXIT

say "состояние до остановки"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv | tee "$OUT/before.csv"
# 1. Обновить pid-файл фактическим pid обёртки :8080. Иначе stop-strata-iq3_s.sh не найдёт обёртку
#    (её cmdline может быть относительным, например ".venv/bin/python serve/server.py", если сервер поднят
#    ~/opt/v100-skinny/strata-resume.sh), снимет только движок, а обёртка через 15 с перезапустит движок
#    (serve/server.py: restart(), RESTART_RETRY_S=15) и заберёт VRAM прямо во время пробы.
SRV_PID=$(ss -ltnp 2>/dev/null | grep ':8080 ' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
if [ -n "${SRV_PID:-}" ]; then
    OLD_PID=$(cat "$D/run/server-iq3_s.pid" 2>/dev/null || echo -)
    [ "$SRV_PID" != "$OLD_PID" ] && { echo "$SRV_PID" > "$D/run/server-iq3_s.pid"; say "pid-файл обновлён: $OLD_PID -> $SRV_PID (обёртка :8080)"; }
else
    say "обёртка :8080 по порту не найдена — stop будет работать по старому pid-файлу/шаблону pgrep"
fi
"$D/bin/stop-strata-iq3_s.sh" | tee "$OUT/stop.log"
sleep 5
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee "$OUT/after-stop.csv"
# 2. Убедиться, что обёртка действительно ушла: если порт 8080 всё ещё занят — снять вручную (SIGTERM),
#    иначе новый сервер не сможет забиндиться («port 8080 is already in use»), а /health будет отвечать
#    от обёртки с мёртвым движком (зафиксировано 2026-10-17T00:12Z).
LEFT=$(ss -ltnp 2>/dev/null | grep ':8080 ' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
if [ -n "${LEFT:-}" ]; then
    say "порт 8080 всё ещё занят (pid $LEFT) — снимаю вручную"
    kill -TERM "$LEFT" 2>/dev/null; sleep 10
    LEFT2=$(ss -ltnp 2>/dev/null | grep ':8080 ' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
    [ -n "${LEFT2:-}" ] && say "ВНИМАНИЕ: порт 8080 по-прежнему занят (pid $LEFT2) — проверять вручную"
fi
# 3. Убедиться, что VRAM действительно освободилась: без этого проба на 256 MiB бессмысленна.
MINFREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
if [ "${MINFREE:-0}" -lt 600 ]; then
    say "ВНИМАНИЕ: минимальная свободная VRAM ${MINFREE} MiB < 600 MiB — ряд до 256 MiB не снять;
         проба пропустит крупные размеры (причина попадёт в CSV). Продолжаю, но результат будет частичным."
fi

say "полный замер P2P (ряд 4 KiB … 256 MiB, все пары, оба направления, конкуренция)"
"$PROJ/scripts/run-p2p.sh" --max-size 256M \
    --note "Фаза B: Strata остановлена, VRAM свободна; полный ряд размеров и конкуренция шести пар" \
    2>&1 | tee "$OUT/probe.log"

# run-p2p.sh пишет в свой каталог со своим штампом — перенесём указатель
LATEST="$(ls -1d "$PROJ"/raw/p2p/*/ 2>/dev/null | tail -1)"
echo "результаты пробы: $LATEST" > "$OUT/points-to.txt"

say "замер завершён"
