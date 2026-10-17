#!/usr/bin/env bash
# ab-run.sh — Этап VIII (п. 11 задания): A/B-стенд «до/после патча» для Strata на 4xV100.
#
# Протокол — §9 документа patch-recommendations.md:
#   manifest.json (sha256 бинарника, конфига, скриптов) -> ячейки матрицы 2x2
#   («auto»/закреплённое разбиение x «бинарник базовый»/«бинарник с патчем») в порядке
#   A0 -> A1 -> B0 -> B1 -> повтор базы (A0r: задаёт шумовой порог прогона и ловит
#   плавучесть auto) -> возврат рабочего запуска -> сводный отчёт.
# Каждая ячейка: старт варианта -> проверка возврата по трём признакам (R6) -> транспортная
# побитовая приёмка (до бенчмарка!) -> прогрев и 3 повтора на 4k/32k/125k с фиксированной
# длиной вывода (R9, R11) -> извлечение счётчиков движка -> остановка.
#
# Ничего не перезаписывается: результаты — raw/ab/<UTC-штамп>-<метка>/.
# Рабочая конфигурация и рабочий запуск не меняются: варианты конфига пишутся в отдельный
# файл strata-185-iq3_s.ab-<ячейка>.json и удаляются в конце; вариант сервиса запускается
# тем же способом, что и основной (абсолютный путь к python — правило R10).
#
# Запуск (длинный, ~40 мин с окном остановки):
#   cd ~/strata185/'Strata+4xV100+NVlink'
#   setsid nohup scripts/ab-run.sh > raw/ab/run-$(date -u +%H%M).log 2>&1 &
#
# usage: ab-run.sh [--arms "A0:auto:base A1:auto:patched B0:split12:base B1:split12:patched A0r:auto:base"]
#                  [--lengths 4k,32k,125k] [--reps 3] [--patched-binary PATH] [--label МЕТКА]
#                  [--skip-text] [--skip-bench] [--restore-only] [--text-targets 1024,8192]
set -u
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
D="$HOME/strata185"
CFG="strata-185-iq3_s.json"
PY="$D/Strata/.venv/bin/python"
ENGINE="$D/Strata/engine-cuda12/strata"
LOGF="$D/Strata/strata-185-iq3_s.log"
PIDF="$D/run/server-iq3_s.pid"
PAT='strata185/Strata/(.venv/bin/python|engine-cuda12/)'
SPLIT_PIN="12,25,37"
ARMS="A0:auto:base A1:auto:patched B0:split12:base B1:split12:patched A0r:auto:base"
LENGTHS="4k,32k,131072"
REPS=3
PATCHED=""
LABEL="validation"
SKIP_TEXT=0; SKIP_BENCH=0; RESTORE_ONLY=0
TEXT_TARGETS="1024,8192"

while [ $# -gt 0 ]; do
  case "$1" in
    --arms) ARMS="$2"; shift 2;;
    --lengths) LENGTHS="$2"; shift 2;;
    --reps) REPS="$2"; shift 2;;
    --patched-binary) PATCHED="$2"; shift 2;;
    --label) LABEL="$2"; shift 2;;
    --skip-text) SKIP_TEXT=1; shift;;
    --skip-bench) SKIP_BENCH=1; shift;;
    --restore-only) RESTORE_ONLY=1; shift;;
    --text-targets) TEXT_TARGETS="$2"; shift 2;;
    *) echo "неизвестный аргумент: $1"; exit 2;;
  esac
done

STAMP="$(date -u +%Y-%m-%dT%H%M%SZ)"
OUT="$PROJ/raw/ab/$STAMP-$LABEL"
mkdir -p "$OUT"
say(){ echo "[$(date -u +%FT%TZ)] $*"; }

sync_pidfile(){
  local p
  p=$(ss -ltnp 2>/dev/null | grep ':8080 ' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  if [ -n "$p" ]; then echo "$p" > "$PIDF"; say "pid-файл обновлён фактическим pid обёртки: $p"; else
    say "ВНИМАНИЕ: процесс на :8080 не найден, pid-файл не обновлён"; fi
}

stop_srv(){
  sync_pidfile
  say "остановка сервиса"
  bash "$D/bin/stop-strata-iq3_s.sh" > "$OUT/stop-$(date -u +%H%M%S).txt" 2>&1
  local born=$(date +%s)
  while pgrep -f "$PAT" >/dev/null 2>&1; do
    if [ $(( $(date +%s) - born )) -gt 180 ]; then say "через 180 с живы — добиваю TERM"
      for p in $(pgrep -f "$PAT" 2>/dev/null); do kill -TERM "$p" 2>/dev/null; done; sleep 20; break; fi
    sleep 5
  done
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
}

start_srv(){   # $1 = имя конфига в каталоге Strata (тот же способ, что у штатного скрипта — R10)
  local cfg="$1"
  cd "$D/Strata" || return 1
  export LD_LIBRARY_PATH="$D/cuda-12.8/lib64"
  nohup "$PY" serve/server.py --engine strata --config "$cfg" --port 8080 \
      > "$D/run/server-iq3_s.out" 2>&1 &
  echo $! > "$D/run/server-iq3_s.pid"
  say "запуск с конфигом $cfg, pid обёртки $!"
}

restore(){
  say "ВОЗВРАТ РАБОЧЕЙ КОНФИГУРАЦИИ ($CFG)"
  stop_srv
  start_srv "$CFG"
  python3 "$PROJ/scripts/ab-ready.py" --timeout 900 | tail -3
  python3 "$PROJ/scripts/correctness-integration.py" 2>/dev/null | tail -2 || true
}

run_cell(){
  local name="$1" split="$2" binarm="$3"
  local dir="$OUT/$name"; mkdir -p "$dir"
  local bin="$ENGINE"
  [ "$binarm" = "patched" ] && [ -n "$PATCHED" ] && bin="$PATCHED"
  say "=== ячейка $name: split=$split, бинарник=$(basename "$bin") ($([ "$bin" = "$ENGINE" ] && echo базовый || echo патч))"
  local logb0=0; [ -f "$LOGF" ] && logb0=$(stat -c %s "$LOGF")
  echo "$logb0" > "$dir/log-offset.txt"
  local cfgname="strata-185-iq3_s.ab-$name.json"
  local args=(--out "$D/Strata/$cfgname" --name "$name")
  [ "$split" != "auto" ] && args+=(--split "$SPLIT_PIN")
  [ "$bin" != "$ENGINE" ] && args+=(--engine "$bin")
  python3 "$PROJ/scripts/ab-mkconfig.py" "${args[@]}" | tee "$dir/config.txt"
  cp "$D/Strata/$cfgname" "$dir/config.json"
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.current.sm \
      --format=csv > "$dir/nvidia-smi-before.csv"
  stop_srv
  start_srv "$cfgname"
  python3 "$PROJ/scripts/ab-ready.py" --timeout 900 > "$dir/ready.json" 2>&1
  local rok=$?
  tail -2 "$dir/ready.json"
  if [ $rok -ne 0 ]; then say "ЯЧЕЙКА $name: сервис не поднялся по трём признакам — ячейка помечена FAILED"
    echo "FAILED: сервис не поднялся" > "$dir/verdict.txt"; rm -f "$D/Strata/$cfgname"; return 1; fi
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.current.sm \
      --format=csv > "$dir/nvidia-smi-after-start.csv"
  # идентичность сборки на момент ячейки (manifest строится при выключенном сервисе -> /props недоступен)
  python3 - "$dir/props.json" <<'PY' 2>/dev/null || true
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path.home() / "strata185/Strata+4xV100+NVlink/scripts"))
import importlib
m = importlib.import_module("ab-manifest")
out = pathlib.Path(sys.argv[1]); out.write_text(json.dumps(m.props_build_info(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("props:", json.dumps(m.props_build_info(), ensure_ascii=False)[:160])
PY
  # 1) побитовая приёмка ДО бенчмарка (§9.3)
  local tc=0
  if [ "$SKIP_TEXT" = "0" ]; then
    local refopt=()
    [ -n "${PREV_TEXT_JSON:-}" ] && [ -f "$PREV_TEXT_JSON" ] && refopt=(--ref "$PREV_TEXT_JSON")
    # валидация метода (патча нет): несовпадение с эталоном предыдущей ячейки — предмет измерения,
    # а не причина отменить бенчмарк; при патче остаётся строгим FAIL (§9.3)
    [ -z "$PATCHED" ] && refopt+=(--allow-ref-mismatch)
    python3 "$PROJ/scripts/ab-text-check.py" --out "$dir" --targets "$TEXT_TARGETS" "${refopt[@]}" \
        > "$dir/text-check.log" 2>&1
    tc=$?
    tail -3 "$dir/text-check.log"
    PREV_TEXT_JSON="$dir/text-check.json"
  fi
  # 2) бенчмарк (только если приёмка пройдена)
  if [ "$SKIP_BENCH" = "0" ] && [ $tc -eq 0 ]; then
    python3 "$PROJ/scripts/baseline-profile.py" --out "$dir/bench" --lengths "$LENGTHS" --reps "$REPS" \
        > "$dir/bench.log" 2>&1
    say "бенчмарк $name: exit $?"
    tail -4 "$dir/bench.log"
  else
    say "бенчмарк $name пропущен (приёмка не пройдена или --skip-bench)"
    echo "BENCH SKIPPED: text-check exit=$tc" > "$dir/verdict.txt"
  fi
  # 3) счётчики движка за время ячейки
  python3 "$PROJ/scripts/ab-extract-log.py" --log "$LOGF" --out "$dir" --from-byte "$logb0" | tee "$dir/counters-summary.txt"
  echo "cell=$name split=$split binary=$bin text_check_exit=$tc" > "$dir/verdict.txt"
  rm -f "$D/Strata/$cfgname"
  stop_srv
}

# ------------------------------------------------------------------ основной ход
say "A/B-стенд: каталог $OUT"
python3 "$PROJ/scripts/ab-manifest.py" --out "$OUT" --label "$LABEL" \
    --note "arms=$ARMS lengths=$LENGTHS reps=$REPS patched=${PATCHED:-<нет: валидация метода на базовом бинарнике>}" \
    2>&1 | tee "$OUT/manifest.log"
if [ "$RESTORE_ONLY" = "1" ]; then restore; echo "АБ_СТЕНД_ВОССТАНОВЛЕНИЕ_ЗАВЕРШЕНО"; exit 0; fi
trap 'say "аварийный выход: возвращаю рабочую конфигурацию"; restore' EXIT

echo "$ARMS" | tr ' ' '\n' > "$OUT/arms-order.txt"
for arm in $ARMS; do
  IFS=: read -r name split binarm <<< "$arm"
  run_cell "$name" "$split" "$binarm"
done

rm -f "$OUT"/stop-*.txt.tmp
trap - EXIT
restore
python3 "$PROJ/scripts/ab-report.py" --out "$OUT" 2>&1 | tee "$OUT/report.log"
echo "АБ_СТЕНД_ЗАВЕРШЁН: $OUT"
