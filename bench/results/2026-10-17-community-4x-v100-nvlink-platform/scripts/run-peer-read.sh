#!/usr/bin/env bash
# run-peer-read.sh — Этап VIII (п. 11) / остаток Этапа III: зонд прямого чтения ядром
# памяти, резидентной для другой карты (нужен для выбора между P2a и P2b).
#
# Изолированность: бинарник и результаты — только внутри каталога проекта; ничего не
# устанавливается; компиляция приватными nvcc 12.8 и g++-14 (решение Q3).
# Ограничение R12: по умолчанию используются GPU0..GPU2 — к GPU3 внешний CUDA-контекст
# при работающем сервисе не подключается. Пары с GPU3 — только в окне остановки
# (--devices 0,1,2,3).
#
# Запуск:
#   scripts/run-peer-read.sh [--max-size 128M] [--devices 0,1,2] [--reps 5] [прочие аргументы пробы]
# Результаты: raw/peer-read/<UTC-штамп>/{peer-read-benchmark.csv,peer-read-summary.json,
#                                       peer-read-correctness.json,conditions.txt,build.log}
set -u

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
CU="$HOME/strata185/cuda-12.8/bin/nvcc"
GXX="$HOME/strata185/gcc185/usr/bin/g++-14"
SRC="$PROJ/source/peer-read-probe.cu"
BIN="$PROJ/build/peer-read-probe"
STAMP="$(date -u +%Y-%m-%dT%H%M%SZ)"
OUT="$PROJ/raw/peer-read/$STAMP"
mkdir -p "$OUT" "$PROJ/build"

[ -x "$CU" ]  || { echo "nvcc не найден: $CU"; exit 1; }
[ -x "$GXX" ] || { echo "g++-14 не найден: $GXX"; exit 1; }

echo "== сборка (приватный тулкит, -arch=sm_70; ничего системного не меняется)"
"$CU" -ccbin "$GXX" -arch=sm_70 -O2 -Wno-deprecated-gpu-targets -o "$BIN" "$SRC" 2>&1 | tee "$OUT/build.log"
[ -x "$BIN" ] || { echo "сборка не удалась"; exit 1; }
echo "binary sha256: $(sha256sum "$BIN" | cut -d' ' -f1)"

# условия замера фиксируем ДО запуска
{
  echo "captured_utc: $(date -u +%FT%TZ)"
  echo "binary: $BIN"
  echo "binary_sha256: $(sha256sum "$BIN" | cut -d' ' -f1)"
  echo "source_sha256: $(sha256sum "$SRC" | cut -d' ' -f1)"
  echo "compiler: $("$CU" --version | tail -1)"
  echo "host_cc: $("$GXX" --version | head -1)"
  echo "strata_running: $(pgrep -f 'engine-cuda12/strata' >/dev/null && echo YES || echo NO)"
  echo "free_vram_mib: $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr '\n' ' ')"
  echo "args: $*"
  echo "--- nvidia-smi до запуска ---"
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.current.sm --format=csv
} > "$OUT/conditions.txt"

"$BIN" --out "$OUT" "$@" > "$OUT/stdout.log" 2>&1 &
PROBE_PID=$!
# параллельно пишем часы/загрузку SM: короткие ядра на простаивающей карте выполняются на
# пониженных частотах, без этой записи полосу нельзя корректно интерпретировать
nvidia-smi --loop=2 --query-gpu=timestamp,index,clocks.current.sm,utilization.gpu,memory.used \
    --format=csv,noheader > "$OUT/clocks.csv" 2>/dev/null &
MON_PID=$!
wait $PROBE_PID; rc=$?
kill $MON_PID 2>/dev/null
cat "$OUT/stdout.log"

{
  echo "--- nvidia-smi после запуска ---"
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.current.sm --format=csv
  echo "exit_code: $rc"
  echo "часы SM во время пробы (min/median/max по строкам clocks.csv):"
  awk -F, 'NF>=4 {print $3}' "$OUT/clocks.csv" 2>/dev/null | sort -n | \
    awk '{v[NR]=$1} END {if (NR) printf "  n=%d min=%s med=%s max=%s\n", NR, v[1], v[int((NR+1)/2)], v[NR]; else print "  пусто"}'
} >> "$OUT/conditions.txt"

echo "результаты: $OUT (exit $rc)"
exit "$rc"
