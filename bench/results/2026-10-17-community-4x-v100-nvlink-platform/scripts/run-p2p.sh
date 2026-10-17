#!/usr/bin/env bash
# run-p2p.sh — Этап III (п.6): сборка и запуск расширенной P2P-пробы.
#
# Изолированность: бинарник и результаты — только внутри каталога проекта.
# Зависимости: приватный CUDA Toolkit 12.8 и приватный g++-14 (уже есть на машине);
# ничего не устанавливается, системные пакеты и службы не меняются.
#
# Запуск:
#   scripts/run-p2p.sh [--max-size 8M] [--note "…"] [доп. аргументы пробы]
# Результаты: raw/p2p/<UTC-штамп>/p2p-{matrix,benchmark}.csv, p2p-correctness.json
set -u

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
CU="$HOME/strata185/cuda-12.8/bin/nvcc"
GXX="$HOME/strata185/gcc185/usr/bin/g++-14"
SRC="$PROJ/source/p2p-probe-ext.cu"
BIN="$PROJ/build/p2p-probe-ext"
STAMP="$(date -u +%Y-%m-%dT%H%M%SZ)"
OUT="$PROJ/raw/p2p/$STAMP"
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
  echo "args: $*"
  echo "--- nvidia-smi до запуска ---"
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.current.sm --format=csv
} > "$OUT/conditions.txt"

"$BIN" --out "$OUT" "$@" 2>&1 | tee "$OUT/stdout.log"

{
  echo "--- nvidia-smi после запуска ---"
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.current.sm --format=csv
} >> "$OUT/conditions.txt"

echo "результаты: $OUT"
