#!/usr/bin/env bash
# Run Strata's GPU-free Python tests one module at a time: tools/test_*.py (setup and tools, AGENTS.md) and
# serve/test_*.py (the server against a mock engine; they open local ports only).
# Usage: run_py_tests.sh OUTDIR    -> OUTDIR/py-tests/<module>.log and OUTDIR/py-tests/summary.tsv
set -u
OUT=$1/py-tests; mkdir -p "$OUT"
S=<WORKDIR>/experiments/03_run/Strata
PY=<WORKDIR>/experiments/03_run/.venv-setup/bin/python
cd "$S" || exit 1
# tools/test_setup_choices.py (HipVision) does not mock setup's ROCm install: with a package index reachable it
# pip-installs ~11 GB of ROCm into .venv-setup (seen 2026-10-16). No index: pip fails at once and nothing is installed.
export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
HB=<WORKDIR>/experiments/06_test/tools/hb.py
printf 'module\texit\tseconds\tresult\n' > "$OUT/summary.tsv"
set -- tools/test_*.py serve/test_*.py
n=$#; i=0
for f; do
  m=$(dirname "$f").$(basename "$f" .py)
  i=$((i + 1))
  t0=$(date +%s)
  # progress goes to stderr (the terminal); stdout stays the one-line result that t1_unit.sh saves
  # hb.py: a status line every 30 s while a module runs, a warning when it stalls 120 s, dies or times out
  python3 "$HB" --every 30 --stall 120 --label "$m" --out "$OUT/$m.log" -- \
    timeout 600 "$PY" -W ignore -m unittest "$m"
  rc=$?
  res=$(grep -E '^(OK|FAILED)' "$OUT/$m.log" | tail -1)
  secs=$(( $(date +%s) - t0 ))
  printf '%s\t%s\t%s\t%s\n' "$m" "$rc" "$secs" "${res:-?}" >> "$OUT/summary.tsv"
  printf '[%s] %3d/%d %-45s %4ss  %s\n' "$(date +%T)" "$i" "$n" "$m" "$secs" "${res:-exit $rc}" >&2
done
echo "python tests: $(awk -F'\t' 'NR>1 && $2==0' "$OUT/summary.tsv" | wc -l) passed, $(awk -F'\t' 'NR>1 && $2!=0' "$OUT/summary.tsv" | wc -l) failed (of $(( $(wc -l < "$OUT/summary.tsv") - 1 )))"
