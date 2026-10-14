#!/usr/bin/env bash
# Same-day comparison for an upstream community report (docs/COMMUNITY_BENCHMARKS.md): our production engine + server
# (0.1.43 + patches, 16 slots) against stock 0.1.44 (8 slots, its own serve/server.py), both text only, same requests.
# Published copy: the folders are env variables, the config names match ../configs/, and a production service is
# stopped (and started again on exit, also on Ctrl-C) only when PROD_UNIT names a systemd --user unit. The run that
# produced this report set PROD_UNIT=strata.service.
#   STRATA=<Strata folder with .venv> OURS_DIR=<server checkout> STOCK_DIR=<server checkout> community_bench_144.sh
#   ARMS=stock ... community_bench_144.sh   one arm only (ours|stock)
# The configs' "exe", "--pack", "--native", "--mtp" and "--expert-profile" paths must be set for your machine first.
# Per arm: short prompts (100 tokens) at 1, 8 and 16 users, 1.2K prompts at 16 users (2 x 512 tokens per user,
# greedy), then 16 agents x 3 turns multi-turn (capped at MT_MAX_S seconds). Results: results/sweep.jsonl,
# results/multiturn.jsonl and results/<label>.txt with labels cb144-<arm>-{short,long,mt}.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
TAG=${TAG:-cb144}
ARMS=${ARMS:-ours stock}
MT_MAX_S=${MT_MAX_S:-240}
export PORT=${PORT:-8095} BIND=127.0.0.1 MEMMAX=${MEMMAX:-224G}
export STRATA=${STRATA:?set STRATA to a Strata folder with its .venv}
PY=$STRATA/.venv/bin/python
CONFIGS=${CONFIGS:-../configs}
PROD_UNIT=${PROD_UNIT:-}

back_to_production() {
  systemctl --user stop strata-bench.scope 2>/dev/null
  if [ -n "$PROD_UNIT" ]; then
    systemctl --user start "$PROD_UNIT"
    echo "[$TAG] $PROD_UNIT started again $(date +%T)"
  fi
}
trap back_to_production EXIT

if [ -n "$PROD_UNIT" ]; then
  echo "[$TAG] $PROD_UNIT down $(date +%T)"
  systemctl --user stop "$PROD_UNIT"
fi

for arm in $ARMS; do
  case $arm in
    ours)  cfg=$CONFIGS/ours-CB144-ours-prod-text.json;    srv=${OURS_DIR:?set OURS_DIR to the modified server checkout} ;;
    stock) cfg=$CONFIGS/stock-CB144-stock-0.1.44-p8.json;  srv=${STOCK_DIR:?set STOCK_DIR to the v0.1.44 checkout} ;;
    *) echo "unknown arm $arm"; continue ;;
  esac
  t0=$(date +%s)
  SERVER_DIR=$srv KEEP=1 ./run_strata_variant.sh "$TAG-$arm-short" "$cfg" 1,8,16 \
    --rounds 2 --max-tok 512 --prompt-tok 100 --temperature 0 2>&1 | tail -6
  SERVER_DIR=$srv ATTACH=1 KEEP=1 ./run_strata_variant.sh "$TAG-$arm-long" "$cfg" 16 \
    --rounds 2 --max-tok 512 --temperature 0 --warmup 0 2>&1 | tail -3
  timeout "$MT_MAX_S" "$PY" -I bench_multiturn.py --url "http://127.0.0.1:$PORT" --label "$TAG-$arm-mt" \
    --users 16 --turns 3 --out results/multiturn.jsonl 2>&1 | tail -4
  systemctl --user stop strata-bench.scope
  echo "[$TAG] arm $arm took $(( $(date +%s) - t0 )) s"
done
