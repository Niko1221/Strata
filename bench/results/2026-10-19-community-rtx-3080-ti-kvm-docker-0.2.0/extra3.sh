#!/bin/bash
# After extra2.sh: UD-IQ4_XS on 0.2.0 with the RAM budget 0.1.x setup chose (stock 0.2.0 was OOM-killed), interleaved with 0.1.43.
. "$HOME/strata-trial/r2/lib.sh"
while pgrep -f "run.sh iq3s ud swift|extra.sh|extra2.sh" >/dev/null; do sleep 60; done
d=/var/lib/docker/volumes/b020-ud/_data/Strata-data/models/unsloth-UD-IQ4_XS
B="resident-budget-gib=55"
for i in 1 2; do
  start_new b020-ud-budget unsloth UD-IQ4_XS "$d" -e STRATA_SET="$B" && wait_ready 3600 && snap "ud-budget-$i" &&
    sweep ud-budget sweep-ud-budget.jsonl
  [ "$i" = 1 ] && start_old b0143-ud unsloth UD-IQ4_XS "$d" && wait_ready 3600 && snap ud-old-3 && sweep ud-old sweep-ud-old.jsonl
done
log "tune ud-budget"
start_new b020-ud-budget-tuned unsloth UD-IQ4_XS "$d" -e STRATA_SET="$B" -e TUNE=quick && wait_ready 7200 &&
  snap ud-budget-tuned && sweep ud-budget-tuned sweep-ud-budget-tuned.jsonl
stop_bench
log "done: extra3"
