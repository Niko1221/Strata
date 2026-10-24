#!/bin/bash
# UD-IQ4_XS on 0.2.0 with the RAM budget and the file-tier prompt-chunk plan off, to test whether the plan explains the slower prompt read.
. "$HOME/strata-trial/r2/lib.sh"
d=/var/lib/docker/volumes/b020-ud/_data/Strata-data/models/unsloth-UD-IQ4_XS
start_new b020-ud-budget-nohostplan unsloth UD-IQ4_XS "$d" -e STRATA_SET="resident-budget-gib=55 STRATA_PREFILL_HOST_PLAN=0" &&
  wait_ready 3600 && snap ud-budget-nohostplan && sweep ud-budget-nohostplan sweep-ud-budget-nohostplan.jsonl
docker run --rm -v b020-ud-budget-nohostplan:/d public.ecr.aws/docker/library/alpine:3 sh -c \
  'grep -iE "prefill|chunk|host plan|page cache" /d/state/strata/launch/docker.engine.log | head -20' > "$R2/out/ud-nohostplan-prefill-lines.txt"
docker run --rm -v b020-ud-budget:/d public.ecr.aws/docker/library/alpine:3 sh -c \
  'grep -iE "prefill|chunk|host plan|page cache" /d/state/strata/launch/docker.engine.log | head -20' > "$R2/out/ud-budget-prefill-lines.txt"
stop_bench
log "done: extra4"
