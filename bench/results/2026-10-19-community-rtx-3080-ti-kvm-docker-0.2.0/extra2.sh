#!/bin/bash
# After extra.sh: IQ3_S on 0.2.0 with the settings planned for the lab endpoint (16 workers, conversation cache).
. "$HOME/strata-trial/r2/lib.sh"
while pgrep -f "run.sh iq3s ud swift|extra.sh" >/dev/null; do sleep 60; done
start_new b020-iq3s-prod qwen IQ3_S "$PROD_MODELS/IQ3_S" \
  -e STRATA_SET="pool-workers=16 conversation-cache-mib=8192 conversation-cache-slots=4" &&
  wait_ready 3600 && snap iq3s-prodcand && sweep iq3s-prodcand sweep-iq3s-prodcand.jsonl
stop_bench
log "done: extra2"
