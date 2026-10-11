#!/bin/bash
# After run.sh: one IQ3_S sweep on the 0.1.42 image, to separate a 0.1.42 -> 0.1.43 change from a machine change.
. "$HOME/strata-trial/r2/lib.sh"
while pgrep -f "run.sh iq3s ud swift" >/dev/null; do sleep 60; done
IMG_OLD=strata:v0.1.42
start_old b0142-iq3s qwen IQ3_S "$PROD_MODELS/IQ3_S" && wait_ready 10800 && snap iq3s-0142 && sweep iq3s-0142 sweep-iq3s-0142.jsonl
stop_bench
log "done: extra"
