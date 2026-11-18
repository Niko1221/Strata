#!/bin/bash
# Watchdog for gap/mtp-ab.sh: after the driver's pid exits, give the host up to 420 s to answer on 8180/8181
# (the run script may still be cold-loading); then clear leftover CLI strata and start the host once.
set -u
FORK=/Users/pc/src/Strata-fork
DPID=${1:?usage: mtp-ab-watch.sh <driver pid>}
while kill -0 "$DPID" 2>/dev/null; do sleep 30; done
for i in $(seq 1 14); do
  sleep 30
  for p in 8180 8181; do
    curl -sf -m 5 "http://127.0.0.1:$p/v1/models" 2>/dev/null | grep -q ud-q4_k_xl-huihui && { echo "host back on $p after ~$((i*30))s"; exit 0; }
  done
done
echo "watchdog: host still down after the driver exited; starting it"
pkill -f "build-metal/strata --pack /Users/pc/src/Strata-data/packs/ud-q4_k_xl-huihui" 2>/dev/null
sleep 5
"$FORK/tools/huihui_session.sh" start pre-mtp-ab
