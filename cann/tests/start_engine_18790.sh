#!/bin/bash
# Start the reference endpoint on 18790 (our engine) and leave the fast one on 18789 alone.
set -euo pipefail
cd "$(dirname "$0")/../.."
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"

if pgrep -f "serve_engine.py" > /dev/null 2>&1; then
    echo "already running: $(pgrep -f serve_engine.py | tr '\n' ' ')"
    exit 0
fi
mkdir -p /data/strata-cann/logs
STRATA_BUDGET_GB="${STRATA_BUDGET_GB:-16}" nohup python3 "$(dirname "$0")/serve_engine.py" --port 18790 > /data/strata-cann/logs/engine-18790.log 2>&1 &
echo "pid $! -> /data/strata-cann/logs/engine-18790.log"
for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:18790/health > /dev/null 2>&1; then
        echo "healthy after ${i}s"
        curl -s http://127.0.0.1:18790/health
        exit 0
    fi
    sleep 1
done
echo "did not become healthy; last log lines:"
tail -20 /data/strata-cann/logs/engine-18790.log
exit 1
