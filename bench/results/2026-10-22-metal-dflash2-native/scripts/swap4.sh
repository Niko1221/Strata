#!/bin/zsh
# swap4.sh <config.json>: stop only this script's last server (pid file), start <config> on port 8092, wait until it serves.
# The server is started with --config before --engine, so another session's `pkill -f "serve/server.py --engine strata"`
# does not match it; it also never kills another session's server.
cd <strata>
P=<data>/dflash2/bench/swap4.pid
[ -f $P ] && { kill $(cat $P) 2>/dev/null; sleep 8; }
nohup .venv/bin/python serve/server.py --config "$1" --engine strata --port 8092 > <data>/dflash2/serve4-$(basename $1 .json).out 2>&1 &
echo $! > $P
for i in $(seq 1 60); do sleep 5; curl -s -m 3 localhost:8092/v1/models | grep -q loaded && { echo ready; exit 0; }; done
echo "not ready"; tail -5 <data>/dflash2/serve4-$(basename $1 .json).out; exit 1
