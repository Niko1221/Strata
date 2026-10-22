#!/bin/zsh
# swap.sh <config.json>: stop the running Strata server, start the one in <config>, wait until it serves
cd <strata>
pkill -f "serve/server.py --engine strata" ; sleep 8
nohup .venv/bin/python serve/server.py --engine strata --config "$1" --port 8091 > <data>/dflash2/serve-$(basename $1 .json).out 2>&1 &
for i in $(seq 1 60); do sleep 5; curl -s -m 3 localhost:8091/v1/models | grep -q loaded && { echo ready; exit 0; }; done
echo "not ready"; tail -5 <data>/dflash2/serve-$(basename $1 .json).out; exit 1
