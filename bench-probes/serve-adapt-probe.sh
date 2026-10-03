#!/bin/sh
# Does the serve loop's adaptive tier fire at all, with the trace reaching the engine?
set -e
cd /home/gostar/Strata
export LD_LIBRARY_PATH=/usr/local/cuda-13.4/bin:/usr/local/cuda-13.4/lib64:${LD_LIBRARY_PATH:-}
export STRATA_TRACE_ADAPT=1
.venv/bin/python serve/server.py --engine strata --config strata-3090-iq4nl-nocache.json --port 31337 \
  > /tmp/srv-probe.log 2>&1 &
SRV=$!
i=0
while [ $i -lt 60 ]; do
  grep -q 'ready:' /tmp/srv-probe.log 2>/dev/null && break
  sleep 2; i=$((i+2))
done
echo "server ready: $(grep -c 'ready:' /tmp/srv-probe.log)"
EPID=$(pgrep -f 'engine/strata --serve' | head -n 1)
echo "engine pid $EPID; STRATA_TRACE_ADAPT in its env: $(tr '\0' '\n' < /proc/$EPID/environ 2>/dev/null | grep -c STRATA_TRACE_ADAPT)"
.venv/bin/python - <<'PY'
import json, urllib.request
b = json.dumps({"model":"strata","messages":[{"role":"user","content":"In one sentence, what does a radix prefix cache do?"}],
                "max_tokens":200,"temperature":0,"stream":False}).encode()
r = json.loads(urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:31337/v1/chat/completions", data=b,
    headers={"Content-Type":"application/json"}), timeout=200).read())
print("generated", r["usage"]["completion_tokens"], "tokens")
PY
echo "--- adapt/pending trace lines from this run ---"
grep -E 'SERVE ADAPT|ADAPT|PENDING' strata-iq3_xxs.log | tail -n 12 || echo "(none)"
kill $SRV 2>/dev/null || true
