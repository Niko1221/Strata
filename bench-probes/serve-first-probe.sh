#!/bin/sh
# Is the FIRST request of a fresh process reproducible?  Separates within-request nondeterminism from
# state that evolves across requests inside one server.
set -e
cd /home/gostar/Strata
export LD_LIBRARY_PATH=/usr/local/cuda-13.4/bin:/usr/local/cuda-13.4/lib64:${LD_LIBRARY_PATH:-}
N=${1:-1}
.venv/bin/python serve/server.py --engine strata --config strata-3090-iq4nl-nocache.json --port 31337 \
  > /tmp/srv-first.log 2>&1 &
SRV=$!
i=0
while [ $i -lt 60 ]; do
  grep -q 'ready:' /tmp/srv-first.log 2>/dev/null && break
  sleep 2; i=$((i+2))
done
.venv/bin/python - "$N" <<'PY'
import hashlib, json, sys, urllib.request
n = int(sys.argv[1])
for k in range(n):
    b = json.dumps({"model":"strata","messages":[{"role":"user","content":"In one sentence, what does a radix prefix cache do?"}],
                    "max_tokens":200,"temperature":0,"stream":False}).encode()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:31337/v1/chat/completions", data=b,
        headers={"Content-Type":"application/json"}), timeout=200).read())
    m = r["choices"][0]["message"]
    blob = (m.get("content") or "") + "|" + (m.get("reasoning_content") or "")
    print("  request %d: %d tokens, digest %s" % (k, r["usage"]["completion_tokens"],
                                                 hashlib.sha256(blob.encode()).hexdigest()[:16]))
PY
kill $SRV 2>/dev/null || true
