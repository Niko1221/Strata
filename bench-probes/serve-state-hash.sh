#!/bin/sh
# Which part of the state differs between request 1 and request N of ONE --serve process?
# The engine prints a STATE_HASH per request (gdn/ple/tail/pooled/dead/kv), so the differing component
# names the subsystem.  Ten identical greedy requests, no cache, adaptive tier left at its default.
set -e
cd /home/gostar/Strata
export LD_LIBRARY_PATH=/usr/local/cuda-13.4/bin:/usr/local/cuda-13.4/lib64:${LD_LIBRARY_PATH:-}
.venv/bin/python serve/server.py --engine strata --config strata-3090-frozenadapt.json --port 31337 \
  > /tmp/srv-hash.log 2>&1 &
SRV=$!
i=0
while [ $i -lt 60 ]; do
  grep -q 'ready:' /tmp/srv-hash.log 2>/dev/null && break
  sleep 2; i=$((i+2))
done
echo "server ready"
.venv/bin/python - <<'PY'
import hashlib, json, urllib.request
digests = []
for k in range(10):
    b = json.dumps({"model":"strata","messages":[{"role":"user","content":"In one sentence, what does a radix prefix cache do?"}],
                    "max_tokens":200,"temperature":0,"stream":False}).encode()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:31337/v1/chat/completions", data=b,
        headers={"Content-Type":"application/json"}), timeout=200).read())
    m = r["choices"][0]["message"]
    blob = (m.get("content") or "") + "|" + (m.get("reasoning_content") or "")
    d = hashlib.sha256(blob.encode()).hexdigest()[:12]
    digests.append(d)
    print("  request %2d: %3d tokens  text %s %s" % (k, r["usage"]["completion_tokens"], d,
          "" if k == 0 or d == digests[k-1] else "  <- CHANGED"))
print("distinct over 10 requests:", len(set(digests)))
PY
kill $SRV 2>/dev/null || true
