#!/bin/sh
# Fresh server, send a SHORT unrelated prompt first, then the target prompt.
#   - if the target matches the fresh-process FIRST request  -> the prompt's own content decides
#   - if it matches the steady-state later requests         -> something latches at warm-up
set -e
cd /home/gostar/Strata
export LD_LIBRARY_PATH=/usr/local/cuda-13.4/bin:/usr/local/cuda-13.4/lib64:${LD_LIBRARY_PATH:-}
.venv/bin/python serve/server.py --engine strata --config strata-3090-frozenadapt-res.json --port 31337 \
  > /tmp/srv-warm.log 2>&1 &
SRV=$!
i=0
while [ $i -lt 60 ]; do
  grep -q 'ready:' /tmp/srv-warm.log 2>/dev/null && break
  sleep 2; i=$((i+2))
done
echo "server ready"
.venv/bin/python - <<'PY'
import hashlib, json, urllib.request
def ask(text, max_tokens=200):
    b = json.dumps({"model":"strata","messages":[{"role":"user","content":text}],
                    "max_tokens":max_tokens,"temperature":0,"stream":False}).encode()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:31337/v1/chat/completions", data=b,
        headers={"Content-Type":"application/json"}), timeout=200).read())
    m = r["choices"][0]["message"]
    blob = (m.get("content") or "") + "|" + (m.get("reasoning_content") or "")
    return r["usage"]["completion_tokens"], hashlib.sha256(blob.encode()).hexdigest()[:12]

TARGET = "In one sentence, what does a radix prefix cache do?"
import hashlib as _h
src = open("/mnt/gostar_data/strata/bench/prompts/2k-2114.ids").read().strip().split(",")
OTHER = ",".join(src[:64])
_b = json.dumps({"model":"strata","messages":[{"role":"user","content":"count to fifty"}],
                 "max_tokens":64,"temperature":0,"stream":False}).encode()
_r = json.loads(urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:31337/v1/chat/completions", data=_b,
    headers={"Content-Type":"application/json"}), timeout=200).read())
print("  request 0 (different prompt):       %3d tokens  %s"
      % (_r["usage"]["completion_tokens"],
         _h.sha256((_r["choices"][0]["message"].get("content") or "").encode()).hexdigest()[:12]))
n, d = ask(TARGET)
print("  request 1 (TARGET, after warm-up):  %3d tokens  %s" % (n, d))
n, d = ask(TARGET)
print("  request 2 (TARGET, again):         %3d tokens  %s" % (n, d))
PY
kill $SRV 2>/dev/null || true