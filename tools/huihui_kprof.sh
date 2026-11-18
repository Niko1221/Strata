#!/bin/bash
# tools/huihui_kprof.sh - one KPROF arm on the live Huihui host (answers corrupted; restore after).
# Restarts the engine with STRATA_METAL_KPROF=1, runs one short request, captures the class table, restarts back.
set -euo pipefail
cd /Users/pc/src/Strata-fork
CFG=strata-ud-q4_k_xl-huihui.json
cp "$CFG" /tmp/huihui_cfg.bak
trap 'cp /tmp/huihui_cfg.bak "$CFG"' EXIT
python3 - "$CFG" <<'EOF'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
d["env"]["STRATA_METAL_KPROF"] = "-1"
json.dump(d, open(p, "w"), indent=2)
EOF
tools/huihui_session.sh restart huihui || true
# one short request to trigger window profiling (window 20)
python3 - <<'EOF'
import json, urllib.request
body = {"model": "m", "messages": [{"role": "user", "content": "Count from 1 to 40, one number per line. No preamble."}], "temperature": 0, "max_tokens": 300, "thinking_budget": 0}
req = urllib.request.Request("http://127.0.0.1:8180/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
import time
for attempt in range(6):
    try:
        raw = urllib.request.urlopen(req, timeout=600).read().decode()
        j = json.loads(raw)
        if "choices" not in j:
            print("probe resp:", raw[:200]); time.sleep(15); continue
        print((j["choices"][0]["message"].get("content") or "")[:80]); break
    except Exception as e:
        print("probe err:", e); time.sleep(15)
EOF
sleep 1
grep -h "strata metal kprof" strata-ud_q4_k_xl-huihui.log | tail -40 || echo "(no kprof lines in log)"
tools/huihui_session.sh restart huihui
echo KPROF_ARM_DONE
