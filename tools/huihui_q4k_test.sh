#!/bin/bash
# Huihui-Q4_K_XL side-by-side test harness (does NOT touch the IQ3_XXS server on 8180).
#
#   tools/huihui_q4k_test.sh start    start the Huihui serve (port 8181), wait until the engine is loaded
#   tools/huihui_q4k_test.sh probe    run the probe set against 8181;  add --both to repeat against 8180 (IQ3_XXS)
#   tools/huihui_q4k_test.sh status   engine log's sizing lines + swap/disk snapshot
#   tools/huihui_q4k_test.sh stop     stop only the server THIS harness started
#   tools/huihui_q4k_test.sh all      start + probe + status (the server stays up)
#
# The model streams its experts from the SSD on this 64 GB M1 Max; a probe while the IQ3_XXS engine is also
# serving measures BOTH sharing one drive and one GPU - for clean numbers stop one of them first (stop only
# closes 8181, never 8180).
set -u
REPO="/Users/pc/src/Strata-fork"
PY="$REPO/.venv/bin/python"
CFG="$REPO/strata-ud-q4_k_xl-huihui.json"
LOG="$REPO/strata-ud-q4_k_xl-huihui.log"
PORT=8181
PIDFILE="/tmp/strata-huihui-serve.pid"
PACK="/Users/pc/src/Strata-data/packs/ud-q4_k_xl-huihui"
GGUF="/Users/pc/src/Strata-data/metal/ud-q4_k_xl-huihui-metal.gguf"

# --coexist (anywhere in the command line): run alongside the 8180 engine, keeping the arena small and unwired.
COEXIST=0
for _a in "$@"; do [ "$_a" = "--coexist" ] && COEXIST=1; done

say() { printf '%s\n' "$*"; }

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

start() {
  [ -f "$GGUF" ] || { say "missing $GGUF (run metal_repack first)"; exit 1; }
  [ -f "$PACK/index.txt" ] || { say "missing $PACK/index.txt (run tools/iq_pack.py first)"; exit 1; }
  if [ -f "$PIDFILE" ] && ! running; then rm -f "$PIDFILE"; fi
  if curl -sf -m 3 http://127.0.0.1:8180/v1/models >/dev/null 2>&1 && [ "$COEXIST" != 1 ]; then
    say "the IQ3_XXS engine on 8180 is live, and Huihui adds a ~41 GB resident footprint whose 30 GB"
    say "arena is WIRED (mlock) - macOS cannot page it down. Both engines on a 64 GB Mac starve:"
    say "page-fault stalls (low GPU use) and compressor/swap ping-pong. For clean numbers stop 8180"
    say "first (your call), or re-run with --coexist: Huihui's arena then stays 12 GB and unwired."
    exit 1
  fi
  if [ "$COEXIST" = 1 ]; then
    export STRATA_METAL_ARENA_GB=12 STRATA_METAL_WIRE_ARENA=0
    say "coexist mode: 12 GB unwired arena (low expert-cache hit rate; slower decode expected)"
  fi
  running && { say "already running (pid $(cat "$PIDFILE"))"; return 0; }
  : > "$LOG"
  nohup "$PY" "$REPO/serve/server.py" --engine strata --config "$CFG" --port "$PORT" >> "$LOG" 2>&1 &
  echo $! > "$PIDFILE"
  say "serve pid $(cat "$PIDFILE"), log $LOG"
  say "loading the engine (first start maps the GGUF and fills the arena; may take a few minutes) ..."
  for i in $(seq 1 180); do
    sleep 5
    if curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      say "ready after $((i*5)) s"
      grep -E "strata metal stream|prompts on the matrix units|arena|hit rate" "$LOG" | tail -8
      return 0
    fi
    running || { say "the serve died; last log lines:"; tail -15 "$LOG"; exit 1; }
  done
  say "not ready after 900 s; last log lines:"; tail -25 "$LOG"; exit 1
}

status() {
  running && say "serve pid $(cat "$PIDFILE"): running" || say "serve: not running"
  say "swap: $(sysctl -n vm.swapusage)"
  df -h / | tail -1
  [ -f "$LOG" ] && { say "engine sizing lines:"; grep -E "strata metal stream|planned resident|prompts on the matrix units|hit rate|expert arena" "$LOG" | tail -8; }
  [ -f "$LOG" ] && { say "last log:"; tail -3 "$LOG"; }
}

probe() {
  running || { say "the Huihui serve is not running; start it first ($0 start)"; exit 1; }
  "$PY" - "$PORT" "$@" <<'PYEOF'
import json, re, sys, time, urllib.request
from hashlib import sha256

both = "--both" in sys.argv
targets = [("huihui-ud-q4_k_xl", 8181), ("iq3_xxs (live server)", 8180)] if both else [("huihui-ud-q4_k_xl", 8181)]
salt = sha256(str(time.time()).encode()).hexdigest()[:8]

body = {
    "short": f"Explain in two sentences why rivers make good borders. Case {salt}.",
    "4k": ("Summarize the key engineering tradeoffs of hybrid CPU/GPU inference in under 200 words. Case " + salt + ". "
           + ("Streaming expert weights from NVMe turns a fixed GPU budget into a memory hierarchy: every token "
              "that routes to a cold expert pays an SSD read, so routing skew, cache admission policy, and the "
              "size of the resident hot set all change the token rate. Chunked prompt GEMMs amortize each expert "
              "read across many rows, which is why longer prompt chunks help when the drive is the bottleneck. "
              ) * 12),
    "12k": ("Given this engineering log, list the three most likely causes of the token-rate drop. Case " + salt + ". "
            + ("At 16:40 the arena hit rate fell from 0.86 to 0.41 after another process started reading a 90 GB "
               "file; drive read latency rose from 0.9 ms to 3.1 ms per 2 MB expert; GPU utilization stayed flat. "
               ) * 26),
}

def post(port, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read().decode())

def timings(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/status", timeout=10) as r:
            return (json.loads(r.read().decode()) or {}).get("last_timings") or {}
    except Exception:
        return {}

print(f"{'target':22} {'probe':11} {'wall_s':>7} {'prompt_n':>8} {'prompt_t/s':>10} {'pred_n':>6} {'pred_t/s':>8}  first words")
for name, port in targets:
    for label in ("short", "short", "4k", "12k"):
        maxtok = 96 if label == "short" else 160
        payload = {"messages": [{"role": "user", "content": body[label]}],
                   "temperature": 0, "max_tokens": maxtok, "stream": False}
        t0 = time.monotonic()
        try:
            r = post(port, payload)
            wall = time.monotonic() - t0
            text = (r.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            t = timings(port)
            pn, ps = t.get("prompt_n"), t.get("prompt_per_second")
            dn, ds = t.get("predicted_n") or 0, t.get("predicted_per_second")
            snippet = re.sub(r"\s+", " ", text).strip()[:48]
            print(f"{name:22} {label:11} {wall:7.2f} {str(pn):>8} {str(ps):>10} {str(dn):>6} {str(ds):>8}  {snippet}")
        except Exception as e:
            print(f"{name:22} {label:11} FAILED: {e}")
PYEOF
}

stop() {
  running && { kill "$(cat "$PIDFILE")"; say "stopped pid $(cat "$PIDFILE")"; rm -f "$PIDFILE"; } || say "nothing to stop (this harness's pid file is empty)"
}

case "${1:-all}" in
  start)  start ;;
  probe)  shift; probe "$@" ;;
  status) status ;;
  stop)   stop ;;
  all)    start; probe; status ;;
  *)      say "usage: $0 [start|probe [--both]|status|stop|all] [--coexist]"; exit 2 ;;
esac
