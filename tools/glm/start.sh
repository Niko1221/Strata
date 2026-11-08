#!/bin/sh
# One-command start of the GLM-5.3-Flash API server (OpenAI-compatible) on the LAN.
#
#   ./start-glm.sh                      # start (or report) the server at the standing default:
#                                         512K tokens x 2 conversations (a bare run always
#                                         comes back to it - the last choice is not the default)
#   ./start-glm.sh --context 512k      # that launch at 512K per conversation (restarts the server)
#   ./start-glm.sh --context 1m        # that launch at 1M per conversation (restarts the server)
#   ./start-glm.sh --thinking low      # the lowest effort: measured, no thinking block comes out
#   ./start-glm.sh --thinking high     # think before answering
#   ./start-glm.sh --thinking max      # the model's own default: the longest thinking
# (the model card's three levels; there is no on/off - low already answers without a thinking block)
#
# Every path and the port can be overridden by environment (STRATA_GLM_GGUF_DIR,
# STRATA_GLM_PORT, STRATA_PYTHON).  First run: full setup (shard SHA-256 checks, the pack
# built once) and then the server.  Later runs: the saved launch config, straight to
# serving.  The server answers on the LAN (0.0.0.0) without an API key - the owner's
# choice; Strata warns about an open port once at start.
set -e
HERE="$(cd "$(dirname "$0")/../.." && pwd)"          # tools/glm/ -> the repository root
PORT="${STRATA_GLM_PORT:-8080}"
CFG="$HOME/.local/share/strata/install/strata-glm-5.3-flash-ud-q2_k_xl.json"
STATE_DIR="$HOME/.local/state/strata/launch"
STRATA="$HERE/bin/strata"
# The python that runs the stack: STRATA_PYTHON wins when set; else the checkout's .venv when
# it can import what the server needs (regex ...); else the system python3 - which crashes the
# server at start here (ModuleNotFoundError: regex) and used to leave the wait loop silent for
# ten minutes.  The choice is printed, so a wrong pick is visible at once.
PY="${STRATA_PYTHON:-}"
if [ -z "$PY" ] && [ -x "$HERE/.venv/bin/python3" ] \
   && "$HERE/.venv/bin/python3" -c "import regex" > /dev/null 2>&1; then
    PY="$HERE/.venv/bin/python3"
fi
[ -n "$PY" ] || PY="python3"

say() { printf '%s\n' "$*"; }
say "Python: $PY"

# The standing default: a bare run serves 512K tokens per conversation x the parallel slots.
# A --context flag chooses for that one launch; the next bare run comes back to the default,
# so a one-off 1M launch never turns itself into the default.
DEFAULT_CTX=524288
exec 9>"/tmp/strata-glm-start.lock"
flock -n 9 || { echo "another start-glm.sh is running; wait for it to finish"; exit 1; }

# ---- arguments: --context SIZE (512k / 1m / a token count) switches and restarts
CTX=""
THINKING=""
while [ $# -gt 0 ]; do
    case "$1" in
        --context) CTX="$2"; shift 2 ;;
        --thinking) THINKING="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        *) shift ;;
    esac
done
THINK_EFFORT=""
if [ -n "$THINKING" ]; then
    case "$THINKING" in
        low|high|max) THINK_EFFORT="$THINKING" ;;
        *) echo "--thinking takes the model's three levels: low, high or max (got: $THINKING)"; exit 1 ;;
    esac
fi
CTX_T=""
if [ -n "$CTX" ]; then
    CTX_T=$(python3 - "$CTX" <<'PYEOF'
import sys
s = sys.argv[1].strip().lower().replace(" ", "")
mul = 1024
if s.endswith("k"):
    mul, s = 1024, s[:-1]
elif s.endswith("m"):
    mul, s = 1024 * 1024, s[:-1]
v = int(float(s) * mul)
assert v >= 1024, "context too small"
print(v)
PYEOF
)
fi

# the launch config that points at the GLM pack: its stem names the runtime profile
LAUNCH_JSON="$(python3 - "$STATE_DIR" "$PORT" <<'PYEOF'
import json, pathlib, sys
d, port = pathlib.Path(sys.argv[1]), sys.argv[2]
for p in sorted(d.glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True):
    try:
        c = json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        continue
    if "--glm-pack" in (c.get("args") or []):
        if str(c.get("port")) != port:
            continue
        print(p.stem)
        break
PYEOF
)"

if [ -z "$LAUNCH_JSON" ]; then
    say "First run: setting GLM-5.3-Flash up (checks the shards, builds the pack once, then starts)..."
    cd "$HERE"
    STRATA_PYTHON="$PY" STRATA_GLM_PARALLEL="${STRATA_GLM_PARALLEL:-2}" \
        STRATA_GLM_THINKING="${STRATA_GLM_THINKING:-low}" sh bin/strata launch \
        --model-path "${STRATA_GLM_GGUF_DIR:-/nvme/models/GLM-5.3-Flash-UD-Q2_K_XL/UD-Q2_K_XL}" \
        --family glm53 --model UD-Q2_K_XL --context "${CTX:-${STRATA_GLM_CTX:-512k}}" \
        --tune skip --access lan --no-api-key --port "$PORT" --yes
    exit 0
fi
PROFILE="launch-$LAUNCH_JSON"

# ---- a context switch: stop the engine, rewrite --max-context in both saved configs.
# A bare run enforces the standing default too: the saved config holds the *last* choice,
# not the default, so after a --context 1m launch the next bare run returns to 512K x 2.
CUR_CTX=""
if [ -n "$LAUNCH_JSON" ] && [ -f "$STATE_DIR/$LAUNCH_JSON.json" ]; then
    CUR_CTX="$(python3 - "$STATE_DIR/$LAUNCH_JSON.json" <<'PYEOF'
import json, sys
a = (json.load(open(sys.argv[1])).get("args") or [])
print(a[a.index("--max-context") + 1] if "--max-context" in a else "")
PYEOF
)"
fi
EFF_CTX="$CTX_T"
if [ -z "$EFF_CTX" ] && [ -n "$LAUNCH_JSON" ] && [ "$CUR_CTX" != "$DEFAULT_CTX" ]; then
    EFF_CTX="$DEFAULT_CTX"
    say "The saved config says $CUR_CTX tokens per conversation; a bare run serves the default $DEFAULT_CTX (switching needs a restart)."
fi
if [ -n "$EFF_CTX" ]; then
    RUNNING=0
    pgrep -x strata-glm > /dev/null 2>&1 && RUNNING=1
    if [ "$RUNNING" = 1 ]; then
        say "Stopping the running server (context change needs a restart)..."
        (cd "$HERE" && STRATA_PYTHON="$PY" sh bin/strata launch --stop > /dev/null 2>&1 || true)
        sleep 2
        pkill -9 -x strata-glm 2>/dev/null || true
        sleep 2
    fi
    python3 - "$STATE_DIR/$LAUNCH_JSON.json" "$CFG" "$EFF_CTX" "${THINK_EFFORT:-keep}" <<'PYEOF'
import json, os, sys
new, effort = str(sys.argv[3]), sys.argv[4]
for path in sys.argv[1:3]:
    d = json.load(open(path))
    a = d.get("args") or []
    if "--max-context" in a:
        a[a.index("--max-context") + 1] = new
    else:
        a += ["--max-context", new]
    d["args"] = a
    d["max_context"] = int(new)
    if effort != "keep":
        d["reasoning_effort"] = effort
    t = path + ".tmp"
    json.dump(d, open(t, "w"), ensure_ascii=False, indent=1)
    os.replace(t, path)
print("context set to", new + (", thinking effort " + effort if effort != "keep" else ""))
PYEOF
    say "Context set to $EFF_CTX tokens per conversation."
    RUNNING=0
fi

# ---- a parallelism switch alone: rewrite --parallel and "parallel", restart
if [ -n "$PARALLEL" ] && [ -z "$EFF_CTX" ] && [ -n "$LAUNCH_JSON" ]; then
    RUNNING=0
    pgrep -x strata-glm > /dev/null 2>&1 && RUNNING=1
    if [ "$RUNNING" = 1 ]; then
        say "Stopping the running server (the parallelism is read at start)..."
        (cd "$HERE" && STRATA_PYTHON="$PY" sh bin/strata launch --stop > /dev/null 2>&1 || true)
        sleep 2
        pkill -9 -x strata-glm 2>/dev/null || true
        sleep 2
    fi
    python3 - "$STATE_DIR/$LAUNCH_JSON.json" "$CFG" "$PARALLEL" <<'PYEOF'
import json, os, sys
n = str(int(sys.argv[3]))
for path in sys.argv[1:3]:
    d = json.load(open(path))
    a = d.get("args") or []
    if "--parallel" in a:
        a[a.index("--parallel") + 1] = n
    else:
        a += ["--parallel", n]
    d["args"] = a
    d["parallel"] = int(n)
    t = path + ".tmp"
    json.dump(d, open(t, "w"), ensure_ascii=False, indent=1)
    os.replace(t, path)
print("parallel set to", n)
PYEOF
    say "Parallel conversations switched to: $PARALLEL"
    RUNNING=0
fi

# ---- a thinking switch alone: rewrite reasoning_effort, restart
if [ -n "$THINK_EFFORT" ] && [ -z "$EFF_CTX" ] && [ -n "$LAUNCH_JSON" ]; then
    RUNNING=0
    pgrep -x strata-glm > /dev/null 2>&1 && RUNNING=1
    if [ "$RUNNING" = 1 ]; then
        say "Stopping the running server (the thinking default is read at start)..."
        (cd "$HERE" && STRATA_PYTHON="$PY" sh bin/strata launch --stop > /dev/null 2>&1 || true)
        sleep 2
        pkill -9 -x strata-glm 2>/dev/null || true
        sleep 2
    fi
    python3 - "$STATE_DIR/$LAUNCH_JSON.json" "$CFG" "$THINK_EFFORT" <<'PYEOF'
import json, os, sys
effort = sys.argv[3]
for path in sys.argv[1:3]:
    d = json.load(open(path))
    d["reasoning_effort"] = effort
    t = path + ".tmp"
    json.dump(d, open(t, "w"), ensure_ascii=False, indent=1)
    os.replace(t, path)
print("reasoning_effort set to", effort)
PYEOF
    say "Thinking default switched to: $THINK_EFFORT"
    RUNNING=0
fi

# A host whose server is gone (a stop signalled to the server alone, a crashed start) holds the
# profile's state file: a new host would die at once with FileExistsError.  Stop the stale one by PID.
if ! pgrep -f "[s]erve\.server --engine" > /dev/null 2>&1; then
    STALE="$(pgrep -f "[s]trata_control\.runtime --host $PROFILE" || true)"
    if [ -n "$STALE" ]; then
        say "Stopping the leftover host ($STALE) of the stopped server: it holds the profile's state file."
        for p in $STALE; do kill "$p" 2>/dev/null || true; done
        sleep 2
    fi
fi

RUNNING=0
pgrep -x strata-glm > /dev/null 2>&1 && RUNNING=1
if [ "$RUNNING" = 1 ]; then
    say "The GLM engine is already running."
else
    say "Starting the server (the model loads for a couple of minutes)..."
    setsid nohup "$PY" -m strata_control.runtime --host "$PROFILE" --startup-wait 1200 \
        > /tmp/strata-glm-server.out 2>&1 9>&- &
fi

# wait for the server to answer (401 = up but a key is needed; 200 = the model is loaded).
# curl prints 000 when it cannot connect at all: that is no answer, not one.
check_up() {
    c="$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)"
    [ -n "$c" ] && [ "$c" != "000" ]
}

# a start whose server died (a missing module, a bad config, a crash) must fail at once with the
# log's last lines - waiting the loop out in silence reads as a hang
server_gone_bail() {
    if ! pgrep -f "[s]erve\.server --engine" > /dev/null 2>&1; then
        say "The server process is gone (it crashed at start). Last log lines:"
        tail -6 "$HOME/.local/state/strata/runtime/$PROFILE.server.log" 2>/dev/null \
            || say "(no server log at $HOME/.local/state/strata/runtime/$PROFILE.server.log)"
        exit 1
    fi
}
i=0
until check_up; do
    i=$((i + 1))
    [ "$i" -ge 600 ] && { say "The server did not answer within 10 minutes. Log: $STATE_DIR/$LAUNCH_JSON.engine.log"; exit 1; }
    sleep 1
    [ "$i" -ge 4 ] && server_gone_bail
done

KEY="$(cd "$HERE" && STRATA_PYTHON="$PY" sh bin/strata launch --show-key 2>/dev/null | tail -1)"
[ -z "$KEY" ] && KEY=$(python3 -c "import json;print(json.load(open('$STATE_DIR/$LAUNCH_JSON.json')).get('api_key',''))")
AUTH=""
[ -n "$KEY" ] && [ "$KEY" != "none" ] && AUTH="-H \"Authorization: Bearer $KEY\""
# wait for the model itself (the engine loads for a couple of minutes; /v1/models answers 200 when it is up)
i=0
until [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' $AUTH "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)" = "200" ]; do
    i=$((i + 1))
    [ "$i" -ge 1200 ] && { say "The model did not finish loading within 20 minutes. Log: $STATE_DIR/$LAUNCH_JSON.engine.log"; exit 1; }
    sleep 1
    [ "$((i % 15))" -eq 14 ] && server_gone_bail
done

CTX_NOW=$(curl -s -m 5 -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0].get('meta',{}).get('n_ctx','?'))" 2>/dev/null || echo "?")
IP="$(python3 -c 'import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.connect(("10.255.255.255",1));print(s.getsockname()[0])' 2>/dev/null || hostname -I | awk '{print $1}')"
say ""
THINK_NOW=$(python3 -c "import json;print(json.load(open('$STATE_DIR/$LAUNCH_JSON.json')).get('reasoning_effort','max'))")
say "GLM-5.3-Flash is up (context: $CTX_NOW tokens/conversation, reasoning_effort: $THINK_NOW)."
say "  LAN:      http://$IP:$PORT/v1        (from any device on this network)"
say "  Local:    http://127.0.0.1:$PORT/v1  (on this machine)"
if [ -n "$KEY" ] && [ "$KEY" != "none" ]; then
    say "  API key:  $KEY"
else
    say "  API key:  none (open on the LAN - anything that can reach this port can use the model)"
fi
say "  Switch:   ./start-glm.sh --context 512k | 1m"
say "  Stop:     sh bin/strata launch --stop"
say "  Logs:     strata logs --follow  (engine log: $STATE_DIR/$LAUNCH_JSON.engine.log)"
