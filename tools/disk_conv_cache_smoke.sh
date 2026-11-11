#!/usr/bin/env bash
# --conversation-cache-disk-mib smoke (the manual GPU-box sequence of the disk conversation cache plan).
#
# strata --serve speaks a stdio protocol (GEN/QUIT lines), not HTTP: the HTTP endpoints come from
# serve/server.py, so this smoke drives the server with a throwaway copy of the run config that
# adds the disk flags and tests build/strata.  The disk-cache lines are the engine's stderr, which
# the server writes to the config's "log" file; each phase prints its path.
#
#   CONFIG=strata-<model>.json  tools/disk_conv_cache_smoke.sh serve     (terminal 1; Ctrl+C ends it)
#   tools/disk_conv_cache_smoke.sh chats                                  (terminal 2)
#   CONFIG=strata-<model>.json  tools/disk_conv_cache_smoke.sh restart    (one terminal, self-checking)
#
# Expected in the printed engine log, grep-able ("conversation cache:" prefix):
#   "parked <N> tokens to disk (<B> bytes) in <ms>"   after B's request switches the session
#   "conversation cache: took back <N> tokens from disk (<B> bytes) in <ms>" when A comes back, and A's answer
#     continues from its own state (nothing of A's prompt re-read: the RESUME line's reused count is A's length)
#   "folder <dir>: <N> conversation(s) kept from earlier runs" after a restart with the flag on
# and a conv-*.sess under the cache folder about the size of the conversation.  Answers must match the same
# requests with the flag off (greedy: temperature 0).
set -euo pipefail

PORT="${PORT:-8080}"
PY="${STRATA_PY:-}"                        # the server's Python: setup's .venv has its packages
[[ -z "$PY" && -x ".venv/bin/python3" ]] && PY=".venv/bin/python3"
[[ -z "$PY" ]] && PY="python3"

need_config() {
    [[ -n "${CONFIG:-}" && -f "$CONFIG" ]] || { echo "set CONFIG to the model's strata-<model>.json (the run config)" >&2; exit 2; }
}

run_cfg() {   # run_cfg <engine log> <out json> [mode] - the config pointed at build/strata, with the disk flags
              # added.  mode: keep (default: the disk cache + --conversation-cache-days 7), nodays (the disk
              # cache with the default keep: 0), off (no disk cache at all - the prune leg)
    python3 - "$CONFIG" "$1" "$2" "${3:-keep}" <<'PY'
import json, pathlib, sys
cfg = json.load(open(sys.argv[1]))
a = list(cfg.get("args") or [])
mode = sys.argv[4]
def strip(args, flag):
    out = []
    skip = False
    for x in args:
        if skip:
            skip = False
            continue
        if x == flag:
            skip = True
            continue
        out.append(x)
    return out
if mode == "off":
    a = strip(a, "--conversation-cache-disk-mib")     # the config itself may carry the flag
elif "--conversation-cache-disk-mib" not in a:
    a += ["--conversation-cache-disk-mib", "16384"]
if mode == "nodays":
    a = strip(a, "--conversation-cache-days")         # the default keep (0) must truly apply
elif mode == "keep" and "--conversation-cache-days" not in a:
    a += ["--conversation-cache-days", "7"]
if "--prompt-cache" not in a:
    a += ["--prompt-cache", "4"]          # parking needs the prompt cache on
cfg["args"] = a
cfg["exe"] = str(pathlib.Path("build/strata").resolve())
cfg["log"] = sys.argv[2]                  # the engine's stderr, where the cache lines land
cfg.pop("api_key", None)                  # local smoke on 127.0.0.1; no key
json.dump(cfg, open(sys.argv[3], "w"))
PY
}

pack_of() {
    python3 -c "import json; a=json.load(open('$CONFIG'))['args']; print(a[a.index('--pack')+1] if '--pack' in a else 'pack/full')"
}

wait_ready() {   # wait_ready <server pid> <server log> <engine log> - the model is loaded when /health says so
    until curl -s "localhost:${PORT}/health" | grep -q '"loaded": true'; do
        kill -0 "$1" 2> /dev/null || { echo "server/engine died before the model loaded; see $2 and $3" >&2; exit 1; }
        sleep 2
    done
}

stop_server() {   # SIGTERM = Ctrl+C's path: the server sends QUIT, the engine parks the live conversation
    kill -TERM "$1"; wait "$1" || true
}

chat() {   # chat '<messages json>'
    curl -s "localhost:${PORT}/v1/chat/completions" -H 'content-type: application/json' \
        -d "{\"model\":\"x\",\"temperature\":0,\"messages\":$1}"
    echo
}

case "${1:-}" in
    serve)
        need_config
        d=$(mktemp -d)
        run_cfg "$d/engine.log" "$d/run.json"
        echo "smoke server on http://localhost:${PORT} (Ctrl+C ends it; engine log: $d/engine.log)"
        exec "$PY" -m serve.server --engine strata --config "$d/run.json" --port "$PORT"
        ;;
    chats)
        # A: a long answer, asked in the background (it must still be reading or generating when B arrives)
        chat '[{"role":"user","content":"count to 500, one number per line"}]' > /tmp/disk_conv_A.json &
        sleep 3
        # B: a different chat - the switch parks A to a file
        chat '[{"role":"user","content":"what is the capital of France?"}]'
        # C: a third chat, so the folder holds more than one file
        chat '[{"role":"user","content":"what is the capital of Germany?"},{"role":"assistant","content":"Berlin."},{"role":"user","content":"And of Italy?"}]'
        wait
        # A again, extended: A's answer was kept, only the new turn is read - this is the disk restore
        A=$(python3 -c 'import json,sys; print(json.dumps(json.load(open("/tmp/disk_conv_A.json"))["choices"][0]["message"]))')
        chat "[${A},{\"role\":\"user\",\"content\":\"now count down from 20\"}]"
        PACK="${PACK:-$(pack_of)}"
        ls -l "${PACK}/conversation_cache"   # the files: conv-<epoch ms>-<n>.sess
        ;;
    restart)
        # Cross-restart check (the QUIT park, keep on): chat A, stop the server (QUIT path parks), start it
        # again, A comes back from its file.  Expected, grepped below:
        #   engine1.log: "parked <N> tokens to disk" - the live conversation went to disk on the QUIT exit
        #   engine2.log: "folder <dir>: 1 conversation(s) kept from earlier runs (<B> bytes)" at startup
        #   engine2.log: "took back <N> tokens from disk" when A continues - the answer matches the
        #     same continuing request served without a restart (greedy: temperature 0).
        # Stop mid-generation too: run a "chats"-sized A, stop while it is generating, and the same lines appear.
        # Then the keep-off legs: a start with the default --conversation-cache-days 0 finds the scan emptying
        # the folder (no conv-*.sess survives it) and parks NOTHING on its own QUIT exit; a disk-off start with
        # a file in the folder logs "pruned 1 parked conversation file(s) (the disk cache is off)".
        need_config
        PACK="${PACK:-$(pack_of)}"
        # start from an empty cache folder: a previous run's file could otherwise satisfy the
        # "kept from earlier runs" and "took back" lines without this run's QUIT park
        echo "restart: removing ${PACK}/conversation_cache"
        rm -rf "${PACK}/conversation_cache"
        d=$(mktemp -d)
        run_cfg "$d/engine1.log" "$d/run1.json"
        "$PY" -m serve.server --engine strata --config "$d/run1.json" --port "$PORT" > "$d/srv1.out" 2> "$d/srv1.log" &
        srv=$!
        wait_ready "$srv" "$d/srv1.log" "$d/engine1.log"
        chat '[{"role":"user","content":"count to 500, one number per line"}]' > "$d/A.json"
        stop_server "$srv"
        grep 'parked .* tokens to disk' "$d/engine1.log"          # the park ran on the QUIT exit
        run_cfg "$d/engine2.log" "$d/run2.json"
        "$PY" -m serve.server --engine strata --config "$d/run2.json" --port "$PORT" > "$d/srv2.out" 2> "$d/srv2.log" &
        srv=$!
        wait_ready "$srv" "$d/srv2.log" "$d/engine2.log"
        grep 'kept from earlier runs' "$d/engine2.log"            # the startup scan found the parked file
        A=$(python3 -c 'import json,sys; print(json.dumps(json.load(open("'"$d"'/A.json"))["choices"][0]["message"]))')
        chat "[${A},{\"role\":\"user\",\"content\":\"now count down from 20\"}]"
        grep 'took back .* from disk' "$d/engine2.log"            # A continued from its file, not a cold re-read
        stop_server "$srv"
        # The keep-off default: a start with no --conversation-cache-days empties the folder at its scan and
        # parks nothing when it quits - so quit WITH a live conversation: without one the exit park is
        # skipped for that reason alone and the leg would prove nothing.
        run_cfg "$d/engine3.log" "$d/run3.json" nodays
        "$PY" -m serve.server --engine strata --config "$d/run3.json" --port "$PORT" > "$d/srv3.out" 2> "$d/srv3.log" &
        srv=$!
        wait_ready "$srv" "$d/srv3.log" "$d/engine3.log"
        chat '[{"role":"user","content":"what is the capital of Spain?"}]' > /dev/null
        stop_server "$srv"
        if ls "${PACK}/conversation_cache"/conv-*.sess > /dev/null 2>&1; then
            echo "FAIL: the default keep left files in the folder" >&2; exit 1
        fi
        echo "default keep: the scan emptied the folder"
        if grep -q 'parked .* tokens to disk' "$d/engine3.log"; then
            echo "FAIL: the default keep parked the live conversation on exit" >&2; exit 1
        fi
        # The disk-off prune: a leftover file in the folder goes at start with a line about it.
        mkdir -p "${PACK}/conversation_cache"
        touch "${PACK}/conversation_cache/conv-leftover.sess"
        run_cfg "$d/engine4.log" "$d/run4.json" off
        "$PY" -m serve.server --engine strata --config "$d/run4.json" --port "$PORT" > "$d/srv4.out" 2> "$d/srv4.log" &
        srv=$!
        wait_ready "$srv" "$d/srv4.log" "$d/engine4.log"
        grep 'pruned 1 parked conversation file(s)' "$d/engine4.log"
        if [[ -e "${PACK}/conversation_cache/conv-leftover.sess" ]]; then
            echo "FAIL: the disk-off start left the leftover file" >&2; exit 1
        fi
        echo "disk-off start pruned the leftover"
        stop_server "$srv"
        echo "restart smoke done; throwaway files in $d (engine logs: engine1.log engine2.log engine3.log engine4.log)"
        ;;
    *)
        echo "usage: CONFIG=strata-<model>.json $0 serve | chats | restart   (PORT=, PACK=)" >&2
        exit 2
        ;;
esac
