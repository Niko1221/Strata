#!/bin/bash
# Disk sessions for the Huihui UD-Q4_K_XL server (the engine that hosts this assistant; the run script puts it
# on port 8180 although the config says 8181, so the port is detected, never assumed).  Config key
# "slot_save_path" -> /Users/pc/src/Strata-data/sessions/ud-q4_k_xl-huihui.
#
#   tools/huihui_session.sh save     [NAME]  # save the conversation the engine holds
#   tools/huihui_session.sh restore  [NAME]  # read a saved file back into the live engine
#   tools/huihui_session.sh stop     [NAME]  # save first, then stop the server (waits for it to end)
#   tools/huihui_session.sh start    [NAME]  # start the server, then restore NAME if the file exists
#   tools/huihui_session.sh restart  [NAME]  # save, stop, rebuild if REBUILD=1, start, restore - one call,
#                                            #   so the engine that hosts this assistant is down only briefly
#
# NAME defaults to "huihui".  The file holds the engine state (about 157 MB of fixed window state plus
# ~38 KB per saved token - about 6.3 GB at the full 162K context), mode 0600,
# kept until deleted.  A later request that continues the restored conversation reads only what follows the
# saved state.  Restore only files this engine wrote, after the same model files, rope and K/V settings (a
# changed --kv/--rope invalidates them; draft, spec, arena and context-size changes do not, though a bigger
# file than this session's context can hold is refused).
#
# REBUILD=1 (with restart) runs ninja -C build-metal between stop and start and aborts (leaving the server
# stopped) when the build fails, so a broken rebuild never loses the conversation: it is on disk.
set -u
REPO="/Users/pc/src/Strata-fork"
DIR="/Users/pc/src/Strata-data/sessions/ud-q4_k_xl-huihui"
HUI="serve/server.py.*ud-q4_k_xl-huihui"
ENG="build-metal/strata.*ud-q4_k_xl-huihui"
MODEL_MARK="ud-q4_k_xl-huihui"
API=""
NAME_DEFAULT="huihui"

cmd="${1:-}"; name="${2:-$NAME_DEFAULT}"

find_api() {   # the port this model is live on (the run script overrides the config: never assume 8181)
    local p
    for p in 8180 8181; do
        if curl -sf -m 3 "http://127.0.0.1:$p/v1/models" 2>/dev/null | grep -q "$MODEL_MARK"; then
            API="http://127.0.0.1:$p"; return 0
        fi
    done
    return 1
}
live() { [ -n "$API" ] && curl -sf -m 3 "$API/v1/models" >/dev/null 2>&1; }
action() { curl -sS -m 900 -X POST "$API/slots/0?action=$1" -H "Content-Type: application/json" \
  -d "{\"filename\": \"$name\"}"; echo; }
saved_ok() { case "$1" in *'"error"'*) return 1 ;; *) return 0 ;; esac; }

case "$cmd" in
  save|restore)
    find_api || { echo "no Huihui server live on 8180/8181"; exit 1; }
    action "$cmd"
    ;;
  list)
    ls -lh "$DIR" 2>/dev/null || echo "no saved files yet ($DIR)"
    ;;
  stop)
    pid=$(pgrep -f "$HUI" | head -1)
    [ -n "$pid" ] || { echo "no Huihui server running"; exit 1; }
    if [ "${SKIP_SAVE:-0}" != "1" ] && find_api; then echo "saving the conversation as $name ..."; action save; fi
    kill -TERM "$pid"
    for _ in $(seq 1 90); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
    kill -0 "$pid" 2>/dev/null && { echo "server $pid still running after 180 s; look at it before starting again"; exit 1; }
    ep=$(pgrep -f "$ENG" | head -1)
    [ -z "$ep" ] || { echo "waiting for the engine ($ep) to free its memory ..."; \
      for _ in $(seq 1 60); do kill -0 "$ep" 2>/dev/null || break; sleep 2; done; }
    echo "server stopped"
    ;;
  start)
    find_api && { echo "already live on ${API#http://}"; exit 0; }
    nohup "$REPO/run-ud_q4_k_xl-huihui.sh" >> "$REPO/strata-ud_q4_k_xl-huihui.log" 2>&1 &
    for _ in $(seq 1 120); do sleep 2; find_api && break; done
    find_api || { echo "engine did not come up; see $REPO/strata-ud_q4_k_xl-huihui.log"; exit 1; }
    if [ -f "$DIR/$name" ]; then echo "restoring $name ..."; action restore; else echo "(no saved file $DIR/$name)"; fi
    ;;
  restart)
    if ! find_api; then echo "not live; starting"; exec "$0" start "$name"; fi
    r=$(curl -sS -m 900 -X POST "$API/slots/0?action=save" -H "Content-Type: application/json" \
        -d "{\"filename\": \"$name\"}")
    echo "saved: $r"
    saved_ok "$r" || { echo "save failed; the engine stays as it is"; exit 1; }
    SKIP_SAVE=1 "$0" stop "$name" || exit 1
    if [ "${REBUILD:-0}" = "1" ]; then
        echo "ninja -C $REPO/build-metal ..."
        ninja -C "$REPO/build-metal" || { echo "build failed; the server is stopped, the conversation is safe in $DIR/$name"; exit 1; }
    fi
    "$0" start "$name"
    ;;
  *)
    sed -n '2,22p' "$0"; exit 2 ;;
esac
