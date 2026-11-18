#!/bin/bash
# Disk sessions for the daily IQ3_XXS server (port 8180, config key
# "slot_save_path" -> /Users/pc/src/Strata-data/sessions/iq3_xxs).
#
#   tools/iq3_session.sh save    [NAME]    # save the conversation the engine holds
#   tools/iq3_session.sh restore [NAME]    # read a saved file back into the live engine
#   tools/iq3_session.sh stop    [NAME]    # save first, then stop the server
#   tools/iq3_session.sh start   [NAME]    # start the server (run-iq3_xxs.sh), then restore NAME if the file exists
#   tools/iq3_session.sh list              # saved files and sizes
#
# NAME defaults to "iq3".  The file holds the engine state (~38 KB per token
# plus ~157 MB fixed window state, measured on this box 2026-10-23; a 63K
# conversation ~2.5 GB), mode 0600, kept until deleted.  A later request that
# continues the restored conversation reads only what follows the saved state.
# Restore only files this engine wrote, after the same model files, rope and
# K/V settings (a changed --kv/--rope invalidates them; draft, spec, arena and
# context-size changes do not, though a bigger file than this session's context
# can hold is refused).  Saving needs the engine binary from the Metal session
# port (build-metal/strata, 2026-10-23): a server started from an older binary
# refuses with "session K/V files are not supported over the Metal window".
set -u
REPO="/Users/pc/src/Strata-fork"
DIR="/Users/pc/src/Strata-data/sessions/iq3_xxs"
API="http://127.0.0.1:8180"
IQ3="serve/server.py.*iq3_xxs"
HUI="serve/server.py.*ud-q4_k_xl-huihui"
NAME_DEFAULT="iq3"

cmd="${1:-}"; name="${2:-$NAME_DEFAULT}"
action() { curl -sS -m 600 -X POST "$API/slots/0?action=$1" -H "Content-Type: application/json" \
  -d "{\"filename\": \"$name\"}"; }
live() { curl -sf -m 3 "$API/v1/models" >/dev/null 2>&1; }

case "$cmd" in
  save)
    live || { echo "the server is not live on 8180"; exit 1; }
    action save; echo
    ;;
  restore)
    live || { echo "the server is not live on 8180"; exit 1; }
    action restore; echo
    ;;
  list)
    ls -lh "$DIR" 2>/dev/null || echo "no saved files yet ($DIR)"
    ;;
  stop)
    pid=$(pgrep -f "$IQ3" | head -1)
    [ -n "$pid" ] || { echo "no IQ3_XXS server running"; exit 1; }
    if live; then echo "saving the conversation as $name ..."; action save; fi
    kill -TERM "$pid"; echo "server stopped"
    ;;
  start)
    live && { echo "already live on 8180"; exit 0; }
    if curl -sf -m 3 http://127.0.0.1:8181/v1/models >/dev/null 2>&1; then
      echo "the Huihui UD-Q4_K_XL server on 8181 is live - this box cannot hold both engines; stop it first"; exit 1
    fi
    nohup "$REPO/run-iq3_xxs.sh" >> "$REPO/strata-iq3_xxs.log" 2>&1 &
    for _ in $(seq 1 90); do sleep 4; live && break; done
    live || { echo "engine did not come up; see $REPO/strata-iq3_xxs.log"; exit 1; }
    if [ -f "$DIR/$name" ]; then echo "restoring $name ..."; action restore; else echo "(no saved file $DIR/$name)"; fi
    ;;
  *)
    sed -n '2,22p' "$0"; exit 2 ;;
esac
