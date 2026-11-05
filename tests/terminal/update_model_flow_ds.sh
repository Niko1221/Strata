#!/usr/bin/env bash
# DeepSeek beside a loaded model, from a BUILT Terminal package, with the real DeepSeek engine packages (0.1.54 and 0.1.55) and a real DeepSeek model.
#
#   TERMINAL_ARCHIVE=<strata-terminal-*.tar.gz> REL28=<dir> REL29=<dir> DSMODEL=<DeepSeek model folder> bash tests/terminal/update_model_flow_ds.sh OLD|NEW
#
#   OLD   run it with the v0.2.9 Terminal (the release the owner's user has): load the model on engine 0.1.54, then launch it again (a restart) with the
#         0.2.9 feed in reach.  Expected on v0.2.9: the launch updates the engine by itself while the model is loaded, and removes the old package folder.
#   NEW   run it with the fixed Terminal: the same launch reaches no feed and changes nothing; `strata update --check` names the model; `strata update`
#         asks, downloads while the model serves, stops it, installs 0.1.55 next to 0.1.54, starts the model again on it; a cut download stops nothing.
# HOME, XDG_* and the settings file are inside $W. Both modes share $DATA (the prepared model) so the 10 minutes of "Preparing your model" happen once.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-NEW}"
W="${FLOW_DIR:-$HOME/update-model-flow-ds}"
DATA="${DATA:-$W/data}"
FAIL=0
say() { printf '\n=== %s  (%s)\n' "$*" "$(date +%H:%M:%S)"; }
ok() { printf 'PASS  %s\n' "$*"; }
bad() { printf 'FAIL  %s\n' "$*"; FAIL=1; }
check() { if eval "$2"; then ok "$1"; else bad "$1  [$2]"; fi; }
PIDS=()
cleanup() { [ -x "${S:-/nonexistent}" ] && "$S" launch --stop >/dev/null 2>&1; for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done; }
trap cleanup EXIT
: "${TERMINAL_ARCHIVE:?}" "${REL28:?}" "${REL29:?}" "${DSMODEL:?}"
H="$W/home-$MODE"; rm -rf "$H" "$W/term-$MODE"; mkdir -p "$H" "$W/term-$MODE" "$W/logs" "$DATA"
tar -xzf "$TERMINAL_ARCHIVE" -C "$W/term-$MODE"
S="$(find "$W/term-$MODE" -maxdepth 3 -path '*/bin/strata' | head -1)"
export HOME="$H" XDG_DATA_HOME="$H/.local/share" XDG_CONFIG_HOME="$H/.config" XDG_STATE_HOME="$H/.local/state" XDG_CACHE_HOME="$H/.cache"
export STRATA_SETTINGS_FILE="$H/app-settings.json"
unset STRATA_NO_ACQUIRE STRATA_COMPONENTS_DIR STRATA_API_KEY STRATA_UPDATE_FEED STRATA_ENGINE_BACKEND STRATA_INSTALL_DIR STRATA_HOME STRATA_UPDATE_TRUSTED_KEYS
PORT_MODEL=18490
echo "mode $MODE; package $(basename "$TERMINAL_ARCHIVE"); launcher $S"
DSDIR="$H/.local/share/strata/install/engine-dsv41"      # (the install folder of this HOME, where DeepSeek packages are put)

serve() { local dir="$1" log="$2" port; shift 2
  port=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])')
  python3 "$HERE/cutserver.py" "$port" "$dir" "$@" > "$log" 2>&1 &
  SERVER_PID=$!; PIDS+=("$SERVER_PID")
  for _ in $(seq 1 50); do curl -fs -o /dev/null "http://127.0.0.1:$port/strata-feed.json" && break; sleep 0.2; done
  PORT="$port"; }
ds_pid() { local p exe; for p in /proc/[0-9]*; do exe=$(readlink "$p/exe" 2>/dev/null) || continue; case "$exe" in "$DSDIR"/*) echo "${p#/proc/}"; return;; esac; done; }
ds_exe() { local p; p=$(ds_pid); [ -n "$p" ] && readlink "/proc/$p/exe"; }
ds_folder() { ds_exe | sed 's#.*/engine-dsv41/\([^/]*\)/.*#\1#'; }
folders() { ls "$DSDIR" 2>/dev/null | grep -v '^\.' | sort | tr '\n' ' '; }
gets() { grep -c "GET /$1" "$2" || true; }
answers() { python3 "$HERE/prompt_check.py" "$PORT_MODEL"; }
launch() {  # $1 feed, rest: extra flags
  local feed="$1"; shift
  "$S" launch --family deepseek41 --model ist-das-2.25bpw --model-path "$DSMODEL" --data-dir "$DATA" --port "$PORT_MODEL" --no-api-key --context 8k --vision off --yes --no-tui --feed "$feed" "$@" 2>&1 | tail -14
  return "${PIPESTATUS[0]}"; }

serve "$REL28" "$W/logs/ds-server28-$MODE.log"; P28=$PORT; PID28=$SERVER_PID; F28="http://127.0.0.1:$P28/strata-feed.json"
say "1: DeepSeek on engine 0.1.54 (the v0.2.8 feed), model loaded"
launch "$F28"; RC=$?
check "the model loaded" "[ $RC -eq 0 ]"
check "its engine process runs from the 0.1.54 package folder" "[ \"\$(ds_folder | cut -d- -f1)\" = 0.1.54 ]"
check "it answers a real question" "answers"
OLDPID=$(ds_pid); OLDEXE=$(ds_exe); echo "engine pid $OLDPID  exe $OLDEXE  folders: $(folders)"
kill "$PID28"; wait "$PID28" 2>/dev/null
serve "$REL29" "$W/logs/ds-server29-$MODE.log"; P29=$PORT; F29="http://127.0.0.1:$P29/strata-feed.json"
ENG55=strata-dsv41-0.1.55-linux-x86_64-cuda-cuda13.tar.gz
ENGQ55=strata-engine-0.1.55-linux-x86_64-cuda-cuda13.tar.gz        # (a PC with only DeepSeek installed is also offered the Strata engine: both are cut)

if [ "$MODE" = OLD ]; then
  say "2 (v0.2.9): launch it again while the model is loaded, the 0.2.9 feed in reach"
  # same choices, one more setting changed so the launch really goes through its steps (and --restart: the screens would ask)
  "$S" launch --family deepseek41 --model ist-das-2.25bpw --model-path "$DSMODEL" --data-dir "$DATA" --port "$PORT_MODEL" --no-api-key --context 12k --vision off --yes --no-tui --restart --feed "$F29" 2>&1 | tee "$W/logs/ds-old-relaunch.txt" | tail -14
  echo "engine package folders now: $(folders)"
  check "[the bug] the launch fetched the new engine by itself ($(gets "$ENG55" "$W/logs/ds-server29-$MODE.log") archive request(s))" "[ \"\$(gets $ENG55 '$W/logs/ds-server29-$MODE.log')\" -ge 1 ]"
  check "[the bug] the 0.1.54 package folder the model was running from is gone" "[ ! -d '$DSDIR'/0.1.54-* ]"
  "$S" launch --stop 2>&1 | tail -1
else
  say "2: launch it again while the model is loaded, the 0.2.9 feed in reach: no request, no change"
  B=$(gets "$ENG55" "$W/logs/ds-server29-$MODE.log"); R0=$(wc -l < "$W/logs/ds-server29-$MODE.log")
  "$S" launch --family deepseek41 --model ist-das-2.25bpw --model-path "$DSMODEL" --data-dir "$DATA" --port "$PORT_MODEL" --no-api-key --context 8k --vision off --yes --no-tui --feed "$F29" 2>&1 | tail -4
  check "the feed was not touched" "[ \"\$(wc -l < '$W/logs/ds-server29-$MODE.log')\" = '$R0' ]"
  check "nothing changed: same engine process, same folders (0.1.54 only)" "[ \"\$(ds_pid)\" = '$OLDPID' ] && [ \"\$(folders)\" = '0.1.54-linux-x86_64-cuda-cuda13 ' ]"

  say "3: the check names the model and the one next step"
  "$S" update --check --feed "$F29" --no-control --backend cuda 2>&1 | tee "$W/logs/ds-check.txt"
  check "DeepSeek update available" "grep -q 'DeepSeek 0.1.54: update available' '$W/logs/ds-check.txt'"
  check "the model is named and the update says it will ask" "grep -q 'asks before it stops it' '$W/logs/ds-check.txt'"

  say "4: a script is refused, a cut download stops nothing"
  "$S" update --feed "$F29" --no-control --backend cuda --yes; RC=$?
  check "exit 7, no archive requested" "[ $RC -eq 7 ] && [ \"\$(gets $ENG55 '$W/logs/ds-server29-$MODE.log')\" = 0 ]"
  serve "$REL29" "$W/logs/ds-servercut.log" "$ENG55,$ENGQ55" 30000000; PCUT=$PORT; PIDCUT=$SERVER_PID
  "$S" update --feed "http://127.0.0.1:$PCUT/strata-feed.json" --no-control --backend cuda --stop-model --start-again; RC=$?
  check "the cut download: exit 6, the model was never stopped" "[ $RC -eq 6 ] && [ \"\$(ds_pid)\" = '$OLDPID' ]"
  kill "$PIDCUT"; wait "$PIDCUT" 2>/dev/null

  say "5: yes, then yes to start again"
  python3 "$HERE/pty_drive.py" --timeout 1500 --answer 'for the update\?.*=>y' --answer 'again with the same settings\?.*=>y' -- "$S" update --feed "$F29" --no-control --backend cuda > "$W/logs/ds-yes.txt" 2>&1; RC=$?
  sed 's/\x1b\[[0-9;?]*[A-Za-z]//g' "$W/logs/ds-yes.txt" | tr -d '\r' | grep -v '^$' | tail -20
  check "exit 0" "[ $RC -eq 0 ]"
  check "the old engine file was never touched while the model ran: the 0.1.54 package is still installed" "[ -d '$DSDIR'/0.1.54-linux-x86_64-cuda-cuda13 ] && [ -x '$OLDEXE' ]"
  check "both packages are installed" "[ \"\$(folders | wc -w)\" = 2 ]"
  check "the model runs on the NEW package" "[ \"\$(ds_folder | cut -d- -f1)\" = 0.1.55 ] && [ \"\$(ds_pid)\" != '$OLDPID' ]"
  check "and answers a real question" "answers"
  "$S" launch --stop 2>&1 | tail -1
fi
say "result ($MODE)"
[ "$FAIL" = 0 ] && echo "ALL PASSED" || echo "SOME FAILED"
exit "$FAIL"
