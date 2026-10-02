#!/bin/bash
# a3000-tune.sh - A/B the sm_86 tuning work on this host, then restart the server.
#
#   ./bench/a3000-tune.sh
#
# Run as the user (nicky). Hugepages need an unlimited locked-memory limit
# (MAP_HUGETLB; see src/core/pinned.cu:202). On NixOS set
# systemd.user.settings.Manager.DefaultLimitMEMLOCK = "infinity" and run this
# script from a systemd unit/scope; an interactive shell only gets the limit
# after a PAM login (security.pam.loginLimits).
#
# Runs, in order:
#   A  0.1.18 engine backup,       --kv int8   - baseline (uses the installed server.py)
#   B  build/strata (0.1.31+),     --kv int8   - engine effect
#   C  build/strata (0.1.31+),     --kv k8v4   - KV effect on top of the new engine
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=/home/nicky/code/forks-Strata
STRATA=/home/nicky/.local/share/strata
CFG=$STRATA/strata-swift-iq3_xxs.json
CFG_OLD=$STRATA/strata-swift-iq3_xxs.json.bak
CFG_A=$STRATA/strata-a.json
CFG_B=$STRATA/strata-b.json
CFG_C=$STRATA/strata-c.json
PY=$STRATA/.venv/bin/python
BACKUP=$STRATA/engine/strata.0.1.18.bak
HUGEPAGES=17408   # 34 GiB of 2 MB pages: covers the 33.97 GiB arena

say() { printf '\n=== %s ===\n' "$*"; }

say "0. hugetlb pool + locked memory limit"
cur=$(cat /proc/sys/vm/nr_hugepages)
if [ "$cur" -lt "$HUGEPAGES" ]; then
    sudo sysctl -w vm.nr_hugepages="$HUGEPAGES"
    echo "vm.nr_hugepages=$HUGEPAGES" | sudo tee /etc/sysctl.d/99-strata-hugepages.conf >/dev/null
fi
echo "nr_hugepages=$(cat /proc/sys/vm/nr_hugepages) HugePages_Free=$(grep '^HugePages_Free:' /proc/meminfo | awk '{print $2}')"
if [ "$(ulimit -l)" != "unlimited" ]; then
    echo "warning: memlock is $(ulimit -l); MAP_HUGETLB will not be used."
    echo "         Run this script from a systemd unit/scope with LimitMEMLOCK=infinity"
    echo "         (see systemd.user.settings.Manager.DefaultLimitMEMLOCK) to enable hugepages."
fi
echo "memlock=$(ulimit -l)"

stop_server() {
    # Stop the systemd user unit and mask it so a pkill is not interpreted as a
    # crash that triggers Restart=on-failure.
    systemctl --user stop strata 2>/dev/null || true
    systemctl --user mask strata 2>/dev/null || true
    pkill -x strata 2>/dev/null || true
    pkill -f 'serve/server\.py --engine strata --config' 2>/dev/null || true
    pkill -f 'serve/server\.py.*8097' 2>/dev/null || true
    pkill -f 'engine/strata.*8097' 2>/dev/null || true
    for i in 1 2 3 4 5; do
        pgrep -x strata >/dev/null || { pgrep -f 'serve/server\.py --engine strata' >/dev/null || break; }
        sleep 2
    done
}

restart() {
    systemctl --user unmask strata 2>/dev/null || true
    systemctl --user start strata 2>/dev/null || setsid "$STRATA/run-swift-iq3_xxs.sh" >/dev/null 2>&1 &
    echo "server restarting on :8080"
}
trap restart EXIT INT TERM

say "1. stop the running server (it comes back at the end)"
stop_server

say "2. prepare configs"
[ -f "$CFG_OLD" ] || cp "$CFG" "$CFG_OLD"
[ -f "$BACKUP" ] || cp -L "$STRATA/engine/strata" "$BACKUP"
"$PY" - <<PY
import json
c = json.load(open('$CFG_OLD'))
c['exe'] = '$BACKUP'
json.dump(c, open('$CFG_A','w'), indent=1)
c = json.load(open('$CFG_OLD'))
c['exe'] = '$ROOT/build/strata'
json.dump(c, open('$CFG_B','w'), indent=1)
c = json.load(open('$CFG'))
c['exe'] = '$ROOT/build/strata'
json.dump(c, open('$CFG_C','w'), indent=1)
PY

run() {
    label="$1"; cfg="$2"; shift 2
    say "$label"
    stop_server
    for e in "$@"; do eval "export $e"; done
    out=$(PY="$PY" "$HERE/e2e.sh" "$cfg" 200 8097) || true
    for e in "$@"; do unset "${e%%=*}"; done
    echo "$out"
    echo "$out" | grep -qE '"decode_tok_s": *[0-9]'
}

run "A: 0.1.18 engine, kv int8" "$CFG_A" SERVE="$STRATA/serve/server.py" || echo "A produced no number - see /tmp/e2e-*.log"
run "B: build/strata, kv int8"  "$CFG_B" || echo "B produced no number - see /tmp/e2e-*.log"
run "C: build/strata, kv k8v4"  "$CFG_C" || echo "C produced no number - see /tmp/e2e-*.log"

say "3. restart the nix-managed server"
stop_server

say "done - A/B/C above; the server restarts on the nix-managed engine"
