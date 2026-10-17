#!/usr/bin/env bash
# remote-vision-watchdog.sh - run the image encoder, and end it with the AI service it serves.
#
# The encoder is started over ssh by strata-vision-remote on the machine that runs the model.  It
# exists to serve THAT machine's Strata server, so its lifetime follows the server's port: this
# script TCP-probes that port back, over the address the ssh session came from ($SSH_CONNECTION -
# whatever path it took, Tailscale or LAN), and reaps the encoder once the port has been closed
# for FAIL-LIMIT probes in a row.  A live port means the service is running, even when it sends
# no images for hours; a hard-killed server, a sleeping PC or a dropped link takes the encoder
# down in ~30 s and its VRAM with it.
#
# This replaces the heartbeat-file design: the wrapper used to touch <hb> over ssh every 20 s,
# and one failed heartbeat command was enough to lose a healthy encoder minutes later while the
# server kept running.  No ssh per beat, no file, no clock - just the port.
#
#   * the probe must SUCCEED once before failures count: the server binds its port a moment after
#     this session starts, and a master that never opens the port (bound to 127.0.0.1?) is
#     reported once and left alone - its own exit ends this session anyway;
#   * when the wrapper is hard-killed, its ssh client dies, this session gets SIGHUP and the
#     encoder goes with it; the probe is the backstop for half-open connections, where no
#     signal arrives at all;
#   * kill by PROCESS GROUP ID, never `pkill -f 'strata-vision --mmproj'`: the ssh command line
#     contains that pattern, so the match includes the shell running pkill and the session dies;
#   * do NOT run cleanup on the normal exit path.  `wait` returns as soon as the encoder stops,
#     and killing its process group at that moment races the encoder's own shutdown - measured as
#     an empty reply and a lost result.  Only the probe-failure branch may kill anything.
#
#   remote-vision-watchdog.sh <service-port> <fail-limit> <pidfile> <command...>
#   (STRATA_VISION_MASTER_IP overrides the address taken from SSH_CONNECTION)
# The encoder's PID is written to <pidfile> so its owner can stop exactly this session's
# encoder later - a global pkill would also hit other sessions on this host.
set -uo pipefail

PORT="${1:?usage: remote-vision-watchdog.sh <service-port> <fail-limit> <pidfile> <command...>}"
FAILS="${2:?}"
PIDFILE="${3:?}"
shift 3

MASTER="${STRATA_VISION_MASTER_IP:-${SSH_CONNECTION%% *}}"
if [ -z "$MASTER" ]; then
  echo "watchdog: no master address: not an ssh session (set STRATA_VISION_MASTER_IP)" >&2
  exit 2
fi

# one probe: a TCP connect, nothing written, closed at once.  A closed port answers instantly
# (refused); a host that is gone (sleep, dropped link) hangs, so `timeout` ends the probe.
probe() { timeout 3 bash -c "exec 3<>/dev/tcp/$MASTER/$PORT" 2>/dev/null; }

# Keep the session's stdin on fd 3 and hand it to the encoder explicitly.  As a plain background
# job (`"$@" &`) the encoder does not get this pipe: measured, it starts, answers nothing, and the
# ENC round trip returns an empty reply.
exec 3<&0
"$@" <&3 &
ENC=$!
echo "$ENC" > "$PIDFILE" 2>/dev/null || true

# Kill by process group. The kernel truncates comm to 15 characters, so the encoder's process name
# is "strata-vision-r" - `pkill -x strata-vision` never matched it.
kill_encoder() {
  kill -TERM -"$ENC" 2>/dev/null || kill -TERM "$ENC" 2>/dev/null
}

# The only killer: the service this encoder serves stopped answering on its port.
(
  seen=0 bad=0
  while sleep 10; do
    if probe; then
      seen=1 bad=0
      continue
    fi
    bad=$((bad + 1))
    if [ "$seen" -eq 0 ]; then
      # never reached the port: say so once, do not reap - a master bound to 127.0.0.1 would
      # otherwise lose its encoder after 30 s with no explanation
      if [ "$bad" -eq 6 ]; then
        echo "watchdog: $MASTER:$PORT never answered - is the Strata server bound to 0.0.0.0 (not 127.0.0.1)? leaving the encoder alone" >&2
      fi
      continue
    fi
    if [ "$bad" -ge "$FAILS" ]; then
      echo "watchdog: $MASTER:$PORT closed for $bad probes (limit $FAILS) - the service is gone, stopping the encoder" >&2
      kill_encoder
      exit 0
    fi
  done
) &
WATCH=$!

# The wrapper asked us to stop (QUIT arrived, or the channel closed): stop watching, leave the
# encoder's own shutdown alone.
trap 'kill "$WATCH" 2>/dev/null; exit 0' TERM INT

wait "$ENC"
rc=$?
kill "$WATCH" 2>/dev/null
exit "$rc"
