#!/usr/bin/env bash
# Bootstrap API keys: creates the first admin key and a regular inference key.
#
# Usage:
#   ./scripts/init-keys.sh                 # admin + user
#   ./scripts/init-keys.sh --admin-only    # admin only
#
# Keep the generated keys safe: the console never displays a full key again.

# This script creates real credentials; run it only when you intend to.
set -euo pipefail

HOST="${STRATA_HOST:-localhost}"
PORT="${STRATA_HTTPS_PORT:-38888}"
API="https://$HOST:$PORT"

ADMIN_ONLY=""
[ "${1:-}" = "--admin-only" ] && ADMIN_ONLY=1

if [ ! -f certs/server.crt ]; then
    echo "certs/server.crt not found — run ./scripts/gen-cert.sh first" >&2
    exit 1
fi

create() {
    local name="$1" role="$2" auth="$3"
    local args=(-sk -X POST "$API/api/keys" -H "Content-Type: application/json")
    [ -n "$auth" ] && args+=(-H "Authorization: Bearer $auth")
    args+=(-d "{\"name\":\"$name\",\"role\":\"$role\"}")
    curl "${args[@]}" 2>/dev/null
}

echo "→ creating admin key ..."
ADMIN_JSON="$(create admin admin "")"
ADMIN_KEY="$(echo "$ADMIN_JSON" | sed -n 's/.*"key":"\([^"]*\)".*/\1/p')"
[ -z "$ADMIN_KEY" ] && { echo "failed: $ADMIN_JSON" >&2; exit 1; }
umask 077
echo "$ADMIN_KEY" > certs/admin.key
chmod 600 certs/admin.key
echo "✓ admin key written to certs/admin.key"

if [ -z "$ADMIN_ONLY" ]; then
    echo "→ creating user key (inference only) ..."
    USER_JSON="$(create default-user user "$ADMIN_KEY")"
    USER_KEY="$(echo "$USER_JSON" | sed -n 's/.*"key":"\([^"]*\)".*/\1/p')"
    [ -z "$USER_KEY" ] && { echo "failed: $USER_JSON" >&2; exit 1; }
    echo "$USER_KEY" > certs/user.key
    chmod 600 certs/user.key
    echo "✓ user key written to certs/user.key"
fi

cat <<EOF

────────────────────────────────────────────────────────────
 API endpoint : $API/v1
 admin key    : certs/admin.key   (console + inference)
 user key     : certs/user.key    (inference only)
 model id     : GET $API/v1/models

 Example:
   curl $API/v1/chat/completions \\
     -H "Authorization: Bearer \$(cat certs/user.key)" \\
     -H "Content-Type: application/json" \\
     -d '{"model":"<MODEL_ID>","messages":[{"role":"user","content":"hi"}]}'
────────────────────────────────────────────────────────────
EOF
