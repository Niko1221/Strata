#!/usr/bin/env bash
# Generate a self-signed TLS certificate.
# Clients must trust this certificate to reach the HTTPS API.
#
# Usage:
#   ./scripts/gen-cert.sh                          # localhost + detected LAN IP
#   ./scripts/gen-cert.sh myhost.lan 192.168.1.20 # explicit name and IP
#
# Note: connecting to an address that is not in the certificate SAN makes
# browsers and clients report a certificate error. Inspect the current SANs with:
#   openssl x509 -in certs/server.crt -noout -text | grep "Subject Alternative"
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p certs

HOSTNAME_ARG="${1:-}"
IP_ARG="${2:-}"

# Best-effort detection of the address used on the default route (LAN access)
AUTO_IP="$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}' || true)"
AUTO_IP="${AUTO_IP:-$IP_ARG}"
COMPUTER="$(hostname 2>/dev/null || echo localhost)"

SAN="DNS:localhost"
[ -n "$COMPUTER" ] && SAN="$SAN,DNS:$COMPUTER"
[ -n "$HOSTNAME_ARG" ] && SAN="$SAN,DNS:$HOSTNAME_ARG"
[ -n "$AUTO_IP" ] && SAN="$SAN,IP:$AUTO_IP"
SAN="$SAN,IP:127.0.0.1"
# de-duplicate
SAN="$(echo "$SAN" | tr ',' '\n' | awk 'NF&&!seen[$0]++' | paste -sd, -)"

if [ -f certs/server.crt ] && [ -z "${FORCE:-}" ]; then
    echo "certs/server.crt already exists. To regenerate: FORCE=1 $0"
    exit 0
fi

echo "Generating self-signed certificate, SAN: $SAN"

# Git Bash / MSYS rewrites /CN=... into a Windows path, which breaks openssl
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"

openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
    -subj "/CN=strata.local" \
    -addext "subjectAltName=$SAN" \
    -keyout certs/server.key \
    -out certs/server.crt 2>/dev/null

echo "✓ certs/server.crt generated"
echo "✓ certs/server.key generated (private key — do not commit)"
echo
echo "Clients must trust this certificate to avoid browser warnings:"
echo "  Linux/WSL : sudo cp certs/server.crt /usr/local/share/ca-certificates/strata.crt && sudo update-ca-certificates"
echo "  Windows   : double-click certs/server.crt → Install Certificate → Current User → Trusted Root Certification Authorities"
echo "  macOS     : sudo security add-trusted-cert -d -k /Library/Keychains/System.keychain certs/server.crt"
echo "  Node.js   : export NODE_EXTRA_CA_CERTS=\$(pwd)/certs/server.crt"
