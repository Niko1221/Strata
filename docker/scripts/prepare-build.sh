#!/usr/bin/env bash
# Prepare the build context for the engine image.
#
# Why this exists: the engine Dockerfile contains `COPY . .`, so the build needs
# the complete Strata source tree. Using the repository root directly as the
# build context would also ship sibling secrets (TLS private keys, API key
# files) to the Docker daemon.
#
# This script copies "source tree + this directory's Dockerfile" into a clean
# temporary directory, so the context never contains credentials.
#
# Usage:
#   ./scripts/prepare-build.sh [source-dir]      # default: ..
#   STRATA_BUILD_CTX=/tmp/strata-ctx ./scripts/prepare-build.sh
set -euo pipefail

cd "$(dirname "$0")/.."
HERE="$PWD"
SRC="${1:-$(cd .. && pwd)}"

if [ ! -f "$SRC/CMakeLists.txt" ] || [ ! -f "$SRC/setup.py" ]; then
    echo "✗ $SRC does not look like the Strata source root (CMakeLists.txt or setup.py missing)" >&2
    echo "  usage: $0 /path/to/Strata" >&2
    exit 1
fi

CTX="${STRATA_BUILD_CTX:-$HERE/.build-context}"
mkdir -p "$CTX"

echo "→ copying sources (build artifacts and credentials excluded)..."
EXCLUDES=(
    --exclude='.git' --exclude='__pycache__' --exclude='*.pyc'
    --exclude='.venv' --exclude='engine' --exclude='build' --exclude='build-*'
    --exclude='dist' --exclude='packs' --exclude='mtp' --exclude='models'
    --exclude='*.log' --exclude='*.done' --exclude='*.key' --exclude='*.crt'
    --exclude='*.pem' --exclude='.env'
)
if command -v rsync >/dev/null 2>&1; then
    rsync -a "${EXCLUDES[@]}" "$SRC"/ "$CTX"/
else
    tar -C "$SRC" -cf - . 2>/dev/null | tar -C "$CTX" -xf - \
        --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='.venv' --exclude='engine' --exclude='build' --exclude='build-*' \
        --exclude='dist' --exclude='packs' --exclude='mtp' --exclude='models' \
        --exclude='*.log' --exclude='*.done' --exclude='*.key' --exclude='*.crt' \
        --exclude='*.pem' --exclude='.env'
fi

echo "→ copying this directory's engine build files..."
cp "$HERE/Dockerfile" "$CTX/Dockerfile"
[ -f "$HERE/docker-entrypoint.sh" ] && cp "$HERE/docker-entrypoint.sh" "$CTX/"

# belt and braces: drop anything that slipped through
find "$CTX" -maxdepth 2 \( -name '*.key' -o -name '*.crt' -o -name '.env' \) -delete 2>/dev/null || true

echo "✓ build context ready: $CTX"
echo "  next: docker compose build strata"
