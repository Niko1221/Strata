#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python tools/strata_models.py --open "$@"
fi
exec python3 tools/strata_models.py --open "$@"
