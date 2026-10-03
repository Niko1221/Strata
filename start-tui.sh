#!/bin/sh
# Optional minimal terminal launcher for Strata (see tools/strata_tui.py).
cd "$(dirname "$0")"
exec python3 tools/strata_tui.py
