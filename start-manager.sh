#!/bin/sh
# Strata Manager for Linux (Arch / Omnarchy and friends): the same Manager as Windows' START-MANAGER.bat.
# Needs the .venv environment setup.sh makes - run ./setup.sh once first if it is missing.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "  The environment .venv is missing - run ./setup.sh once first (it installs the model too)."
  echo "  Then ./start-manager.sh opens the Manager. (Or: python3 gui/manager.py)"
  exit 1
fi
exec .venv/bin/python gui/manager.py "$@"