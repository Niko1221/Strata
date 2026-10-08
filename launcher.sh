#!/bin/sh
# Strata's launcher: one page to pick which model to run and with which settings - the sizes setup offers and what
# they need, the models this PC already has, presets you can save and change, setup's calibration, and the speed of
# each model (published, measured here, and live while it runs).
# It changes nothing by itself: a download runs setup exactly as ./setup.sh --setup does, and a start runs the same
# serve/server.py as run-<model>.sh. Closing this terminal leaves a started model running.
cd "$(dirname "$0")" || exit 1
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python -m launcher "$@"
fi
for c in python3 python; do
  if command -v $c >/dev/null 2>&1 && $c -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    echo "Strata's own Python environment (.venv) is missing: run ./setup.sh once to install or repair it."
    echo "Starting the launcher with $c instead - a download still uses setup's own environment."
    exec $c -m launcher "$@"
  fi
done
echo "The launcher needs Python 3.10 or newer. Run ./setup.sh once - it installs Python and the model - then"
echo "./launcher.sh."
exit 1
