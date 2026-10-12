#!/bin/sh
# Benchmark this Strata install the way it is used and write a report you can share (docs/COMMUNITY_BENCHMARKS.md).
# It starts the model with your settings (or uses the one already running), takes about 10 minutes and changes nothing.
# Options: --clients 4 (several requests at once), --long (a 32K prompt), --compare <older results folder>.
cd "$(dirname "$0")" || exit 1
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python tools/strata_bench.py "$@"
fi
echo "Strata is not set up here yet: run ./setup.sh first."
exit 1
