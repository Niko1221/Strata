#!/bin/sh
# Benchmark this Strata install on a Mac the way it is used and write a report you can share
# (docs/COMMUNITY_BENCHMARKS.md): double-click it in Finder. It runs ./bench.sh: the model starts with your settings
# (or the running one is used), about 10 minutes, nothing is changed. Options: --clients 4, --long, --compare <folder>.
main() {
  cd "$(dirname "$0")" || exit 1
  if [ "$(uname -s)" != Darwin ]; then
    echo "This launcher is for a Mac. On Linux run ./bench.sh, on Windows BENCH.bat."
    exit 1
  fi
  sh ./bench.sh "$@"
  status=$?
  echo
  printf 'Press Enter to close this window. '
  read -r _
  exit "$status"
}
main "$@"
