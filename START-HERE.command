#!/bin/sh
# Strata for Mac: double-click this file in Finder (it opens in Terminal). The first run installs everything and starts
# the model; later runs just start it. It is ./setup.sh, for people who do not want to type: options are passed on the
# same way (SETUP.command is "--setup", UPDATE.command runs ./update.sh, BENCH.command runs ./bench.sh).
# Needs an Apple Silicon Mac (M1 or newer), macOS 15 or newer and Apple's Command Line Tools (xcode-select --install).
# Closing the Terminal window stops the model, as on Windows.
# If macOS says the file "cannot be opened because it is from an unidentified developer" (a copy downloaded as a zip
# carries that mark; a git clone does not): right-click it, choose Open, then Open again. Once is enough.
main() {
  cd "$(dirname "$0")" || exit 1
  if [ "$(uname -s)" != Darwin ]; then
    echo "This launcher is for a Mac. On Linux run ./setup.sh, on Windows START-HERE.bat."
    exit 1
  fi
  sh ./setup.sh "$@"
  status=$?
  if [ "$status" -ne 0 ]; then
    echo
    printf 'Strata stopped with an error (the reason is above). Press Enter to close this window. '
    read -r _
  fi
  exit "$status"
}
main "$@"
