#!/bin/sh
# Update Strata on a Mac without starting the model: double-click it in Finder. It runs ./update.sh, which gets the
# newest code (git pull, when this folder is a git clone), then the engine (built again when its source changed), the
# Python packages and each installed model's settings. The model files are not touched. Start the model later with
# START-HERE.command. Options are passed on to setup.py.
main() {
  cd "$(dirname "$0")" || exit 1
  if [ "$(uname -s)" != Darwin ]; then
    echo "This launcher is for a Mac. On Linux run ./update.sh, on Windows UPDATE.bat."
    exit 1
  fi
  sh ./update.sh "$@"
  status=$?
  echo
  if [ "$status" -ne 0 ]; then
    printf 'The update stopped with an error (the reason is above). Press Enter to close this window. '
  else
    printf 'Updated. Start the model with START-HERE.command. Press Enter to close this window. '
  fi
  read -r _
  exit "$status"
}
# all of it in a function, read before it runs: the git pull in update.sh can change this very file
main "$@"
