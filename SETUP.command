#!/bin/sh
# Change settings or install another model (the same as START-HERE.command --setup). Double-click it in Finder.
exec sh "$(dirname "$0")/START-HERE.command" --setup "$@"
