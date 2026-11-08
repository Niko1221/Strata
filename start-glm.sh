#!/bin/sh
# One-command start of the GLM-5.3-Flash API server on the LAN.
# The real script is tools/glm/start.sh; this wrapper exists so the repository
# root has one obvious entry point. See that file for the environment overrides.
exec sh "$(dirname "$0")/tools/glm/start.sh" "$@"
