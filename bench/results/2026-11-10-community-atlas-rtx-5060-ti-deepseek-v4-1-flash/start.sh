#!/bin/sh
cd "<strata_worktree>"
exec "/usr/sbin/python3" "<strata_worktree>/serve/server.py" "--engine" "strata" "--config" "<strata_worktree>/strata-deepseek-v4.1-flash-ist-das-2.00bpw.json" "--port" "8080" "--open"
