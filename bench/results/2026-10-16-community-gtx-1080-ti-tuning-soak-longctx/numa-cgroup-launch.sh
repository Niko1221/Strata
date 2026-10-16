#!/bin/bash
# launch.sh CMD... (run as root) - put this process tree in the node-1 cpuset cgroup, drop to wangyi, exec CMD.
set -u
CG=/sys/fs/cgroup/strata-ab
mkdir -p $CG
echo 22-43,66-87 > $CG/cpuset.cpus
echo $BASHPID > $CG/cgroup.procs
exec sudo -n -u wangyi "$@"
