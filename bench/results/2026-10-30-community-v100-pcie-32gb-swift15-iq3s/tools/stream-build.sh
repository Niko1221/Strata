#!/usr/bin/env bash
# RES-20261029-01 G1.3 (TASK-20261029-04): fetch and compile McCalpin's STREAM on /ai (build only; g13-run.sh runs it
# with the engine stopped). 80M doubles per array = 640 MB x 3, well above 4x the 55 MB of L3.
set -eu
T=<tools>/stream; mkdir -p $T; cd $T
[ -f stream.c ] || curl -sfo stream.c https://www.cs.virginia.edu/stream/FTP/Code/stream.c
nice -n 10 gcc -O3 -march=native -fopenmp -DSTREAM_ARRAY_SIZE=80000000 -DNTIMES=10 -mcmodel=medium stream.c -o stream
echo "# stream built: $(sha256sum stream.c | cut -c1-16) source, $(ls -la stream | awk '{print $5}') bytes"
