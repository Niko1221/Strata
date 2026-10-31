#!/usr/bin/env bash
# RES-20261029-01 G1.3 (TASK-20261029-04): the bench-nsys arm's "exe". Launches the prof engine inside an Nsight Systems
# session without capturing; the run then does `nsys start --session=g13 --sample=none --cpuctxsw=none` just before one
# decode request and `nsys stop --session=g13` after it, so the trace holds decode windows, not the load. CUDA-graph node
# tracing, because the decode window is one graph. nsys 2024.6 takes --sample/--cpuctxsw only on start, not on launch
# (the 2026-10-29 21:08 smoke run's engine exited on them).
exec /usr/local/cuda-12.8/bin/nsys launch --session-new="${NSYS_SESSION:-g13}" --trace=cuda,osrt --cuda-graph-trace=node \
  <strata-dev>/build-v100-prof/strata "$@"
