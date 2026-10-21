#!/bin/sh
# Run the Strata engine under Nsight Systems, so the GPU's kernels, the copies and the
# idle gaps of a real run are recorded.  The engine's own arguments follow.
#   NSYS_OUT   where the .nsys-rep goes (without the suffix)
# The engine path comes first, so Nsight stops reading its own options there and
# passes the rest (--serve --pack ...) to the engine.
exec /opt/cuda/nsight_systems/bin/nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-event-trace=false --cuda-graph-trace=node \
  -o "${NSYS_OUT:-/home/mo/strata-mainline/bench/results/2026-10-21-gyro-titanx/nsys-tools/strata}" \
  --force-overwrite true \
  /home/mo/strata-mainline/build-cuda12/strata "$@"
