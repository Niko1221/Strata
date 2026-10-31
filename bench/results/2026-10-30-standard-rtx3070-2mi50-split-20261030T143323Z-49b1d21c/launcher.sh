#!/usr/bin/env bash
#
# Usage: the "exe" of a split_serve run config (stock 0.2.7 serve/split.py starts both
#        engines with this one executable)
#
# Starts the right engine build for each role.  The prefill role is the one whose
# arguments carry --split-role-prefill (put in "prefill_args"); that marker is removed,
# and the CUDA build runs on the RTX 3070 with the CUDA libraries.  Every other start is
# the decode role: the gfx906 build on both MI50s with the gfx906 opt-in switches.
CUDA_EXE=/st/src-0.2.7/build-cuda86/strata
HIP_EXE=/st/src-0.2.7/build-906/strata
args=()
prefill=0
for a in "$@"; do
  if [ "$a" = --split-role-prefill ]; then prefill=1; else args+=("$a"); fi
done
if [ $prefill = 1 ]; then
  unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
  export CUDA_VISIBLE_DEVICES=0
  export LD_LIBRARY_PATH=/cudalib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  exec "$CUDA_EXE" "${args[@]}"
fi
export HIP_VISIBLE_DEVICES=0,1 STRATA_GR_SPLIT=1 STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 STRATA_GFX906_ATTN_REDUCE12=1
exec "$HIP_EXE" "${args[@]}"
