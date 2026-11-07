#!/usr/bin/env bash
set -u
cd /mnt/data4tb/Strata-gyro
export CUDA_VISIBLE_DEVICES=GPU-a351101d-34cd-ef2b-8a8b-14aa2395e8ce,GPU-71d31235-62ca-4433-aad2-9aa2038164df
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec .venv/bin/python -m serve.server --engine strata --config strata-groot-gyro-pair.json --port 18044
