#!/bin/bash
cd /mnt/data4tb/Strata-gyro
export CUDA_VISIBLE_DEVICES=GPU-71d31235-62ca-4433-aad2-9aa2038164df
.venv/bin/python -m serve.server --engine strata --config strata-fn-gyro-abl-gpu2.json --port 18046
