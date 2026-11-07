#!/usr/bin/env bash
# R2 grid row: Gyro-Coder TQ2_0 native layer-split across RTX 2080 Ti + RTX 3090 (:18037).
set -u
cd /mnt/data4tb/Strata-gyro
export CUDA_VISIBLE_DEVICES=GPU-a351101d-34cd-ef2b-8a8b-14aa2395e8ce,GPU-71d31235-62ca-4433-aad2-9aa2038164df
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
evict() { /mnt/data4tb/proj/fp4-trial/deepseek-vs-glm/evict-cache.sh /mnt/data4tb/Strata-data/models/gyro-coder/Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf; }
trap evict EXIT INT TERM
exec .venv/bin/python -m serve.server --engine strata --config strata-gyro-coder-pair.json --port 18037
