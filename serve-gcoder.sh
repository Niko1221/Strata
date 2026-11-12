#!/usr/bin/env bash
# gcoder lane: Strata serving Qwen3.8-Flash-Next-Gyro-Coder on GPU2 (3090, :18032).
#   up:   systemctl --user start strata-gcoder
#   down: systemctl --user stop  strata-gcoder
# On exit it evicts its own 70.6 GB GGUF from the page cache (evict-cache.sh
# pattern) so RAM is clean when the 3090 is needed for other work.
set -u
cd /mnt/data4tb/Strata-gyro
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
GGUF=/mnt/data4tb/Strata-data/models/gyro-coder/Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf
evict() { /mnt/data4tb/proj/fp4-trial/deepseek-vs-glm/evict-cache.sh "$GGUF"; }
trap evict EXIT INT TERM
exec .venv/bin/python -m serve.server --engine strata --config strata-gyro-coder.json --port 18032
