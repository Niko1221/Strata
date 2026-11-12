#!/usr/bin/env bash
# R5 grid row: Groot-Gyro-M on the RTX 2080 Ti (11GB tier, 4260-slot hot tier, :18043).
#   up:   systemctl --user start strata-groot-2080ti
#   down: systemctl --user stop  strata-groot-2080ti
# Evicts its own GGUFs from page cache on exit (evict-cache.sh pattern).
set -u
cd /mnt/data4tb/Strata-gyro
export CUDA_VISIBLE_DEVICES=GPU-a351101d-34cd-ef2b-8a8b-14aa2395e8ce
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
evict() {
  /mnt/data4tb/proj/fp4-trial/deepseek-vs-glm/evict-cache.sh /mnt/data4tb/models/groot-65b/Groot-65B-A6B-GyroM.gguf
  for s in /mnt/data4tb/models/groot-65b/Groot-65B-A6B-261023-Q4_K_M-0000*.gguf; do
    /mnt/data4tb/proj/fp4-trial/deepseek-vs-glm/evict-cache.sh "$s"
  done
}
trap evict EXIT INT TERM
exec .venv/bin/python -m serve.server --engine strata --config strata-groot-gyro-2080ti.json --port 18043
