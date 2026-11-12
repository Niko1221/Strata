#!/bin/bash
cd /mnt/data4tb/Strata-gyro
export CUDA_VISIBLE_DEVICES=GPU-1f04a049-27be-1609-e94b-af36d818373c
.venv/bin/python -m serve.server --engine strata --config strata-groot-q4-gpu2.json --port 18036
