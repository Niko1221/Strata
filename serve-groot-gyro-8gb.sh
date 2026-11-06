#!/bin/bash
cd /mnt/data4tb/Strata-gyro
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m serve.server --engine strata --config strata-groot-gyro-8gb.json --port 18042
