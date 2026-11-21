#!/bin/bash
# venv for the transformers PR #51050 oracle (CPU torch). Log: /var/ramdisk/inkling/venv.log
set -ex
cd /var/cheapssd/inkling-oracle
export PATH=/root/.local/bin:$PATH
[ -d transformers ] || git clone -q --filter=blob:none https://github.com/huggingface/transformers.git
cd transformers && git fetch -q origin pull/51050/head:pr-51050 && git checkout -q pr-51050 && git log -1 --format="%h %ad %s" --date=iso && cd ..
uv venv -p /opt/miniconda3/bin/python3.12 .venv
. .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch
uv pip install -e ./transformers safetensors tiktoken numpy accelerate pillow
python -c "import torch, transformers; print(torch.__version__, transformers.__version__); from transformers import Inkling2LoopForConditionalGeneration; print(\"OK\")"
echo SETUP-DONE
