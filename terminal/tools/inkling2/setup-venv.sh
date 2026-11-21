#!/bin/bash
# setup-venv.sh DIR: a venv in DIR for the transformers PR #51050 oracle (CPU torch); needs git and uv on PATH
set -ex
cd "${1:?usage: setup-venv.sh DIR}"
[ -d transformers ] || git clone -q --filter=blob:none https://github.com/huggingface/transformers.git
cd transformers && git fetch -q origin pull/51050/head:pr-51050 && git checkout -q pr-51050 && git log -1 --format="%h %ad %s" --date=iso && cd ..
uv venv -p 3.12 .venv
. .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch
uv pip install -e ./transformers safetensors tiktoken numpy accelerate pillow
python -c "import torch, transformers; print(torch.__version__, transformers.__version__); from transformers import Inkling2LoopForConditionalGeneration; print(\"OK\")"
echo SETUP-DONE
