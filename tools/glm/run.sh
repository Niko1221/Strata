#!/bin/sh
# One-command chat with GLM-5.3-Flash on Strata (the ported strata-glm engine).
#
#   ./run-glm.sh                     # chat (needs the model's pack, built once by iq_pack.py)
#   ./run-glm.sh --kv fp16           # float16 latent cache instead of INT8
#   ./run-glm.sh --max-context 131072 --gpus 0,1 --temperature 0.7
#
# The pack is <model-folder>/glm-pack (tools/iq_pack.py --compat-bf16 output, ~900 MB beside the
# shards; the engine reads the expert weights from the shard files themselves). If it is missing
# this script builds it first.
set -e
cd "$(dirname "$0")/../.."

PACK=${STRATA_GLM_PACK:-/nvme/models/GLM-5.3-Flash-UD-Q2_K_XL/glm-pack}
SHARDS=${STRATA_GLM_GGUF_DIR:-/nvme/models/GLM-5.3-Flash-UD-Q2_K_XL/UD-Q2_K_XL}
PY=${STRATA_PYTHON:-python3}

if [ ! -f build/strata-glm ] && [ -z "$STRATA_GLM_EXE" ]; then
    echo "strata-glm is not built yet: cmake -S . -B build -DSTRATA_BUILD_GLM=ON && ninja -C build strata-glm" >&2
    exit 1
fi
if [ ! -f "$PACK/index.txt" ]; then
    echo "No pack at $PACK - building it once with tools/iq_pack.py (reads every shard; the weights stay where they are)..."
    STRATA_GGUF_PY=${STRATA_GGUF_PY:-/nvme/strata-deps/llama.cpp/gguf-py} \
        $PY tools/iq_pack.py --gguf "$SHARDS"/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf \
        --out "$PACK" --compat-bf16
fi
exec $PY tools/glm/chat.py --pack "$PACK" "$@"
