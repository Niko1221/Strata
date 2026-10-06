#!/usr/bin/env bash
# strata_start.sh [MAXCTX] [PORT]: start the Strata server (Coder IQ1_M pack, the user's two-shard GGUF, MTP q2_0 runtime) on this GPU.
# STRATA_DIR selects the Strata checkout (default ~/ref/Strata = 0.1.33).
# Log: /tmp/strata-coder.log (engine) and /tmp/strata-serve.out (python front end). Stop with: pkill -x strata
CTX=${1:-32768}; PORT=${2:-8081}
M=/home/user/models; D=/home/user/strata-data
source /path/to/bench-repo/rocmenv.sh
SD=${STRATA_DIR:-$HOME/ref/Strata}   # the checkout whose build-hip/strata is used (STRATA_DIR=~/ref/Strata-0.1.40 for 0.1.40)
cd $SD
cat > strata-coder.json <<EOF
{"exe":"build-hip/strata","args":["--pack","$D/packs/coder-IQ1_M","--native","$M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf","--ple-gguf","$M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf","--mmap-experts","--expert-profile","data/expert-profile-coder.bin","--expert-cache","auto","--prefill","auto","--spec","4","--spec-min-p","0.5","--mtp","$D/mtp-rt","--max-context","$CTX","--kv","int8"],"cwd":".","tokenizer":"$D/packs/coder-IQ1_M/tokenizer","model_name":"strata-coder-iq1m","log":"/tmp/strata-coder.log","host":"127.0.0.1","port":$PORT}
EOF
nohup setsid $HOME/ref/Strata/.venv/bin/python -m serve.server --engine strata --config strata-coder.json --port $PORT > /tmp/strata-serve.out 2>&1 < /dev/null &
echo started
