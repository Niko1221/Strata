#!/bin/bash
# Check the reference endpoint on 18790: our engine, OpenAI-compatible, same greedy text as llama.cpp.
set -uo pipefail
BASE="http://127.0.0.1:18790"

if ! curl -sf "$BASE/health" > /dev/null 2>&1; then
    echo "skipped: nothing listening on 18790 (bash scripts/start_engine_18790.sh)"
    exit 0
fi

text=$(curl -s -X POST "$BASE/v1/completions" -H 'Content-Type: application/json' \
    -d '{"prompt":"The capital of France is","max_tokens":6,"temperature":0}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])')

want=" Paris, a city renowned for"
if [ "$text" != "$want" ]; then
    echo "18790 returned '$text', expected '$want'"
    exit 1
fi

chat=$(curl -s -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":4,"temperature":0}' \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["object"], len(d["strata"]["tokens"]))')

echo "18790 completions: '$text' (matches llama.cpp)"
echo "18790 chat: $chat"
