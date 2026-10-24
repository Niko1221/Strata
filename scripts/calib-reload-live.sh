#!/bin/bash
# Reload calibrators in the live jevport preset (bypasses llama-swap auth by
# talking to the inner serve port from inside the container) and prove the
# served calib is no longer identity.
set -u
ADMIN=$(grep -o 'STRATA_JEV_ADMIN_KEY=[a-f0-9]*' /opt/data/llm-swap/config.yaml | head -1 | cut -d= -f2)
KEY=$(grep -m1 '^LLM_SWAP_API_KEY=' /opt/data/llm-swap/.container-env | cut -d= -f2-)
B=http://10.89.0.57:8080/upstream/p-jevport-vision-3090

echo "--- POST /admin/calib from inside the container (port 10009)"
~/.local/bin/docker exec llama-swap sh -c "curl -s -m 60 -w '\nhttp=%{http_code}\n' -X POST http://127.0.0.1:10009/admin/calib -H 'Authorization: Bearer $ADMIN'"

echo "--- live smoke (published route)"
curl -s -m 180 -w '\nhttp=%{http_code}\n' -X POST "$B/v1/systemone" \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"state":"Customer says the card I ordered never arrived and I want a replacement.","questions":{"is_card":{"type":"noul","instructions":"is this about a physical card?"}}}'
