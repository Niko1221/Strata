#!/usr/bin/env bash
# 一次性验收：双卡(5060Ti + MI50) serve 能不能起来，前端与 API 是否都可用，跑完自动收摊。
set -u
cd /g/01_Projects/Strata
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

PORT=${1:-8090}
CFG=strata-run-mi50.json
CURL="curl -s --noproxy *"

./.venv/Scripts/python.exe serve/server.py --engine strata --config "$CFG" --port "$PORT" > /tmp/dual-test.log 2>&1 &
SPID=$!
echo "== server pid $SPID, waiting for /health =="

ready=0
for i in $(seq 1 160); do
  if curl -s --noproxy '*' -m 3 "http://127.0.0.1:$PORT/health" -o /dev/null 2>/dev/null; then
    echo "== ready after $((i*3))s =="
    ready=1
    break
  fi
  sleep 3
done

if [ "$ready" != 1 ]; then
  echo "!! never became healthy; last log lines:"
  tail -30 /tmp/dual-test.log
  taskkill //PID $SPID //T //F >/dev/null 2>&1
  exit 1
fi

echo ""
echo "--- GET /health ---"
curl -s --noproxy '*' -m 10 "http://127.0.0.1:$PORT/health"; echo
echo "--- GET /v1/models ---"
curl -s --noproxy '*' -m 10 "http://127.0.0.1:$PORT/v1/models"; echo
echo "--- GET /  (the front page) ---"
curl -s --noproxy '*' -m 10 -o /tmp/dual-index.html -w "http=%{http_code} bytes=%{size_download} type=%{content_type}\n" "http://127.0.0.1:$PORT/"
head -c 180 /tmp/dual-index.html; echo
echo "--- GET /web/app.js ---"
curl -s --noproxy '*' -m 10 -o /dev/null -w "http=%{http_code} bytes=%{size_download}\n" "http://127.0.0.1:$PORT/web/app.js"

echo ""
echo "--- POST /v1/chat/completions (non-stream) ---"
curl -s --noproxy '*' -m 900 -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash-next-iq3_s-mi50","max_tokens":40,"messages":[{"role":"user","content":"用一句话说明你是哪家的模型。"}]}' \
  "http://127.0.0.1:$PORT/v1/chat/completions"; echo

echo ""
echo "--- the peer tier, from the engine's own log ---"
grep -a "peer OpenCL" strata-run-mi50.log | head -3

echo ""
echo "== stopping =="
taskkill //PID $SPID //T //F 2>/dev/null | head -3
sleep 12
echo "leftover strata.exe:"
tasklist //FI "IMAGENAME eq strata.exe" //FO CSV //NH 2>/dev/null | head -3
