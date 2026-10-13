#!/usr/bin/env bash
# 验收 RUN-DUAL.bat：双击式的双卡入口，起来后前端和 API 都要能用，跑完收摊。
set -u
cd /g/01_Projects/Strata
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export STRATA_NO_OPEN=1

PORT=8080
cmd //c RUN-DUAL.bat > /tmp/dual-bat.log 2>&1 &
WID=$!
echo "== launcher pid $WID =="

ready=0
for i in $(seq 1 170); do
  if curl -s --noproxy '*' -m 3 "http://127.0.0.1:$PORT/health" -o /dev/null 2>/dev/null; then
    echo "== ready after $((i*3))s =="
    ready=1
    break
  fi
  sleep 3
done

if [ "$ready" != 1 ]; then
  echo "!! never healthy; launcher log:"
  tail -30 /tmp/dual-bat.log
  taskkill //PID $WID //T //F >/dev/null 2>&1
  exit 1
fi

echo ""
echo "--- 启动器自己打印了什么 ---"
sed -n '1,20p' /tmp/dual-bat.log

echo ""
echo "--- 前端页面 ---"
curl -s --noproxy '*' -m 10 -o /dev/null -w "GET /            http=%{http_code} bytes=%{size_download}\n" "http://127.0.0.1:$PORT/"
curl -s --noproxy '*' -m 10 -o /dev/null -w "GET /web/app.js  http=%{http_code} bytes=%{size_download}\n" "http://127.0.0.1:$PORT/web/app.js"

echo ""
echo "--- API：非流式 ---"
curl -s --noproxy '*' -m 900 -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash-next-iq3_s-mi50","max_tokens":32,"messages":[{"role":"user","content":"一句话：你是谁？"}]}' \
  "http://127.0.0.1:$PORT/v1/chat/completions" | head -c 400; echo

echo ""
echo "--- API：流式（前 3 个 chunk）---"
curl -s --noproxy '*' -m 900 -N -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash-next-iq3_s-mi50","stream":true,"max_tokens":24,"messages":[{"role":"user","content":"数到三"}]}' \
  "http://127.0.0.1:$PORT/v1/chat/completions" | head -3

echo ""
echo "--- 引擎日志里的 peer 行 ---"
grep -a "peer OpenCL" strata-run-mi50.log | tail -2

echo ""
echo "== stopping =="
taskkill //PID $WID //T //F >/dev/null 2>&1
for i in $(seq 1 60); do
  if ! tasklist //FI "IMAGENAME eq strata.exe" //FO CSV //NH 2>/dev/null | grep -q strata; then
    echo "== engine gone after $((i*2))s =="
    break
  fi
  sleep 2
done
echo "leftover:"
tasklist //FI "IMAGENAME eq strata.exe" //FO CSV //NH 2>/dev/null | head -3
