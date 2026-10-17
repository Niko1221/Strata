#!/usr/bin/env bash
# Пропускная способность peer-копирования В МОМЕНТ, когда сервер модели делает префил:
# это та полоса NVLink, которая реально осталась бы движку под нагрузкой.
# Параллельно пишется трафик PCIe (nvidia-smi dmon -s t), чтобы было видно, что нагрузка была.
cd ~/strata185
export LD_LIBRARY_PATH=$HOME/strata185/cuda-12.8/lib64:$LD_LIBRARY_PATH
CFG=$HOME/strata185/Strata/strata-185-iq3_s.json
KEY=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('api_key',''))" "$CFG")
python3 -c 'print("x "*31000)' > /tmp/payload.txt
python3 - <<'PY'
import json
body = open('/tmp/payload.txt').read()
open('/tmp/req.json', 'w').write(json.dumps(
    {"messages": [{"role": "user", "content": body}], "max_tokens": 1, "temperature": 0}))
PY
nohup nvidia-smi dmon -s t -c 14 > /tmp/dmon-load.txt 2>&1 &
nohup curl -s -m 300 -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" \
  --data @/tmp/req.json > /tmp/load.out 2>&1 &
sleep 5
echo "--- проба во время префила"
/tmp/p2p-bw 2>&1 | grep -A3 "GPU1 <-> GPU2"
wait
echo "--- ответ сервера (первые 200 байт):"; head -c 200 /tmp/load.out; echo
echo "--- трафик PCIe в эти секунды (rxpci/txpci, МБ/с):"; sed -n "1,22p" /tmp/dmon-load.txt
