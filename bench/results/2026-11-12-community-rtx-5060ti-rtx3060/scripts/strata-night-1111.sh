#!/usr/bin/env bash
# strata-night-1111.sh — ночной замер Strata, 2026-11-11 (идеи из документации PolyStrata):
#   A. layer_split 18 / 22 / 24 при боевых флагах (15–20 мерили 26.10, лучший 17; у пользователя PolyStrata #2644
#      с парой 4060 Ti + 3060 чтение промпта росло до 28 слоёв на 3060 — проверяем, есть ли это у нас);
#   B. переключатели движка по одному на split 17: STRATA_PF_FUSED, STRATA_QSA_SPLIT_ATTN, STRATA_SPLIT_MTP_BATCH,
#      STRATA_PROMPT_ATTN_IMMA, STRATA_PREFILL_HELP;
#   C. длина рассуждения: оригинал IQ3_S против Swift 1.5 IQ3_S (если подготовлен strata-swift-prepare.sh).
# Каждый вариант: холодное чтение 32k ×2 и 100k ×2 (strata-prefill.py), генерация на 3 вопросах ×2 без рассуждения.
# В C — те же 3 вопроса и 3 задачи на рассуждение с рассуждением по умолчанию: токены ответа и время.
# Итоги: <root>/logs/night-1111/<вариант>/; сводка — summary.txt. В конце возвращается strata-iq3_s.json.
# Запуск: nohup <root>/scripts/strata-night-1111.sh > <root>/logs/night-1111.out 2>&1 &
# PLAN=combo — только связка полезных переключателей (B) против base-17, итоги в $OUT (по умолчанию night-combo).
set -uo pipefail
LLM=<root>; S=$LLM/strata; PLAN=${PLAN:-full}
[ "$PLAN" = combo ] && OUT=${OUT:-$LLM/logs/night-combo} || OUT=${OUT:-$LLM/logs/night-1111}
cd $S && set -a && . $LLM/config/llm.env && set +a
mkdir -p $OUT
API=http://127.0.0.1:8070/v1/chat/completions
QUICK=("Объясни в пяти пунктах, чем отличается TCP от UDP."
       "Напиши на Python функцию бинарного поиска с комментариями."
       "Explain how a B-tree index works in a database, step by step.")
THINK=("В комнате три выключателя, в соседней — три лампы. Зайти к лампам можно один раз. Как узнать, какой выключатель к какой лампе?"
       "Найди все целые x, при которых x^2 - 7x + 12 делится на x - 2. Объясни решение."
       "В таблице orders (id, user_id, created_at, total) найди для каждого пользователя второй по величине заказ. Напиши SQL для PostgreSQL и объясни, как он работает с одинаковыми суммами.")

wait_loaded() {
  for _ in $(seq 1 120); do
    curl -s -m5 -H "Authorization: Bearer $LLM_API_KEY" http://127.0.0.1:8070/v1/status | grep -q '"loaded": true' && return 0
    sleep 10
  done
  echo "НЕ ЗАГРУЗИЛСЯ за 20 минут"; return 1
}
restart_with() { ln -sfn "$1" $S/strata-active.json && sudo systemctl restart strata && wait_loaded; }

# ask <файл jsonl> <метка> <вопрос> <max_tokens> <think 0|1>
ask() {
  python3 - "$@" <<'EOF'
import json, os, sys, time, urllib.request
out, tag, q, mt, think = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5] == "1"
body = {"messages": [{"role": "user", "content": q}], "max_tokens": mt, "temperature": 0, "seed": 1234}
if not think:
    body["chat_template_kwargs"] = {"enable_thinking": False}
req = urllib.request.Request("http://127.0.0.1:8070/v1/chat/completions", json.dumps(body).encode(),
                             headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["LLM_API_KEY"]})
t0 = time.time()
try:
    r = json.load(urllib.request.urlopen(req, timeout=1800))
except Exception as e:
    print(f"{tag}: ОШИБКА {e}"); sys.exit(0)
t, u, m = r.get("timings", {}), r.get("usage", {}), r["choices"][0]["message"]
row = {"tag": tag, "q": q[:40], "think": think, "decode": t.get("predicted_per_second"),
       "completion": u.get("completion_tokens"), "reasoning_chars": len(m.get("reasoning_content") or ""),
       "answer_chars": len(m.get("content") or ""), "finish": r["choices"][0].get("finish_reason"),
       "wall": round(time.time() - t0, 1)}
open(out, "a").write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"{tag}: decode {row['decode']} tok/s, {row['completion']} токенов, {row['wall']} с, {row['finish']}", flush=True)
EOF
}

# variant <имя> <база-конфиг> <layer_split> [ENV=VAL ...]
variant() {
  local name=$1 base=$2 split=$3; shift 3
  local O=$OUT/$name; mkdir -p $O
  python3 - "$base" "$split" "$@" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
c["layer_split"] = sys.argv[2]
for kv in sys.argv[3:]:
    k, v = kv.split("=", 1)
    c.setdefault("env", {})[k] = v
json.dump(c, open("strata-night-test.json", "w"), indent=1)
EOF
  echo "== [$(date +%T)] $name: split $split $*"
  restart_with strata-night-test.json || return 1
  local log; log=$(python3 -c 'import json;print(json.load(open("strata-night-test.json"))["log"])')
  grep -E "layer split: CUDA[01] runs" "$log" | tail -2 > $O/split.txt
  .venv/bin/python $LLM/scripts/strata-prefill.py $O > $O/prefill.out 2>&1 || echo "prefill: ошибка, см. $O/prefill.out"
  for rep in 1 2; do for q in "${QUICK[@]}"; do ask $O/decode.jsonl "$name/quick" "$q" 400 0; done; done
}

think_run() {  # think_run <имя> <конфиг>
  local O=$OUT/$1; mkdir -p $O
  echo "== [$(date +%T)] $1: длина рассуждения"
  restart_with "$2" || return 1
  for q in "${QUICK[@]}" "${THINK[@]}"; do ask $O/think.jsonl "$1/think" "$q" 16000 1; done
}

# при любом выходе — боевой конфиг (если прогон оборвался посередине)
trap 'ln -sfn strata-iq3_s.json $S/strata-active.json' EXIT

if [ "$PLAN" = combo ]; then
  # переключатели, давшие +1…5% к чтению промпта по одному (night-1111); base до и после — проверка дрейфа
  variant base-17 strata-iq3_s.json 17
  variant combo3 strata-iq3_s.json 17 STRATA_SPLIT_MTP_BATCH=1 STRATA_PROMPT_ATTN_IMMA=1 STRATA_PF_FUSED=1
  variant combo2 strata-iq3_s.json 17 STRATA_SPLIT_MTP_BATCH=1 STRATA_PROMPT_ATTN_IMMA=1
  variant base-17-end strata-iq3_s.json 17
else
# A. граница слоёв
for k in 18 22 24; do variant split-$k strata-iq3_s.json $k; done
# B. переключатели на боевой границе
variant base-17 strata-iq3_s.json 17
for e in STRATA_PF_FUSED STRATA_QSA_SPLIT_ATTN STRATA_SPLIT_MTP_BATCH STRATA_PROMPT_ATTN_IMMA STRATA_PREFILL_HELP; do
  variant env-$e strata-iq3_s.json 17 $e=1
done
# C. оригинал против Swift
think_run think-iq3_s strata-iq3_s.json
if [ -f $LLM/Strata-data/packs/swift-iq3_s/native_experts.txt ]; then
  variant swift-17 strata-swift-iq3_s.json 17
  think_run think-swift strata-swift-iq3_s.json
else
  echo "Swift не подготовлен (нет pack) — часть C только для оригинала"
fi
fi

restart_with strata-iq3_s.json && echo "ВОЗВРАЩЁН БОЕВОЙ КОНФИГ"
python3 - $OUT <<'EOF' | tee $OUT/summary.txt
import json, pathlib, statistics as st, sys
root = pathlib.Path(sys.argv[1])
print(f"{'вариант':34} {'32k':>7} {'100k':>7} {'decode':>7}")
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    pf = [json.loads(l) for l in open(d / "prefill.jsonl")] if (d / "prefill.jsonl").exists() else []
    med = lambda s: round(st.median([r["prompt"] for r in pf if r["size"] == s and r["prompt"]]), 0) if any(r["size"] == s for r in pf) else "-"
    dec = [json.loads(l)["decode"] for l in open(d / "decode.jsonl")] if (d / "decode.jsonl").exists() else []
    if pf or dec:
        print(f"{d.name:34} {med('32k'):>7} {med('100k'):>7} {round(st.median([x for x in dec if x]), 1) if dec else '-':>7}")
    if (d / "think.jsonl").exists():
        rows = [json.loads(l) for l in open(d / "think.jsonl")]
        print(f"{d.name:34} рассуждение: всего {sum(r['completion'] or 0 for r in rows)} токенов, "
              f"{round(sum(r['wall'] for r in rows))} с, обрывов {sum(r['finish'] == 'length' for r in rows)}")
EOF
