#!/usr/bin/env bash
# strata-night-0213.sh — ночное сравнение Strata: боевой движок 0.1.45.3 против сборки v0.2.13.1 (движок 0.1.59)
# с перенесённым дисковым кэшем бесед (<root>/strata-0213, ветка conv-cache-disk-0213).
#   prod        — боевой сервис strata (strata-active.json, как есть), движок 0.1.45.3;
#   v0213-mtp   — 0.2.13 с MTP (strata-0213/strata-iq3_s.json, те же флаги и env);
#   v0213-df2   — 0.2.13 с DFlash2 (strata-iq3_s-dflash2.json; без --pipeline-windows — DFlash2 с ними не работает),
#                 только если в <root>/Strata-data/dflash2 есть пакет драфтера;
#   v0213-disk  — проверка дискового кэша бесед на новой сборке: RAM-кэш 256 МиБ / 1 слот, две беседы по ~30k токенов
#                 чередуются A1 B1 A2 B2; в логе должны быть «conversation disk: wrote» и «read ... reused».
# Каждый вариант (кроме disk): холодное чтение 8k/32k/100k ×2 (strata-prefill.py), генерация на 3 вопросах ×2 без рассуждения.
# Новая версия запускается не сервисом strata (его ExecStart не трогаем), а временным юнитом systemd-run strata-0213
# с теми же ключами (llm.env, strata.env) и LimitMEMLOCK=infinity, как в server/etc/systemd/system/strata.service.
# В конце временный юнит гасится и поднимается боевой strata. Итоги: <root>/logs/night-0213/<вариант>/, summary.txt.
# Запуск: nohup <root>/scripts/strata-night-0213.sh > <root>/logs/night-0213.out 2>&1 &
# PLAN=c3 — те же варианты (без disk) с тремя переключателями из night-combo (+9% к чтению 32k на боевой сборке)
# и с тремя задачами на рассуждение: скорость генерации драфтеров на длинном ответе. Итоги в night-0213b.
set -uo pipefail
# PLAN=disk — только проверка дискового кэша для сборки в N (например N=<root>/strata-pr), итоги в $OUT.
LLM=<root>; S=$LLM/strata; N=${N:-$LLM/strata-0213}; PLAN=${PLAN:-full}
[ "$PLAN" = c3 ] && OUT=${OUT:-$LLM/logs/night-0213b} || OUT=${OUT:-$LLM/logs/night-0213}
C3=(STRATA_SPLIT_MTP_BATCH=1 STRATA_PROMPT_ATTN_IMMA=1 STRATA_PF_FUSED=1)
cd $S && set -a && . $LLM/config/llm.env && set +a
mkdir -p $OUT
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
stop_new() {
  sudo systemctl stop strata-0213 2>/dev/null
  sudo systemctl reset-failed strata-0213 2>/dev/null
  for _ in $(seq 1 30); do ss -ltn | grep -q ':8070 ' || return 0; sleep 2; done
  echo "порт 8070 не освободился"; return 1
}
start_prod() { stop_new; sudo systemctl restart strata && wait_loaded; }
# with_env <исходный конфиг> <новый конфиг> KEY=VAL...: копия конфига с добавленными переменными окружения
with_env() {
  python3 - "$@" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
for kv in sys.argv[3:]:
    k, v = kv.split("=", 1)
    c.setdefault("env", {})[k] = v
json.dump(c, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
EOF
}
# start_prod_cfg <конфиг в $S>: боевой сервис с другим конфигом (в конце скрипта ссылка возвращается)
start_prod_cfg() { stop_new; ln -sfn "$1" $S/strata-active.json && sudo systemctl restart strata && wait_loaded; }
# start_new <конфиг в $N>: боевой сервис гасится, новая версия поднимается временным юнитом (ключ не попадает в argv)
start_new() {
  sudo systemctl stop strata; stop_new || return 1
  sudo systemd-run --unit=strata-0213 --collect -p User=user -p Group=user \
    -p EnvironmentFile=$LLM/config/llm.env -p EnvironmentFile=-$LLM/config/strata.env \
    -p LimitMEMLOCK=infinity -p LimitNOFILE=65536 -p WorkingDirectory=$N \
    -E PATH=/usr/local/cuda-13.1/bin:/usr/local/bin:/usr/bin:/bin \
    /bin/sh -c "STRATA_API_KEY=\"\$LLM_API_KEY,\$STRATA_UI_KEY\" exec $N/.venv/bin/python serve/server.py --engine strata --config $1 --host 0.0.0.0 --port 8070" \
    >/dev/null || return 1
  wait_loaded
}

# ask <файл jsonl> <метка> <max_tokens> <think 0|1> <сообщения JSON или текст вопроса>  — печатает ответ в $ASK_REPLY (файл)
ASK_REPLY=$OUT/.last-reply
ask() {
  python3 - "$ASK_REPLY" "$@" <<'EOF'
import json, os, sys, time, urllib.request
reply, out, tag, mt, think, q = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5] == "1", sys.argv[6]
msgs = json.loads(q) if q.startswith("[") else [{"role": "user", "content": q}]
body = {"messages": msgs, "max_tokens": mt, "temperature": 0, "seed": 1234}
if not think:
    body["chat_template_kwargs"] = {"enable_thinking": False}
req = urllib.request.Request("http://127.0.0.1:8070/v1/chat/completions", json.dumps(body).encode(),
                             headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["LLM_API_KEY"]})
t0 = time.time()
try:
    r = json.load(urllib.request.urlopen(req, timeout=1800))
except Exception as e:
    print(f"{tag}: ОШИБКА {e}"); open(reply, "w").write(""); sys.exit(0)
t, u, m = r.get("timings", {}), r.get("usage", {}), r["choices"][0]["message"]
open(reply, "w").write(m.get("content") or "")
row = {"tag": tag, "q": msgs[-1]["content"][-40:], "think": think, "decode": t.get("predicted_per_second"),
       "prompt": t.get("prompt_per_second"), "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
       "completion": u.get("completion_tokens"), "answer_chars": len(m.get("content") or ""),
       "finish": r["choices"][0].get("finish_reason"), "wall": round(time.time() - t0, 1)}
open(out, "a").write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"{tag}: decode {row['decode']} tok/s, промпт {row['prompt_n']} (кэш {row['cache_n']}), {row['completion']} токенов, "
      f"{row['wall']} с, {row['finish']}", flush=True)
EOF
}

engine_log() { python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["log"])' "$1"; }

# bench <имя> <start-команда...>
bench() {
  local name=$1; shift
  local O=$OUT/$name; mkdir -p $O
  echo "== [$(date +%T)] $name"
  "$@" || { echo "$name: не стартовал"; return 1; }
  curl -s -m5 -H "Authorization: Bearer $LLM_API_KEY" http://127.0.0.1:8070/v1/status > $O/status.json
  $S/.venv/bin/python $LLM/scripts/strata-prefill.py $O > $O/prefill.out 2>&1 || echo "prefill: ошибка, см. $O/prefill.out"
  for rep in 1 2; do for q in "${QUICK[@]}"; do ask $O/decode.jsonl "$name/quick" 400 0 "$q"; done; done
  if [ "$PLAN" = c3 ]; then for q in "${THINK[@]}"; do ask $O/think.jsonl "$name/think" 16000 1 "$q"; done; fi
}

# Дисковый кэш бесед на новой сборке: RAM-кэш мал, беседа уходит на диск сразу (direct) или при вытеснении,
# следующий её ход читается с диска.
disk_check() {
  local O=$OUT/v0213-disk; mkdir -p $O
  echo "== [$(date +%T)] v0213-disk"
  python3 - $N/strata-iq3_s.json $N/strata-night-disk.json <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
a = c["args"]
for flag, val in (("--conversation-cache-mib", "256"), ("--conversation-cache-slots", "1")):
    a[a.index(flag) + 1] = val
c["log"] = sys.argv[2].replace(".json", ".log")
json.dump(c, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
EOF
  local log; log=$(engine_log $N/strata-night-disk.json)
  local from; from=$( [ -f "$log" ] && wc -c < "$log" || echo 0)
  start_new strata-night-disk.json || { echo "v0213-disk: не стартовал"; return 1; }
  [ "$from" -gt "$(wc -c < "$log")" ] && from=0          # лог переписан заново при старте
  # A1 B1 A2 B2: две беседы по ~30k токенов (разные куски стога), второй ход каждой продолжает первый
  python3 - $O/turns.jsonl $LLM/bench/prompts/hay-680k.txt <<'EOF2'
import json, os, sys, time, urllib.request, uuid
out, hay = sys.argv[1], open(sys.argv[2], encoding="utf-8").read()
def post(msgs, tag):
    body = {"messages": msgs, "max_tokens": 16, "temperature": 0, "seed": 1234,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:8070/v1/chat/completions", json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["LLM_API_KEY"]})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=1800))
    except Exception as e:
        print(f"{tag}: ОШИБКА {e}", flush=True); return ""
    t = r.get("timings", {})
    row = {"tag": tag, "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "wall": round(time.time() - t0, 1)}
    open(out, "a").write(json.dumps(row) + "\n")
    print(f"{tag}: промпт {row['prompt_n']}, из кэша {row['cache_n']}, {row['wall']} с", flush=True)
    return r["choices"][0]["message"].get("content") or ""
q1, q2 = "Назови одним словом язык этого кода.", "Какая функция в этом коде самая длинная? Ответь одним названием."
conv = {"A": f"[A {uuid.uuid4()}]\n" + hay[:100000], "B": f"[B {uuid.uuid4()}]\n" + hay[200000:300000]}
first = {}
for k in "AB":
    first[k] = [{"role": "user", "content": conv[k] + "\n\n" + q1}]
    first[k].append({"role": "assistant", "content": post(first[k], k + "1")})
for k in "AB":
    post(first[k] + [{"role": "user", "content": q2}], k + "2")
EOF2
  sleep 2
  tail -c +$((from + 1)) "$log" | grep -aE "conversation (disk|cache):" > $O/disk-lines.txt
  local w r
  w=$(grep -ac "conversation disk: wrote" $O/disk-lines.txt); r=$(grep -aE -c "conversation disk: read .*reused" $O/disk-lines.txt)
  echo "v0213-disk: записей на диск $w, чтений с диска $r" | tee $O/verdict.txt
  if [ "$w" -ge 1 ] && [ "$r" -ge 1 ]; then echo "ДИСКОВЫЙ КЭШ РАБОТАЕТ" | tee -a $O/verdict.txt
  else echo "ДИСКОВЫЙ КЭШ НЕ СРАБОТАЛ — см. $O/disk-lines.txt" | tee -a $O/verdict.txt; fi
}

ACTIVE=$(readlink $S/strata-active.json)
echo "боевой конфиг: $ACTIVE"
# при любом выходе — боевой конфиг (если прогон оборвался посередине; сервис поднимает вызывающая цепочка)
trap '[ "$(readlink $S/strata-active.json)" = "$ACTIVE" ] || ln -sfn "$ACTIVE" $S/strata-active.json' EXIT
if [ "$PLAN" = disk ]; then
  disk_check
elif [ "$PLAN" = c3 ]; then
  with_env $S/$ACTIVE $S/strata-night-c3.json "${C3[@]}"
  with_env $N/strata-iq3_s.json $N/strata-c3.json "${C3[@]}"
  with_env $N/strata-iq3_s-dflash2.json $N/strata-c3-dflash2.json "${C3[@]}"
  bench prod-c3 start_prod_cfg strata-night-c3.json
  bench v0213-mtp-c3 start_new strata-c3.json
  bench v0213-df2-c3 start_new strata-c3-dflash2.json
  grep -aE "DFlash2|drafting|draft-conf" "$(engine_log $N/strata-c3-dflash2.json)" | tail -5 > $OUT/v0213-df2-c3/drafter.txt
else
bench prod start_prod
bench v0213-mtp start_new strata-iq3_s.json
if ls $LLM/Strata-data/dflash2/*.pack >/dev/null 2>&1; then
  bench v0213-df2 start_new strata-iq3_s-dflash2.json
  grep -aE "DFlash2|drafting|draft-conf" "$(engine_log $N/strata-iq3_s-dflash2.json)" | tail -5 > $OUT/v0213-df2/drafter.txt
else
  echo "DFlash2: пакета драфтера нет в $LLM/Strata-data/dflash2 — вариант пропущен"
fi
disk_check
fi

stop_new
[ "$(readlink $S/strata-active.json)" = "$ACTIVE" ] || ln -sfn "$ACTIVE" $S/strata-active.json
sudo systemctl start strata && wait_loaded && echo "ВОЗВРАЩЁН БОЕВОЙ СЕРВИС ($ACTIVE)"
python3 - $OUT <<'EOF' | tee $OUT/summary.txt
import json, pathlib, statistics as st, sys
root = pathlib.Path(sys.argv[1])
print(f"{'вариант':16} {'8k':>7} {'32k':>7} {'100k':>7} {'decode':>7}")
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    pf = [json.loads(l) for l in open(d / "prefill.jsonl")] if (d / "prefill.jsonl").exists() else []
    med = lambda s: round(st.median([r["prompt"] for r in pf if r["size"] == s and r["prompt"]]), 0) if any(r["size"] == s and r["prompt"] for r in pf) else "-"
    dec = [json.loads(l)["decode"] for l in open(d / "decode.jsonl")] if (d / "decode.jsonl").exists() else []
    if pf or dec:
        print(f"{d.name:16} {med('8k'):>7} {med('32k'):>7} {med('100k'):>7} {round(st.median([x for x in dec if x]), 1) if any(dec) else '-':>7}")
    if (d / "verdict.txt").exists():
        print(f"{d.name:16} " + " / ".join(open(d / "verdict.txt").read().split("\n")).strip(" /"))
        for l in open(d / "turns.jsonl") if (d / "turns.jsonl").exists() else []:
            r = json.loads(l)
            print(f"{'':16} {r['tag']}: промпт {r['prompt_n']}, из кэша {r['cache_n']}, {r['wall']} с")
    if (d / "think.jsonl").exists():
        rows = [json.loads(l) for l in open(d / "think.jsonl")]
        dec = [r["decode"] for r in rows if r["decode"]]
        print(f"{d.name:16} рассуждение: decode медиана {round(st.median(dec), 1) if dec else '-'} tok/s, "
              f"{sum(r['completion'] or 0 for r in rows)} токенов, {round(sum(r['wall'] for r in rows))} с, "
              f"обрывов {sum(r['finish'] == 'length' for r in rows)}")
EOF
