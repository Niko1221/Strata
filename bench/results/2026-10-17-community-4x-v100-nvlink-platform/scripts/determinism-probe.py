#!/usr/bin/env python3
"""Диагностика недетерминированности жадной генерации (продолжение T1/T2 из correctness-integration.py).

Один и тот же промпт фиксированного содержимого, temperature=0, длина вывода фиксирована (R9).
Сохраняются полные тексты ответов; считается общая префиксная длина и позиция первого расхождения.
"""
import json, os, sys, hashlib, datetime, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
mod = import_module("correctness-integration")

out_dir = sys.argv[1] if len(sys.argv) > 1 else "raw/correctness/" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
os.makedirs(out_dir, exist_ok=True)

def probe(target_tokens, reps, mx=128, tag=""):
    p = mod.fixed_prompt(target_tokens)
    runs = []
    for i in range(reps):
        r = mod.gen(p, mx)
        runs.append(r)
        print(f"  run {i+1}: http={r['http']} prompt_n={r.get('prompt_n')} cached={r.get('cached')} "
              f"completion={r.get('completion')} finish={r.get('finish')} sha={r.get('text_sha')} "
              f"draft={r.get('draft_ok')}/{r.get('draft_n')}", flush=True)
    texts = [r.get("text") or "" for r in runs]
    # общая префиксная длина по всем парам
    pairs = []
    for a, b in itertools.combinations(range(reps), 2):
        k = 0
        while k < min(len(texts[a]), len(texts[b])) and texts[a][k] == texts[b][k]:
            k += 1
        pairs.append({"pair": f"{a+1}-{b+1}", "common_prefix_chars": k,
                      "len_a": len(texts[a]), "len_b": len(texts[b])})
    return {"prompt_tokens_target": target_tokens, "max_tokens": mx, "tag": tag,
            "prompt_sha": hashlib.sha256(p.encode()).hexdigest()[:16],
            "prompt_n": runs[0].get("prompt_n"),
            "cached": [r.get("cached") for r in runs],
            "shas": [r.get("text_sha") for r in runs],
            "distinct_sha": len(set(r.get("text_sha") for r in runs)),
            "draft_accepted": [f"{r.get('draft_ok')}/{r.get('draft_n')}" for r in runs],
            "pairwise_prefix": pairs,
            "texts": texts,
            "first_200_chars": [t[:200] for t in texts]}

res = {"started_utc": mod.now_utc(), "probes": []}
for tgt, reps, tag in [(64, 5, "short"), (1024, 4, "medium"), (8192, 3, "long")]:
    print(f"=== probe {tag}: target {tgt} tokens, {reps} повторов ===", flush=True)
    res["probes"].append(probe(tgt, reps, tag=tag))
res["finished_utc"] = mod.now_utc()
json.dump(res, open(out_dir + "/divergence-probe.json", "w"), ensure_ascii=False, indent=1)
print("\n=== сводка ===")
for p in res["probes"]:
    print(f"{p['tag']:>7} prompt_n={p['prompt_n']} distinct_sha={p['distinct_sha']} cached={p['cached']}")
    for pr in p["pairwise_prefix"]:
        print(f"    {pr['pair']}: common_prefix_chars={pr['common_prefix_chars']} (len {pr['len_a']}/{pr['len_b']})")
print("готово:", out_dir)
