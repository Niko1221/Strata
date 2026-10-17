#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Этап VIII (п. 11, §9.3): транспортная приёмка патча ДО бенчмарка.

Правило приёмки (§8.1 patch-recommendations.md): перенос данных не имеет права менять биты
вывода. Поэтому любое сравнение «до/после» начинается не со скоростей, а с text_sha на
устойчивых промптах фиксированного содержимого (тот же fixed_prompt, что в Этапе VI).

Прогон (редакция 2, 2026-10-17T05:50Z): состояние движка после старта прогревается НЕ одним
запросом. Факт прогона 2026-10-17T05:46Z (raw/ab/2026-10-17T054547Z-validation/A0/): после
одного прогрева повтор 1 и повтор 2 разошлись побитово (общий префикс 559 из ~630 символов),
а hit rate экспертного кэша по строкам движка за это время поднялся 98.8 % -> 99.5 % -> 99.8 %
(файловый уровень ещё участвует). Поэтому приёмка строится так: запросы повторяются до
появления ДВУХ ПОДРЯД идентичных text_sha (не более ATTEMPTS запросов); первые из них —
прогрев по R11, устойчивая пара — отпечаток ячейки. Если устойчивость не наступила, приёмка
не пройдена и бенчмарк не проводится.

Сравнение с эталоном предыдущей ячейки (--ref) фиксируется отдельно от внутренней
устойчивости: несовпадение с эталоном означает «текст не воспроизводится между перезапусками
(или патч изменил вычисление)». По умолчанию это FAIL и бенчмарк не проводится; флаг
--allow-ref-mismatch понижает до предупреждения (нужно для валидации метода на одном и том же
бинарнике: там отсутствие патча, и расхождение само является предметом измерения).

usage: ab-text-check.py --out DIR [--targets 1024,8192] [--attempts 6] [--max-tokens 128]
                        [--ref PATH_к_текст-чек_предыдущей_ячейки] [--allow-ref-mismatch]
"""
import argparse
import datetime
import itertools
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
mod = __import__("correctness-integration")


def common_prefix(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", default="1024,8192")
    ap.add_argument("--attempts", type=int, default=6,
                    help="максимум запросов на цель; стоп при двух подряд одинаковых sha")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--ref", default="")
    ap.add_argument("--allow-ref-mismatch", action="store_true")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = {}
    if a.ref and os.path.exists(a.ref):
        ref = json.load(open(a.ref, encoding="utf-8"))
    ref_by_target = {p["target_tokens"]: p for p in ref.get("probes", [])}

    res = {"kind": "ab-text-check", "protocol": "stabilize-then-compare (редакция 2)",
           "started_utc": mod.now_utc(), "attempts_max": a.attempts,
           "max_tokens": a.max_tokens, "ref": a.ref or None,
           "allow_ref_mismatch": bool(a.allow_ref_mismatch), "probes": []}
    overall_ok = True
    ref_mismatch_seen = False
    for tgt in [int(x) for x in a.targets.split(",")]:
        prompt = mod.fixed_prompt(tgt)
        runs = []
        stab_at = None
        for i in range(a.attempts):
            r = mod.gen(prompt, a.max_tokens)
            runs.append({k: r.get(k) for k in ("http", "prompt_n", "cached", "completion", "finish",
                                               "text_sha", "text_len", "prompt_s", "gen_s")} |
                        {"text": r.get("text") or ""})
            print(f"  probe {i+1}/{a.attempts} target={tgt}: http={r.get('http')} "
                  f"prompt_n={r.get('prompt_n')} gen={r.get('completion')} finish={r.get('finish')} "
                  f"sha={r.get('text_sha')}", flush=True)
            if i >= 1 and runs[i]["text_sha"] == runs[i - 1]["text_sha"] and \
                    all(runs[j]["http"] == 200 for j in (i - 1, i)):
                stab_at = i
                print(f"  target={tgt}: устойчивая пара на запросах {i}/{i+1} "
                      f"(запросы 1..{i-1} — прогрев по R11)", flush=True)
                break
        shas = [r["text_sha"] for r in runs]
        self_ok = stab_at is not None
        # сравнение с эталоном
        refp = ref_by_target.get(tgt)
        ref_cmp = None
        ref_ok = None
        if refp:
            ref_shas = refp.get("shas") or []
            ref_sha = ref_shas[-1] if ref_shas else None
            ref_text = (refp.get("texts") or [""])[-1]
            last = runs[-1]
            pref = common_prefix(last["text"], ref_text) if ref_text else 0
            ref_cmp = {"ref_shas": ref_shas, "sha_match": bool(ref_sha) and last["text_sha"] == ref_sha,
                       "common_prefix_chars": pref, "len_ref": len(ref_text),
                       "len_new": max(len(r["text"]) for r in runs),
                       "prefix_ratio": round(pref / max(1, max(len(ref_text), max(len(r["text"]) for r in runs))), 4)}
            ref_ok = ref_cmp["sha_match"]
        pairs = [{"pair": f"{i+1}-{j+1}", "common_prefix_chars": common_prefix(runs[i]["text"], runs[j]["text"])}
                 for i, j in itertools.combinations(range(len(runs)), 2)]
        probe = {"target_tokens": tgt, "prompt_sha": __import__("hashlib").sha256(prompt.encode()).hexdigest()[:16],
                 "prompt_n": runs[0].get("prompt_n"), "shas": shas, "self_consistent": self_ok,
                 "stable_pair_at": stab_at, "requests_used": len(runs),
                 "stable_sha": (runs[-1]["text_sha"] if self_ok else None),
                 "warmup_requests": (stab_at - 1) if stab_at else None,
                 "fixed_length_ok": all(r.get("completion") == a.max_tokens and r.get("finish") == "length" for r in runs),
                 "pairwise_prefix": pairs, "ref_compare": ref_cmp, "ref_match": ref_ok,
                 "texts": [r["text"] for r in runs]}
        res["probes"].append(probe)
        if not self_ok:
            overall_ok = False
        if ref_ok is False:
            ref_mismatch_seen = True
        print(f"  target={tgt}: self_consistent={self_ok} ref_match={ref_ok}", flush=True)

    res["finished_utc"] = mod.now_utc()
    res["verdict"] = ("PASS" if overall_ok else
                      "FAIL: побитовая устойчивость внутри ячейки не наступила за %d запросов" % a.attempts)
    if overall_ok and ref_mismatch_seen:
        res["verdict"] = ("PASS_WITH_WARNING: внутри ячейки побитово устойчиво, но text_sha не совпал "
                          "с эталоном предыдущей ячейки" if a.allow_ref_mismatch else
                          "FAIL: text_sha не совпал с эталоном — патч изменил вычисление (или текст "
                          "не воспроизводится между перезапусками), бенчмарк не проводится")
    res["ref_mismatch"] = ref_mismatch_seen
    (out / "text-check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("вердикт:", res["verdict"])
    print("файл:", out / "text-check.json")
    sys.exit(0 if res["verdict"].startswith("PASS") else 3)


if __name__ == "__main__":
    main()
