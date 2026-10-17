#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Этап VIII: проверка возврата сервиса по ТРЁМ признакам (правило R6) + ожидание подъёма.

Одного /health недостаточно: в двух инцидентах /health отвечала обёртка без живого движка.
Признаки: (1) процессы движка живы, (2) модель загружена в VRAM (порог на каждой карте И
суммарный порог по всем картам), (3) реальный /v1/chat/completions вернул choices.

Уточнение порога VRAM (редакция 2, 2026-10-17T06:16Z). Прежний порог 14000 МиБ на карту был
рассчитан на --layer-split auto и НЕ проходит для закреплённого разбиения 12,25,37: на нём
CUDA0 загружает только слои 12-24 и легальный измеренный уровень — 13744 МиБ (Этап IV,
baseline-profile.md §8.2: «VRAM GPU0 ниже (13.7 GiB вместо 15.1)»). Из-за этого ячейка B0
прогона 2026-10-17T054922Z была ошибочно помечена FAILED при полностью работоспособном
сервисе (health 200, chat отвечает, hit rate 100 %). Теперь признак (2) = «суммарно по картам
не ниже --vram-total-mib И на каждой карте не ниже --vram-mib». Пороги выбраны так, чтобы
отличать «модель загружена» от «обёртка без движка» (там 7 МиБ на карту): минимум по картам
13000 МиБ и сумма 57000 МиБ — ниже обоих известных рабочих состояний (auto: 15.1-16.0 GiB на
карту, ~62 GiB сумма; 12,25,37: 13.7/15.2/14.9/16.0, ~59.9 GiB) и выше любого состояния без
модели.

usage: ab-ready.py [--timeout 900] [--vram-mib 13000] [--vram-total-mib 57000] [--gpus 0,1,2,3] [--quiet]
Выход 0 — сервис вернулся; 1 — нет (причины в stdout).
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.request

D = pathlib.Path.home() / "strata185"
CFG = D / "Strata" / "strata-185-iq3_s.json"
PAT = r"strata185/Strata/(.venv/bin/python|engine-cuda12/)"


def api_key():
    return json.loads(CFG.read_text(encoding="utf-8-sig")).get("api_key", "")


def health():
    try:
        req = urllib.request.Request("http://127.0.0.1:8080/health",
                                     headers={"Authorization": "Bearer " + api_key()})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode()[:400]
    except Exception as e:
        return 0, str(e)


def chat():
    body = json.dumps({"model": "qwen3.8-flash-next-iq3_s",
                       "messages": [{"role": "user", "content": "Ответь ровно одно слово: готово"}],
                       "max_tokens": 16, "temperature": 0, "reasoning_effort": "none"}).encode()
    try:
        req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=body,
                                     headers={"Authorization": "Bearer " + api_key(),
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            j = json.loads(r.read().decode())
        ch = (j.get("choices") or [{}])[0]
        return bool(j.get("choices")), (ch.get("message") or {}).get("content", "")[:40]
    except Exception as e:
        return False, str(e)


def procs():
    try:
        out = subprocess.run(["pgrep", "-f", PAT], capture_output=True, text=True).stdout.split()
    except Exception:
        out = []
    engine = []
    try:
        out2 = subprocess.run(["pgrep", "-f", "engine-cuda12/strata"], capture_output=True, text=True).stdout.split()
        engine = out2
    except Exception:
        pass
    return out, engine


def vram(gpus):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
    except Exception:
        return {}
    res = {}
    for ln in out.splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) >= 2 and p[0].isdigit():
            res[int(p[0])] = int(p[1])
    return {g: res.get(g, -1) for g in gpus}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--vram-mib", type=int, default=13000,
                    help="минимум МиБ на КАЖДУЮ карту (порог «модель загружена», не «auto-раскладка»)")
    ap.add_argument("--vram-total-mib", type=int, default=57000,
                    help="минимум суммарных МиБ по всем перечисленным картам")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    gpus = [int(x) for x in a.gpus.split(",")]
    t0 = time.time()
    last = {}
    while time.time() - t0 < a.timeout:
        st, hb = health()
        pro, eng = procs()
        vm = vram(gpus)
        ok_procs = len(eng) > 0
        ok_vram = all(v >= a.vram_mib for v in vm.values()) and sum(vm.values()) >= a.vram_total_mib
        ok_chat, sample = (False, "") if st != 200 else chat()
        last = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "health": st,
                "health_body": hb[:200], "wrapper_pids": pro, "engine_pids": eng,
                "vram_mib": vm, "ok_procs": ok_procs, "ok_vram": ok_vram,
                "vram_total_mib": sum(vm.values()), "vram_gate": {"per_card_mib": a.vram_mib,
                                                                  "total_mib": a.vram_total_mib},
                "ok_chat": ok_chat, "chat_sample": sample,
                "elapsed_s": round(time.time() - t0, 1)}
        if ok_procs and ok_vram and ok_chat:
            if not a.quiet:
                print(json.dumps(last, ensure_ascii=False, indent=1))
            print("READY через %.0f с (три признака: процессы живы, VRAM %s сумма %d, chat вернул choices)" %
                  (time.time() - t0, vm, sum(vm.values())))
            return 0
        time.sleep(10)
    print("НЕ ГОТОВ через %.0f с:" % (time.time() - t0))
    print(json.dumps(last, ensure_ascii=False, indent=1))
    return 1


if __name__ == "__main__":
    sys.exit(main())
