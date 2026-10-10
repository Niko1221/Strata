#!/usr/bin/env python3
"""A/B stand for Strata 0.1.41 against 0.1.42 on one Tesla V100 host (Linux, sm_70).

One phase per engine version, identical prompts and order:
  * a chars/token calibration on 20,000 characters of filler;
  * a 128K-token Russian haystack needle at depths 10/50/90 % (greedy, 256-token cap);
  * a Russian story prompt (450-token cap);
  * four short prompts (ru-chat, code, math, tool-json; 400-token cap).

Usage:  python3 ab.py --phase post-0142 --out post-0142.json [--only calib,needle,ru,short]
        STRATA_ENDPOINT=http://127.0.0.1:8080/v1/chat/completions python3 ab.py ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.request

ENDPOINT = os.environ.get("STRATA_ENDPOINT", "http://127.0.0.1:8080/v1/chat/completions")
MODEL = "qwen3.8-flash-next-iq3_s-v100"

CODE = "ХЯ-6043-ВЕКТОР"
NEEDLE = (
    "СЛУЖЕБНАЯ ЗАПИСЬ 24-БИС. Контрольный код доступа к терминалу узла: {code}. "
    "Код подтверждён дежурным инженером и внесён в журнал смены."
).format(code=CODE)
QUESTION = (
    "В документе выше есть служебная запись с контрольным кодом доступа к терминалу. "
    "Назови этот код. Ответь только кодом, без пояснений."
)

NODES = ["РУ-12", "АСУ-7", "ТП-3", "ЩСУ-9", "КРУ-4", "НПС-2", "СЦБ-11", "ВЛ-6"]
EVENTS = [
    "кратковременное повышение давления в контуре охлаждения",
    "срабатывание защиты по перегрузке привода",
    "расхождение показаний двух датчиков уровня",
    "останов насоса по сигналу вибрации",
    "просадка напряжения на секции собственных нужд",
    "замыкание в цепи управления задвижкой",
    "рост температуры подшипника выше уставки",
    "потеря связи с контроллером нижнего уровня",
]
ACTIONS = [
    "перевод узла на резервный контур",
    "вызов ремонтной бригады и оформление наряда",
    "повторный пуск после осмотра и продувки",
    "переключение на ручное управление до выяснения причин",
    "проверку изоляции и протяжку клеммных соединений",
    "замену датчика из состава обменного фонда",
]


def make_filler(chars: int) -> str:
    """Deterministic long Russian text (a shift log), seeded so every phase builds the same one."""
    rnd = random.Random(42)
    parts: list[str] = []
    n = 0
    while sum(len(p) for p in parts) < chars:
        n += 1
        hh, mm = divmod(rnd.randrange(0, 1440), 60)
        parts.append(
            "Запись {n}. В {hh:02d}:{mm:02d} на узле {node} зафиксировано {event}, "
            "класс {cls}; дежурная смена выполнила {action}. Отметка в журнале "
            "сохраняется в архиве смены и сверяется при передаче вахты.\n".format(
                n=n, hh=hh, mm=mm,
                node=rnd.choice(NODES), event=rnd.choice(EVENTS),
                cls=rnd.choice(["А", "Б", "В", "предупредительный", "аварийный"]),
                action=rnd.choice(ACTIONS),
            )
        )
    return "".join(parts)[:chars]


SHORT_PROMPTS = [
    ("ru-chat", "Объясни в трёх предложениях, зачем в nginx нужен `proxy_read_timeout` и что будет при слишком малом значении."),
    ("code", "Напиши на Python функцию `parse_size(s: str) -> int`, которая переводит строки вида '12GiB', '350M', '2T' в байты. Только код и одна строка пояснения."),
    ("math", "Реши по шагам: поезд прошёл 240 км за 2 ч 40 мин, затем 180 км за 1 ч 50 мин. Какова средняя скорость на всём пути в км/ч? Ответ с точностью до 0.1."),
    ("tool-json", "Верни строго JSON без пояснений: объект с полями host=\"127.0.0.1\", port=8080, engine=\"strata\", tags=[\"llm\",\"v100\"]."),
]


def chat(messages, max_tokens, timeout=None):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    wall = time.time() - t0
    ch = payload["choices"][0]
    msg = ch.get("message") or {}
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "finish_reason": ch.get("finish_reason"),
        "usage": payload.get("usage", {}),
        "timings": payload.get("timings", {}),
        "wall_s": round(wall, 2),
    }


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-tokens", type=int, default=131072)
    ap.add_argument("--only", default="calib,needle,ru,short")
    args = ap.parse_args()
    only = set(args.only.split(","))

    results: dict = {"phase": args.phase, "endpoint": ENDPOINT,
                     "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "target_tokens": args.target_tokens, "results": {}}

    def save():
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=1)

    # 1. Calibration: chars per token on 20,000 characters of the filler (a cheap prompt).
    ratio = 2.6
    if "calib" in only:
        sample = make_filler(20000)
        log("calib: 20,000 characters -> request")
        r = chat([{"role": "user", "content": sample + "\n\nСколько записей в журнале? Ответь числом."}], 8, timeout=1800)
        pt = r["usage"].get("prompt_tokens") or 0
        if pt:
            ratio = 20000 / pt
        results["calib"] = {"chars": 20000, "prompt_tokens": pt, "chars_per_token": round(ratio, 3),
                            "prompt_per_second": r["timings"].get("prompt_per_second")}
        log("calib: prompt_tokens=%s chars/token=%.3f" % (pt, ratio))
        save()

    # 2. Needle at the target length, depths 10/50/90 %.
    if "needle" in only:
        filler = make_filler(int(args.target_tokens * ratio))
        results["results"]["needle"] = {}
        for depth in (10, 50, 90):
            pos = int(len(filler) * depth / 100)
            doc = filler[:pos] + "\n" + NEEDLE + "\n" + filler[pos:]
            log("needle %d%%: ~%d characters -> request" % (depth, len(doc)))
            try:
                r = chat([{"role": "user", "content": doc + "\n\n" + QUESTION}], 256, timeout=7200)
            except Exception as exc:  # noqa: BLE001
                log("needle %d%%: ERROR %s" % (depth, exc))
                results["results"]["needle"][str(depth)] = {"error": repr(exc)}
                save()
                continue
            hay = (r["content"] + " " + r["reasoning"]).replace(" ", "")
            rec = {
                "depth_pct": depth, "chars": len(doc), "prompt_tokens": r["usage"].get("prompt_tokens"),
                "cached_tokens": (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"),
                "found": CODE.replace(" ", "") in hay,
                "answer": r["content"][:200], "finish_reason": r["finish_reason"],
                "timings": r["timings"], "wall_s": r["wall_s"],
            }
            results["results"]["needle"][str(depth)] = rec
            log("needle %d%%: found=%s prompt_tokens=%s prefill=%.1f tok/s decode=%.1f tok/s (cache_n=%s)"
                % (depth, rec["found"], rec["prompt_tokens"],
                   r["timings"].get("prompt_per_second") or 0, r["timings"].get("predicted_per_second") or 0,
                   r["timings"].get("cache_n")))
            save()

    # 3. Russian story: a long answer to a short prompt.
    if "ru" in only:
        prompt = ("Напиши подробный рассказ (не менее 350 слов) о ночной смене диспетчера "
                  "на железнодорожной станции: только проза, без списков и заголовков.")
        log("ru-decode: request")
        try:
            r = chat([{"role": "user", "content": prompt}], 450, timeout=3600)
            results["results"]["ru_decode"] = {
                "completion_tokens": r["usage"].get("completion_tokens"),
                "timings": r["timings"], "wall_s": r["wall_s"],
                "sha256_text": hashlib.sha256(r["content"].encode()).hexdigest()[:16],
                "text_head": r["content"][:400],
            }
            log("ru-decode: %.1f tok/s, draft %s/%s, wall %.1f s"
                % (r["timings"].get("predicted_per_second") or 0, r["timings"].get("draft_n_accepted"),
                   r["timings"].get("draft_n"), r["wall_s"]))
        except Exception as exc:  # noqa: BLE001
            log("ru-decode: ERROR %s" % exc)
            results["results"]["ru_decode"] = {"error": repr(exc)}
        save()

    # 4. Short prompts: answer identity and decode speed on ordinary requests.
    if "short" in only:
        results["results"]["short"] = {}
        for name, prompt in SHORT_PROMPTS:
            log("short %s: request" % name)
            try:
                r = chat([{"role": "user", "content": prompt}], 400, timeout=1800)
                results["results"]["short"][name] = {
                    "prompt": prompt,
                    "prompt_tokens": r["usage"].get("prompt_tokens"),
                    "completion_tokens": r["usage"].get("completion_tokens"),
                    "timings": r["timings"], "wall_s": r["wall_s"],
                    "content": r["content"], "reasoning": r["reasoning"][:1500],
                }
                log("short %s: %.1f tok/s decode, %.1f tok/s prefill"
                    % (name, r["timings"].get("predicted_per_second") or 0,
                       r["timings"].get("prompt_per_second") or 0))
            except Exception as exc:  # noqa: BLE001
                log("short %s: ERROR %s" % (name, exc))
                results["results"]["short"][name] = {"error": repr(exc)}
            save()

    results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    log("done: %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
