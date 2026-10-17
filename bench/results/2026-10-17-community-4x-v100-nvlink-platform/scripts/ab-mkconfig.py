#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Этап VIII: конфигурация ячейки A/B-стенда (изолированный вариант рабочего конфига).

Рабочий конфиг не меняется (требование п. 11 «не меняй существующий рабочий запуск»):
вариант пишется в отдельный файл. Отличия от рабочего — только те, что заданы явно:
--layer-split (значение заменяется, а не добавляется) и путь к бинарнику движка (для
«бинарника с патчем»).

usage: ab-mkconfig.py --out PATH [--name МЕТКА] [--split 12,25,37] [--engine PATH]
                     [--src PATH_рабочего_конфига]
"""
import argparse
import hashlib
import json
import pathlib

D = pathlib.Path.home() / "strata185"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="variant")
    ap.add_argument("--split", default="")
    ap.add_argument("--engine", default="")
    ap.add_argument("--src", default=str(D / "Strata" / "strata-185-iq3_s.json"))
    a = ap.parse_args()

    cfg = json.loads(pathlib.Path(a.src).read_text(encoding="utf-8-sig"))
    args = list(cfg.get("args") or [])
    if a.split:
        if "--layer-split" in args:
            i = args.index("--layer-split")
            args[i + 1] = a.split
        else:
            args += ["--layer-split", a.split]
    cfg["args"] = args
    if a.engine:
        cfg["exe"] = a.engine
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"создан {out} (name={a.name}, sha256={sha[:16]}…)")
    print(f"  layer-split: {args[args.index('--layer-split')+1] if '--layer-split' in args else 'нет'}")
    print(f"  exe: {cfg.get('exe')}")


if __name__ == "__main__":
    main()
