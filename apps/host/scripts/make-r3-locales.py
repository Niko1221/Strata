"""Writes src/locales/r3-<lang>.json: the strings added by the owner's GUI feedback round 3 (branch f-0291-ui-r3).
Run from apps/host:  python scripts/make-r3-locales.py
Every entry names all nine languages; the keys are r3_01.. in the order of S. r3-en.json is the English source. Placeholders ({0}, {1}) must be kept in every
language. A separate file keeps this work from colliding with the other catalogs (the same reason pick-, upd-, df2- and the others exist)."""
import json
import re
from pathlib import Path

LANGS = ["en", "zh", "ja", "de", "es", "ru", "pt", "ko", "fr"]
S = []


def add(en, zh, ja, de, es, ru, pt, ko, fr):
    S.append(dict(en=en, zh=zh, ja=ja, de=de, es=es, ru=ru, pt=pt, ko=ko, fr=fr))


# 2: the Read pictures row while the installed engine has no picture reader (a tooltip, not a paragraph)
add("Needs an engine update",
    "需要更新引擎", "エンジンの更新が必要です", "Braucht ein Engine-Update", "Necesita actualizar el motor", "Нужно обновить движок", "Precisa de uma atualização do motor", "엔진 업데이트가 필요합니다", "Nécessite une mise à jour du moteur")

#@@MORE@@

here = Path(__file__).resolve().parent.parent / "src" / "locales"
NBSP = " "


def french(text):
    text = re.sub(r"(?<=[A-Za-zÀ-ſ])'(?=[A-Za-zÀ-ſ{])", "’", text)
    text = re.sub(r"(?<=\S) ([:;?!])", NBSP + r"\1", text)
    text = re.sub(r"(?<=[\d}]) %", NBSP + "%", text)
    return text


for lang in LANGS:
    data = {}
    for i, entry in enumerate(S):
        text = entry[lang]
        assert sorted(re.findall(r"\{\d+\}", entry["en"])) == sorted(re.findall(r"\{\d+\}", text)), (lang, entry["en"], text)
        data[f"r3_{i + 1:02d}"] = french(text) if lang == "fr" else text
    (here / f"r3-{lang}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("written", len(S), "strings in", len(LANGS), "languages")
