"""Writes src/locales/ui-<lang>.json: the interface strings of the 0.2.9.1 interface changes (branch f-0291-ui).
Run from apps/host:  python scripts/make-ui-locales.py
Every entry names all nine languages; the keys are ui01.. in the order of S. ui-en.json is the English source. Placeholders ({0}, {1}) must be
kept in every language. A separate file keeps this work from colliding with the other catalogs (the same reason tune-, tools-, fix- and gh- exist)."""
import json
import re
from pathlib import Path

LANGS = ["en", "zh", "ja", "de", "es", "ru", "pt", "ko", "fr"]
S = []


def add(**kw):
    assert sorted(kw) == sorted(LANGS), sorted(set(LANGS) ^ set(kw))
    S.append(kw)


# #4725 the approval stays in view
add(en="Waiting for your answer",
    zh="等待你的回答", ja="あなたの返答を待っています", de="Wartet auf deine Antwort", es="Esperando tu respuesta",
    ru="Ждёт вашего ответа", pt="Aguardando a sua resposta", ko="답변을 기다리는 중", fr="En attente de votre réponse")
add(en="Waiting for your answer below",
    zh="等待你在下方回答", ja="下で返答をお待ちしています", de="Wartet unten auf deine Antwort", es="Esperando tu respuesta abajo",
    ru="Ждёт вашего ответа ниже", pt="Aguardando a sua resposta abaixo", ko="아래에서 답변을 기다리는 중", fr="En attente de votre réponse ci-dessous")

here = Path(__file__).resolve().parent.parent / "src" / "locales"
NBSP = " "


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
        data[f"ui{i + 1:02d}"] = french(text) if lang == "fr" else text
    (here / f"ui-{lang}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("written", len(S), "strings in", len(LANGS), "languages")
