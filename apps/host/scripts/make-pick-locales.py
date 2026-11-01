"""Writes src/locales/pick-<lang>.json: the strings of the model version list (VersionPicker), the code-block preview button and the This computer panel (branch f-0291-ui-fixes2).
Run from apps/host:  python scripts/make-pick-locales.py
Every entry names all nine languages; the keys are vp01.. in the order of S. pick-en.json is the English source. Placeholders ({0}, {1}) must be kept
in every language. A separate file keeps this work from colliding with the other catalogs (the same reason tune-, tools-, fix-, gh-, ui- and engine- exist)."""
import json
import re
from pathlib import Path

LANGS = ["en", "zh", "ja", "de", "es", "ru", "pt", "ko", "fr"]
S = []


def add(en, zh, ja, de, es, ru, pt, ko, fr):
    S.append(dict(en=en, zh=zh, ja=ja, de=de, es=es, ru=ru, pt=pt, ko=ko, fr=fr))


add("{0} download",
    "下载 {0}", "ダウンロード {0}", "{0} Download", "Descarga de {0}", "Загрузка {0}", "Download de {0}", "다운로드 {0}", "Téléchargement de {0}")
add("Needs {0} of memory",
    "需要 {0} 内存", "{0} のメモリーが必要", "Braucht {0} Speicher", "Necesita {0} de memoria", "Нужно {0} памяти", "Precisa de {0} de memória", "메모리 {0} 필요", "Nécessite {0} de mémoire")
add("Needs {0} of memory and about {1} GB of disk",
    "需要 {0} 内存和约 {1} GB 磁盘空间", "{0} のメモリーと約 {1} GB のディスク容量が必要", "Braucht {0} Speicher und etwa {1} GB Festplattenplatz", "Necesita {0} de memoria y unos {1} GB de disco",
    "Нужно {0} памяти и около {1} ГБ на диске", "Precisa de {0} de memória e cerca de {1} GB de disco", "메모리 {0}와 디스크 약 {1} GB 필요", "Nécessite {0} de mémoire et environ {1} Go de disque")
add("Slower on this computer",
    "在这台电脑上较慢", "このコンピューターでは遅くなります", "Auf diesem Computer langsamer", "Más lento en este equipo", "На этом компьютере медленнее", "Mais lento neste computador", "이 컴퓨터에서는 더 느립니다", "Plus lent sur cet ordinateur")
add("Show details",
    "显示详情", "詳細を表示", "Details anzeigen", "Mostrar detalles", "Показать подробности", "Mostrar detalhes", "자세히 보기", "Afficher les détails")
add("Hide details",
    "隐藏详情", "詳細を隠す", "Details ausblenden", "Ocultar detalles", "Скрыть подробности", "Ocultar detalhes", "자세히 숨기기", "Masquer les détails")
add("{0} RAM + VRAM",
    "{0} 内存 + 显存", "{0} の RAM + VRAM", "{0} RAM + VRAM", "{0} de RAM + VRAM", "{0} RAM + VRAM", "{0} de RAM + VRAM", "RAM + VRAM {0}", "{0} de RAM + VRAM")

add("Close preview",
    "关闭预览", "プレビューを閉じる", "Vorschau schließen", "Cerrar vista previa", "Закрыть предпросмотр", "Fechar pré-visualização", "미리보기 닫기", "Fermer l'aperçu")

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
        data[f"vp{i + 1:02d}"] = french(text) if lang == "fr" else text
    (here / f"pick-{lang}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("written", len(S), "strings in", len(LANGS), "languages")
