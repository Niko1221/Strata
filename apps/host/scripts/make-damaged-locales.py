"""Writes src/locales/dmg-<lang>.json: the surface for chat files the host cannot read (#5421).
Run from apps/host:  python scripts/make-damaged-locales.py
Every entry names all nine languages; the keys are dmg01.. in the order of S. dmg-en.json is the English source. Placeholders ({0}) must be kept in every
language. A separate file keeps this work from colliding with the other catalogs (the same reason sav-, upd-, df2-, pick- and the others exist)."""
import json
import re
from pathlib import Path

LANGS = ["en", "zh", "ja", "de", "es", "ru", "pt", "ko", "fr"]
S = []


def add(en, zh, ja, de, es, ru, pt, ko, fr):
    S.append(dict(en=en, zh=zh, ja=ja, de=de, es=es, ru=ru, pt=pt, ko=ko, fr=fr))


add("Unreadable chats",
    "无法读取的聊天",
    "読み込めないチャット",
    "Unlesbare Chats",
    "Chats ilegibles",
    "Нечитаемые чаты",
    "Chats ilegíveis",
    "읽을 수 없는 채팅",
    "Conversations illisibles")
add("Strata could not read these chat files. Delete one to remove the file.",
    "Strata 无法读取这些聊天文件。删除其中一个即可移除该文件。",
    "Strata はこれらのチャットファイルを読み込めませんでした。削除するとファイルが消えます。",
    "Strata konnte diese Chat-Dateien nicht lesen. Löschen Sie eine, um die Datei zu entfernen.",
    "Strata no pudo leer estos archivos de chat. Elimina uno para quitar el archivo.",
    "Strata не смог прочитать эти файлы чатов. Удалите один, чтобы убрать файл.",
    "O Strata não conseguiu ler estes arquivos de chat. Exclua um para remover o arquivo.",
    "Strata가 이 채팅 파일들을 읽을 수 없습니다. 하나를 삭제하면 파일이 제거됩니다.",
    "Strata n'a pas pu lire ces fichiers de conversation. Supprimez-en un pour retirer le fichier.")
add("Strata could not read this chat file, so its messages are not shown.",
    "Strata 无法读取此聊天文件，因此不显示其消息。",
    "Strata はこのチャットファイルを読み込めなかったため、メッセージは表示されません。",
    "Strata konnte diese Chat-Datei nicht lesen, daher werden ihre Nachrichten nicht angezeigt.",
    "Strata no pudo leer este archivo de chat, así que no se muestran sus mensajes.",
    "Strata не смог прочитать этот файл чата, поэтому его сообщения не показаны.",
    "O Strata não conseguiu ler este arquivo de chat, então as mensagens não são exibidas.",
    "Strata가 이 채팅 파일을 읽을 수 없어서 메시지가 표시되지 않습니다.",
    "Strata n'a pas pu lire ce fichier de conversation ; ses messages ne sont donc pas affichés.")
add("Delete chat file",
    "删除聊天文件",
    "チャットファイルを削除",
    "Chat-Datei löschen",
    "Eliminar archivo de chat",
    "Удалить файл чата",
    "Excluir arquivo de chat",
    "채팅 파일 삭제",
    "Supprimer le fichier de conversation")
add("Delete {0}.json",
    "删除 {0}.json",
    "{0}.json を削除",
    "{0}.json löschen",
    "Eliminar {0}.json",
    "Удалить {0}.json",
    "Excluir {0}.json",
    "{0}.json 삭제",
    "Supprimer {0}.json")
add("Delete this chat file?",
    "删除此聊天文件？",
    "このチャットファイルを削除しますか？",
    "Diese Chat-Datei löschen?",
    "¿Eliminar este archivo de chat?",
    "Удалить этот файл чата?",
    "Excluir este arquivo de chat?",
    "이 채팅 파일을 삭제할까요?",
    "Supprimer ce fichier de conversation ?")
add("Keep file",
    "保留文件",
    "ファイルを保持",
    "Datei behalten",
    "Conservar archivo",
    "Оставить файл",
    "Manter arquivo",
    "파일 유지",
    "Garder le fichier")
add("Delete file",
    "删除文件",
    "ファイルを削除",
    "Datei löschen",
    "Eliminar archivo",
    "Удалить файл",
    "Excluir arquivo",
    "파일 삭제",
    "Supprimer le fichier")
add("The file {0} is deleted. Nothing else changes, and the chat is not coming back.",
    "文件 {0} 已删除。其他内容不变，此聊天无法恢复。",
    "ファイル {0} が削除されました。他は変更されず、チャットは戻りません。",
    "Die Datei {0} wird gelöscht. Sonst ändert sich nichts, und der Chat kommt nicht zurück.",
    "El archivo {0} se elimina. Nada más cambia y el chat no volverá.",
    "Файл {0} будет удалён. Ничего больше не изменится, и чат не вернётся.",
    "O arquivo {0} é excluído. Nada mais muda, e o chat não volta.",
    "파일 {0}이(가) 삭제됩니다. 다른 것은 변하지 않고 채팅은 돌아오지 않습니다.",
    "Le fichier {0} est supprimé. Rien d'autre ne change et la conversation ne reviendra pas.")
add("The chat file could not be deleted",
    "聊天文件无法删除",
    "チャットファイルを削除できませんでした",
    "Die Chat-Datei konnte nicht gelöscht werden",
    "No se pudo eliminar el archivo de chat",
    "Не удалось удалить файл чата",
    "O arquivo de chat não pôde ser excluído",
    "채팅 파일을 삭제할 수 없습니다",
    "Le fichier de conversation n'a pas pu être supprimé")

here = Path(__file__).resolve().parent.parent / "src" / "locales"
NBSP = " "


def french(text):
    text = re.sub(r"(?<=[A-Za-zÀ-ſ])'(?=[A-Za-zÀ-ſ{])", "’", text)
    text = re.sub(r"(?<=\S) ([:;?!])", NBSP + r"\1", text)
    return text


for lang in LANGS:
    data = {}
    for i, entry in enumerate(S):
        text = entry[lang]
        assert sorted(re.findall(r"\{\d+\}", entry["en"])) == sorted(re.findall(r"\{\d+\}", text)), (lang, entry["en"], text)
        data[f"dmg{i + 1:02d}"] = french(text) if lang == "fr" else text
    with (here / f"dmg-{lang}.json").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
print("written", len(S), "strings in", len(LANGS), "languages")
