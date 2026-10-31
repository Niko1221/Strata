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

# #4804 why Start tuning is greyed out
add(en="Strata is still checking this model for saved tunings. Start tuning works as soon as that is done.",
    zh="Strata 仍在检查此模型是否有已保存的调优。检查完成后即可开始调优。",
    ja="Strata がこのモデルの保存済みチューニングを確認しています。確認が終わりしだい、チューニングを開始できます。",
    de="Strata prüft dieses Modell noch auf gespeicherte Abstimmungen. Sobald das fertig ist, kannst du die Abstimmung starten.",
    es="Strata todavía está comprobando si este modelo tiene ajustes guardados. Podrás empezar el ajuste en cuanto termine.",
    ru="Strata ещё проверяет, есть ли у этой модели сохранённые настройки. Как только проверка закончится, можно будет запустить настройку.",
    pt="O Strata ainda está verificando se este modelo tem ajustes salvos. Assim que terminar, você poderá iniciar o ajuste.",
    ko="Strata가 이 모델의 저장된 최적화를 확인하는 중입니다. 확인이 끝나면 바로 최적화를 시작할 수 있습니다.",
    fr="Strata vérifie encore si ce modèle a des optimisations enregistrées. Dès que c’est fini, vous pourrez lancer l’optimisation.")
add(en="Strata could not read the tuning details for this model, so tuning cannot start. Close this window and open it again.",
    zh="Strata 无法读取此模型的调优信息，因此无法开始调优。请关闭此窗口后重新打开。",
    ja="Strata がこのモデルのチューニング情報を読み取れなかったため、チューニングを開始できません。このウィンドウを閉じて、もう一度開いてください。",
    de="Strata konnte die Abstimmungsdaten dieses Modells nicht lesen, deshalb lässt sich die Abstimmung nicht starten. Schließe dieses Fenster und öffne es erneut.",
    es="Strata no pudo leer los datos de ajuste de este modelo, así que el ajuste no puede empezar. Cierra esta ventana y ábrela de nuevo.",
    ru="Strata не смогла прочитать данные настройки этой модели, поэтому настройку нельзя запустить. Закройте это окно и откройте его снова.",
    pt="O Strata não conseguiu ler os dados de ajuste deste modelo, então o ajuste não pode começar. Feche esta janela e abra-a novamente.",
    ko="Strata가 이 모델의 최적화 정보를 읽지 못해 최적화를 시작할 수 없습니다. 이 창을 닫았다가 다시 여세요.",
    fr="Strata n’a pas pu lire les données d’optimisation de ce modèle, l’optimisation ne peut donc pas démarrer. Fermez cette fenêtre et rouvrez-la.")
add(en="Checking for saved tunings\u2026",
    zh="正在检查已保存的调优\u2026", ja="保存済みチューニングを確認しています\u2026", de="Gespeicherte Abstimmungen werden geprüft\u2026", es="Comprobando los ajustes guardados\u2026",
    ru="Проверяем сохранённые настройки\u2026", pt="Verificando os ajustes salvos\u2026", ko="저장된 최적화를 확인하는 중\u2026", fr="Vérification des optimisations enregistrées\u2026")

# #4802 the progress card
add(en="Still working for {0}. You don't need to do anything.",
    zh="仍在处理，已用时 {0}。你无需做任何操作。",
    ja="処理中です（{0}経過）。操作は必要ありません。",
    de="Läuft noch seit {0}. Du musst nichts tun.",
    es="Sigue trabajando desde hace {0}. No tienes que hacer nada.",
    ru="Всё ещё работает: {0}. От вас ничего не требуется.",
    pt="Ainda trabalhando há {0}. Você não precisa fazer nada.",
    ko="{0}째 작업 중입니다. 따로 할 일은 없습니다.",
    fr="Toujours en cours depuis {0}. Vous n'avez rien à faire.")
add(en="The buttons below wait until this finishes.",
    zh="下方的按钮会在此操作完成后恢复可用。",
    ja="この処理が終わるまで、下のボタンは使えません。",
    de="Die Schaltflächen unten warten, bis das fertig ist.",
    es="Los botones de abajo esperan a que esto termine.",
    ru="Кнопки ниже подождут, пока это не закончится.",
    pt="Os botões abaixo esperam até isto terminar.",
    ko="아래 버튼은 이 작업이 끝날 때까지 기다립니다.",
    fr="Les boutons ci-dessous attendent la fin de cette opération.")

# #4798 two addresses
add(en="API base URL on this computer",
    zh="此电脑上的 API 基础地址", ja="このパソコンでの API ベース URL", de="API-Basis-URL auf diesem Computer", es="URL base de la API en este equipo",
    ru="Базовый URL API на этом компьютере", pt="URL base da API neste computador", ko="이 컴퓨터의 API 기본 URL", fr="URL de base de l'API sur cet ordinateur")
add(en="Address for other devices",
    zh="其他设备使用的地址", ja="他のデバイス用のアドレス", de="Adresse für andere Geräte", es="Dirección para otros dispositivos",
    ru="Адрес для других устройств", pt="Endereço para outros dispositivos", ko="다른 기기용 주소", fr="Adresse pour les autres appareils")
add(en="Other devices use this address. It has its own port, separate from the one Strata's window uses, because only the model's API is shared on the network, not the app.",
    zh="其他设备使用此地址。它有自己的端口，与 Strata 窗口使用的端口分开，因为网络上只共享模型的 API，而不是应用本身。",
    ja="他のデバイスはこのアドレスを使います。ネットワークで共有されるのはモデルの API だけでアプリ本体ではないため、Strata のウィンドウが使うポートとは別のポートを使います。",
    de="Andere Geräte verwenden diese Adresse. Sie hat einen eigenen Port, getrennt von dem des Strata-Fensters, weil im Netzwerk nur die API des Modells freigegeben wird, nicht die App.",
    es="Los demás dispositivos usan esta dirección. Tiene su propio puerto, distinto del que usa la ventana de Strata, porque en la red solo se comparte la API del modelo, no la aplicación.",
    ru="Другие устройства используют этот адрес. У него свой порт, отдельный от порта окна Strata, потому что в сети доступен только API модели, а не само приложение.",
    pt="Os outros dispositivos usam este endereço. Ele tem a própria porta, separada da que a janela do Strata usa, porque só a API do modelo é compartilhada na rede, não o aplicativo.",
    ko="다른 기기는 이 주소를 사용합니다. 네트워크에는 앱이 아니라 모델의 API만 공유되므로, Strata 창이 쓰는 포트와는 별도의 포트를 씁니다.",
    fr="Les autres appareils utilisent cette adresse. Elle a son propre port, distinct de celui de la fenêtre de Strata, car seule l'API du modèle est partagée sur le réseau, pas l'application.")

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
