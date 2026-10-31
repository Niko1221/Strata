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

# #3723 Context memory and the out-of-graphics-memory reason
add(en="INT8 is the standard: good quality and a medium amount of memory.",
    zh="INT8 是标准选项：质量好，内存占用适中。",
    ja="INT8 は標準の設定です。品質が良く、メモリ使用量は中程度です。",
    de="INT8 ist der Standard: gute Qualität und mittlerer Speicherbedarf.",
    es="INT8 es el estándar: buena calidad y un uso de memoria medio.",
    ru="INT8 — стандартный вариант: хорошее качество и средний расход памяти.",
    pt="INT8 é o padrão: boa qualidade e uso médio de memória.",
    ko="INT8은 표준 설정입니다. 품질이 좋고 메모리 사용량은 보통입니다.",
    fr="INT8 est le réglage standard : bonne qualité et une consommation de mémoire moyenne.")
add(en="Q4 uses about half the memory of INT8, but is less precise on very long documents.",
    zh="Q4 占用的内存约为 INT8 的一半，但在很长的文档上精度较低。",
    ja="Q4 は INT8 の約半分のメモリで済みますが、非常に長い文書では精度が少し下がります。",
    de="Q4 braucht etwa halb so viel Speicher wie INT8, ist aber bei sehr langen Dokumenten weniger genau.",
    es="Q4 usa aproximadamente la mitad de memoria que INT8, pero es menos preciso con documentos muy largos.",
    ru="Q4 занимает примерно вдвое меньше памяти, чем INT8, но менее точен на очень длинных документах.",
    pt="Q4 usa cerca de metade da memória do INT8, mas é menos preciso em documentos muito longos.",
    ko="Q4는 INT8의 약 절반 메모리를 쓰지만, 아주 긴 문서에서는 정밀도가 떨어집니다.",
    fr="Q4 utilise environ moitié moins de mémoire que INT8, mais il est moins précis sur de très longs documents.")
add(en="K8 / V4 uses about a quarter less memory than INT8 and is more precise than Q4. Prompts are 2 to 5% slower.",
    zh="K8 / V4 比 INT8 约少占用四分之一的内存，精度高于 Q4。处理提示词会慢 2% 到 5%。",
    ja="K8 / V4 は INT8 より約4分の1少ないメモリで済み、Q4 より精度が高い設定です。プロンプトの処理は 2〜5% 遅くなります。",
    de="K8 / V4 braucht etwa ein Viertel weniger Speicher als INT8 und ist genauer als Q4. Prompts sind 2 bis 5 % langsamer.",
    es="K8 / V4 usa aproximadamente una cuarta parte menos de memoria que INT8 y es más preciso que Q4. Los prompts son entre un 2 y un 5 % más lentos.",
    ru="K8 / V4 занимает примерно на четверть меньше памяти, чем INT8, и точнее, чем Q4. Запросы обрабатываются на 2–5 % медленнее.",
    pt="K8 / V4 usa cerca de um quarto a menos de memória que o INT8 e é mais preciso que o Q4. Os prompts ficam de 2 a 5 % mais lentos.",
    ko="K8 / V4는 INT8보다 메모리를 약 4분의 1 적게 쓰고 Q4보다 정밀합니다. 프롬프트 처리는 2~5% 느려집니다.",
    fr="K8 / V4 utilise environ un quart de mémoire en moins que INT8 et est plus précis que Q4. Les prompts sont 2 à 5 % plus lents.")
add(en="The model did not fit in your graphics card’s memory.",
    zh="模型放不进你的显卡显存。",
    ja="モデルがグラフィックスカードのメモリに収まりませんでした。",
    de="Das Modell hat nicht in den Speicher deiner Grafikkarte gepasst.",
    es="El modelo no cabía en la memoria de tu tarjeta gráfica.",
    ru="Модель не поместилась в память вашей видеокарты.",
    pt="O modelo não coube na memória da sua placa de vídeo.",
    ko="모델이 그래픽 카드 메모리에 들어가지 않았습니다.",
    fr="Le modèle ne tenait pas dans la mémoire de votre carte graphique.")
add(en="Lower the conversation context, or set Context memory to Q4 in Expert settings, then try again.",
    zh="请调低对话上下文，或在专家设置中把“上下文内存”设为 Q4，然后重试。",
    ja="会話のコンテキストを小さくするか、上級設定で「コンテキストメモリー」を Q4 にして、もう一度お試しください。",
    de="Verkleinere den Gesprächskontext oder stelle in den Experteneinstellungen „Kontextspeicher“ auf Q4 und versuche es erneut.",
    es="Reduce el contexto de la conversación o pon «Memoria de contexto» en Q4 en Ajustes expertos, y vuelve a intentarlo.",
    ru="Уменьшите контекст разговора или установите «Память контекста» в Q4 в разделе «Экспертные настройки» и повторите попытку.",
    pt="Reduza o contexto da conversa ou defina «Memória de contexto» como Q4 em Configurações avançadas e tente de novo.",
    ko="대화 컨텍스트를 줄이거나 전문가 설정에서 ‘컨텍스트 메모리’를 Q4로 바꾼 뒤 다시 시도하세요.",
    fr="Réduisez le contexte de la conversation, ou réglez « Mémoire de contexte » sur Q4 dans Réglages experts, puis réessayez.")

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
