/** Complete the existing import/search UI entries; preserve catalog IDs and English key order. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
const directory = resolve(import.meta.dirname, '../../app/host/src/locales');
const english = JSON.parse(readFileSync(resolve(directory, 'en.json'), 'utf8'));
const ids = [...Array.from({length:28},(_,i)=>`uxweb${String(i+1).padStart(3,'0')}`),
  ...[1,2,3,4,5,6,7,...Array.from({length:18},(_,i)=>i+9)].map(i=>`uxskill${String(i).padStart(3,'0')}`),
  'uxtool001','uxtool002','uxweb029'];
const translations = {
de: `Web durchsuchen
Nur Ihre Suchanfrage wird an DuckDuckGo / Bing gesendet. Kein Konto oder API-Schlüssel erforderlich. Ihr Gespräch und Ihr privater Speicher werden nicht für die Suche verwendet.
Suchanfrage
Ziel
Ergebnisse für die Frage verwenden
Werkzeugeinstellungen öffnen
Suche läuft…
Nur Auszüge. Die vollständigen Seiten wurden nicht gelesen. Die Quellen gelten für die nächste Antwort mit ausgeschalteten autonomen Werkzeugen.
Keine Ergebnisse. Versuchen Sie eine andere Suchanfrage.
Suchquellen bereit · Werkzeuge für diese Antwort aus
Quellen entfernen
Suchquellen · nur Auszüge
Websuche öffnen, eine Anfrage wählen und Auszüge anhängen
Die Einstellungen der Websuche konnten nicht geladen werden.
Die Websuche ist aus. Aktivieren Sie sie unter Einstellungen → Werkzeuge.
Die manuelle Suche unterstützt derzeit nur DuckDuckGo / Bing. Wählen Sie dies unter Einstellungen → Werkzeuge.
Ein gültiger Chat ist erforderlich.
Bestätigen Sie die Suche vor dem Senden.
Geben Sie eine Suchanfrage mit 1–400 Zeichen ein.
Eine Suche läuft bereits. Stoppen Sie sie oder warten Sie.
Die Suche wurde abgebrochen.
Die Websuche ist derzeit eingeschränkt. Versuchen Sie es später erneut.
Der Suchanbieter war nicht erreichbar. Prüfen Sie die Verbindung und versuchen Sie es erneut.
Die Suchquellen sind abgelaufen oder gehören zu einem anderen Chat. Suchen Sie erneut.
Öffnen Sie die Websuche in der lokalen Strata-App.
Suchquellen benötigen eine normale Frage.
Autonome Recherche benötigt ein Agentenmodell. Verwenden Sie mit diesem Modell die Websuche zum Anhängen von Quellen.
Suchquellen verwenden eine einfache Antwort ohne Werkzeuge.
SKILL.md importieren
Aus Ordner
Importiert nur Anweisungen. Begleitende Skripte, Quellen und Abhängigkeiten werden weder installiert noch ausgeführt. Prüfen Sie die Anweisungen vor dem Import.
SKILL.md wählen
Oder SKILL.md einfügen
Skill-Vorschau
Import läuft…
Bis zu 64 KiB. Name und Beschreibung im Frontmatter sowie höchstens 40000 Anweisungszeichen sind erforderlich.
Trennen Sie Ausdrücke mit Kommas. Im normalen Chat werden aktive Skills angewendet, wenn Ihre Frage einen Auslöser enthält.
Beispielfrage: {0}
bitte helfen Sie bei dieser Aufgabe.
Andere Anweisungen wählen
Aus Ordner fügt einen lokalen Skill-Ordner hinzu. Die Agentenkompatibilität umfasst Anweisungen; Begleitdateien benötigen möglicherweise eine separate Einrichtung.
Im normalen Chat werden aktive Skills bei passenden Auslösern in Ihrer Frage angewendet. Prüfen Sie vorher den Arbeitsbereich.
Wählen Sie eine SKILL.md-Datei mit höchstens 64 KiB.
Die Skill-Datei konnte nicht gelesen werden.
Die Skill-Vorschau konnte nicht geöffnet werden.
Der Skill konnte nicht importiert werden.
SKILL.md benötigt einen Namen mit höchstens 80 Zeichen.
SKILL.md benötigt eine einzeilige Beschreibung mit höchstens 400 Zeichen.
Skill-Anweisungen müssen 1–40000 Zeichen enthalten.
Ein Skill mit diesem Namen existiert bereits. Benennen Sie ihn vor dem Import um.
Öffnen Sie den Skill-Import in der lokalen Strata-App.
Öffnen Sie zuerst die Vorschau und bestätigen Sie den Import.
Fügen Sie vor dem Import mindestens einen Auslöser hinzu.
DuckDuckGo / Bing funktioniert ohne API-Schlüssel. Wählen Sie über die Websuche im Chat eine Anfrage und hängen Sie Auszüge an.
Kein Schlüssel gespeichert. Die autonome Agentensuche verwendet ersatzweise DuckDuckGo / Bing. Die manuelle Websuche unterstützt nur den integrierten Anbieter.
Suchen`,
es: `Buscar en la web
Solo la consulta que escriba se envía a DuckDuckGo / Bing. No necesita cuenta ni clave API. Su conversación y memoria privada no se usan para crear la consulta.
Consulta de búsqueda
Destino
Usar resultados en la pregunta
Abrir ajustes de herramientas
Buscando…
Solo fragmentos. No se leyeron las páginas completas. Las referencias se aplican a la siguiente respuesta con las herramientas autónomas desactivadas.
Sin resultados. Pruebe otra consulta.
Referencias listas · herramientas desactivadas para esta respuesta
Quitar referencias
Referencias de búsqueda · solo fragmentos
Abrir la búsqueda web para elegir una consulta y adjuntar fragmentos
No se pudieron cargar los ajustes de búsqueda web.
La búsqueda web está desactivada. Actívela en Ajustes → Herramientas.
La búsqueda manual solo admite DuckDuckGo / Bing por ahora. Elíjalo en Ajustes → Herramientas.
Se necesita un chat válido.
Confirme la búsqueda antes de enviarla.
Escriba una consulta de 1–400 caracteres.
Ya hay una búsqueda en curso. Deténgala o espere.
La búsqueda se canceló.
La búsqueda web está limitada en este momento. Inténtelo más tarde.
No se pudo contactar con el proveedor de búsqueda. Compruebe la conexión e inténtelo de nuevo.
Las referencias caducaron o pertenecen a otro chat. Busque de nuevo.
Abra la búsqueda web en la aplicación local de Strata.
Las referencias de búsqueda requieren una pregunta normal.
La investigación autónoma necesita un modelo de agente. Con este modelo, use la búsqueda web para adjuntar referencias.
Las referencias usan una respuesta simple con las herramientas desactivadas.
Importar SKILL.md
Desde carpeta
Solo importa instrucciones. Los scripts, referencias y dependencias complementarios no se instalan ni ejecutan. Revise las instrucciones antes de importar.
Elegir SKILL.md
O pegar SKILL.md
Vista previa de habilidad
Importando…
Hasta 64 KiB. Requiere nombre y descripción en los metadatos iniciales y hasta 40000 caracteres de instrucciones.
Separe las frases con comas. En el chat normal, las habilidades activas se aplican cuando la pregunta contiene una frase de activación.
Pregunta de ejemplo: {0}
por favor, ayúdeme con esta tarea.
Elegir otras instrucciones
Desde carpeta añade una carpeta local de habilidades. La compatibilidad del agente cubre las instrucciones; los recursos complementarios pueden necesitar configuración aparte.
En el chat normal, las habilidades activas se aplican si la pregunta contiene una frase de activación. Revise antes el ámbito del espacio de trabajo.
Elija un archivo SKILL.md de hasta 64 KiB.
No se pudo leer el archivo de habilidad.
No se pudo mostrar la vista previa de la habilidad.
No se pudo importar la habilidad.
SKILL.md necesita un nombre de hasta 80 caracteres.
SKILL.md necesita una descripción de una sola línea de hasta 400 caracteres.
Las instrucciones deben tener 1–40000 caracteres.
Ya existe una habilidad con este nombre. Cámbielo antes de importar.
Abra la importación de habilidades en la aplicación local de Strata.
Primero revise la vista previa y confirme Importar.
Añada al menos una frase de activación antes de importar.
DuckDuckGo / Bing funciona sin clave API. Use la búsqueda web del chat para elegir una consulta y adjuntar fragmentos.
No hay una clave guardada. La búsqueda autónoma del agente recurre a DuckDuckGo / Bing. La búsqueda web manual solo admite el proveedor integrado.
Buscar`,
fr: `Recherche web
Seule votre requête est envoyée à DuckDuckGo / Bing. Aucun compte ni clé API n’est nécessaire. Votre conversation et votre mémoire privée ne servent pas à créer la requête.
Requête de recherche
Destination
Utiliser les résultats dans la question
Ouvrir les réglages des outils
Recherche…
Extraits uniquement. Les pages complètes n’ont pas été lues. Les références servent à la prochaine réponse, avec les outils autonomes désactivés.
Aucun résultat. Essayez une autre requête.
Références prêtes · outils désactivés pour cette réponse
Retirer les références
Références de recherche · extraits uniquement
Ouvrir la recherche web pour choisir une requête et joindre des extraits
Les réglages de recherche web n’ont pas pu être chargés.
La recherche web est désactivée. Activez-la dans Paramètres → Outils.
La recherche manuelle prend actuellement en charge DuckDuckGo / Bing uniquement. Choisissez ce fournisseur dans Paramètres → Outils.
Une conversation valide est nécessaire.
Confirmez la recherche avant de l’envoyer.
Saisissez une requête de 1 à 400 caractères.
Une recherche est déjà en cours. Arrêtez-la ou attendez.
La recherche a été annulée.
La recherche web est limitée pour le moment. Réessayez plus tard.
Le fournisseur de recherche est inaccessible. Vérifiez la connexion puis réessayez.
Les références ont expiré ou appartiennent à une autre conversation. Relancez la recherche.
Ouvrez la recherche web dans l’application locale Strata.
Les références de recherche nécessitent une question ordinaire.
La recherche autonome nécessite un modèle agent. Avec ce modèle, utilisez la recherche web pour joindre des références.
Les références utilisent une réponse simple avec les outils désactivés.
Importer SKILL.md
Depuis un dossier
Seules les instructions sont importées. Les scripts, références et dépendances associés ne sont ni installés ni exécutés. Vérifiez les instructions avant l’importation.
Choisir SKILL.md
Ou coller SKILL.md
Aperçu de la compétence
Importation…
Jusqu’à 64 Kio. Un nom et une description sont requis dans les métadonnées initiales, avec au plus 40000 caractères d’instructions.
Séparez les expressions par des virgules. Dans une conversation ordinaire, les compétences actives s’appliquent si votre question contient une expression de déclenchement.
Exemple de question : {0}
veuillez m’aider pour cette tâche.
Choisir d’autres instructions
Depuis un dossier ajoute un dossier local de compétence. La compatibilité de l’agent couvre les instructions ; les ressources associées peuvent nécessiter une configuration distincte.
Dans une conversation ordinaire, les compétences actives s’appliquent aux expressions de déclenchement de votre question. Vérifiez d’abord la portée de l’espace de travail.
Choisissez un fichier SKILL.md de 64 Kio maximum.
Le fichier de compétence n’a pas pu être lu.
L’aperçu de la compétence n’a pas pu être affiché.
La compétence n’a pas pu être importée.
SKILL.md nécessite un nom de 80 caractères maximum.
SKILL.md nécessite une description sur une ligne de 400 caractères maximum.
Les instructions doivent contenir de 1 à 40000 caractères.
Une compétence porte déjà ce nom. Renommez-la avant l’importation.
Ouvrez l’importation de compétences dans l’application locale Strata.
Affichez d’abord l’aperçu et confirmez l’importation.
Ajoutez au moins une expression de déclenchement avant l’importation.
DuckDuckGo / Bing fonctionne sans clé API. Dans la conversation, utilisez la recherche web pour choisir une requête et joindre des extraits.
Aucune clé n’est enregistrée. La recherche autonome de l’agent utilise DuckDuckGo / Bing en secours. La recherche web manuelle prend uniquement en charge le fournisseur intégré.
Rechercher`,
ko: `웹 검색
입력한 검색어만 DuckDuckGo / Bing으로 전송됩니다. 계정이나 API 키는 필요하지 않습니다. 대화와 비공개 기억은 검색어를 만드는 데 사용되지 않습니다.
검색어
대상
검색 결과를 질문에 사용
도구 설정 열기
검색 중…
발췌문만 제공됩니다. 전체 페이지는 읽지 않았습니다. 참조 자료는 자율 도구를 끈 상태로 다음 답변에 적용됩니다.
결과가 없습니다. 다른 검색어를 입력하세요.
검색 참조 준비 완료 · 이 답변의 도구 꺼짐
참조 제거
검색 참조 · 발췌문만
웹 검색을 열어 검색어를 선택하고 발췌문 첨부
웹 검색 설정을 불러오지 못했습니다.
웹 검색이 꺼져 있습니다. 설정 → 도구에서 켜세요.
수동 검색은 현재 DuckDuckGo / Bing만 지원합니다. 설정 → 도구에서 선택하세요.
유효한 채팅이 필요합니다.
전송하기 전에 검색을 확인하세요.
1–400자의 검색어를 입력하세요.
이미 검색 중입니다. 중지하거나 기다리세요.
검색이 취소되었습니다.
현재 웹 검색이 제한되어 있습니다. 나중에 다시 시도하세요.
검색 제공업체에 연결하지 못했습니다. 연결을 확인하고 다시 시도하세요.
검색 참조가 만료되었거나 다른 채팅의 자료입니다. 다시 검색하세요.
로컬 Strata 앱에서 웹 검색을 여세요.
검색 참조에는 일반 질문이 필요합니다.
자율 조사에는 에이전트 모델이 필요합니다. 이 모델에서는 웹 검색으로 참조를 첨부하세요.
검색 참조는 도구를 끈 일반 답변에 사용됩니다.
SKILL.md 가져오기
폴더에서
지침만 가져옵니다. 관련 스크립트, 참조 자료 및 의존성은 설치하거나 실행하지 않습니다. 가져오기 전에 지침을 검토하세요.
SKILL.md 선택
또는 SKILL.md 붙여넣기
스킬 미리 보기
가져오는 중…
최대 64 KiB입니다. 머리말 메타데이터에 이름과 설명이 필요하며 지침은 최대 40000자입니다.
문구를 쉼표로 구분하세요. 일반 채팅에서는 질문에 실행 조건 문구가 포함되면 활성화된 스킬이 적용됩니다.
예시 질문: {0}
이 작업을 도와주세요.
다른 지침 선택
폴더에서는 로컬 스킬 폴더를 추가합니다. 에이전트 호환성은 지침에 적용되며 관련 리소스는 별도 설정이 필요할 수 있습니다.
일반 채팅에서는 질문에 실행 조건 문구가 포함되면 활성화된 스킬이 적용됩니다. 사용 전에 작업 공간 범위를 확인하세요.
64 KiB 이하의 SKILL.md 파일을 선택하세요.
스킬 파일을 읽지 못했습니다.
스킬을 미리 보지 못했습니다.
스킬을 가져오지 못했습니다.
SKILL.md에는 최대 80자의 이름이 필요합니다.
SKILL.md에는 한 줄로 된 최대 400자의 설명이 필요합니다.
스킬 지침은 1–40000자여야 합니다.
이 이름의 스킬이 이미 있습니다. 가져오기 전에 이름을 바꾸세요.
로컬 Strata 앱에서 스킬 가져오기를 여세요.
먼저 스킬을 미리 보고 가져오기를 확인하세요.
가져오기 전에 실행 조건 문구를 하나 이상 추가하세요.
DuckDuckGo / Bing은 API 키 없이 작동합니다. 채팅의 웹 검색에서 검색어를 선택하고 발췌문을 첨부하세요.
저장된 키가 없습니다. 자율 에이전트 검색은 DuckDuckGo / Bing을 대체 수단으로 사용합니다. 수동 웹 검색은 내장 제공업체만 지원합니다.
검색`,
pt: `Pesquisar na web
Apenas a consulta digitada é enviada ao DuckDuckGo / Bing. Não é necessária uma conta nem chave de API. A conversa e a memória privada não são usadas para criar a consulta.
Consulta de pesquisa
Destino
Usar resultados na pergunta
Abrir configurações das ferramentas
Pesquisando…
Apenas trechos. As páginas completas não foram lidas. As referências valem para a próxima resposta com as ferramentas autônomas desativadas.
Nenhum resultado. Tente outra consulta.
Referências prontas · ferramentas desativadas nesta resposta
Remover referências
Referências de pesquisa · apenas trechos
Abrir a pesquisa web para escolher uma consulta e anexar trechos
Não foi possível carregar as configurações da pesquisa web.
A pesquisa web está desativada. Ative em Configurações → Ferramentas.
A pesquisa manual atualmente só aceita DuckDuckGo / Bing. Escolha em Configurações → Ferramentas.
É necessário um chat válido.
Confirme a pesquisa antes de enviá-la.
Digite uma consulta de 1–400 caracteres.
Uma pesquisa já está em andamento. Pare ou aguarde.
A pesquisa foi cancelada.
A pesquisa web está limitada no momento. Tente novamente mais tarde.
Não foi possível acessar o provedor de pesquisa. Verifique a conexão e tente novamente.
As referências expiraram ou pertencem a outro chat. Pesquise novamente.
Abra a pesquisa web no aplicativo local do Strata.
As referências de pesquisa exigem uma pergunta comum.
A pesquisa autônoma exige um modelo de agente. Com este modelo, use a pesquisa web para anexar referências.
As referências usam uma resposta simples com as ferramentas desativadas.
Importar SKILL.md
Da pasta
Importa somente instruções. Scripts, referências e dependências associados não são instalados nem executados. Revise as instruções antes de importar.
Escolher SKILL.md
Ou colar SKILL.md
Prévia da habilidade
Importando…
Até 64 KiB. Exige nome e descrição nos metadados iniciais e até 40000 caracteres de instruções.
Separe as frases por vírgulas. No chat comum, habilidades ativas se aplicam quando a pergunta inclui uma frase de acionamento.
Exemplo de pergunta: {0}
por favor, ajude com esta tarefa.
Escolher outras instruções
Da pasta adiciona uma pasta local de habilidade. A compatibilidade do agente cobre as instruções; recursos associados podem exigir configuração separada.
No chat comum, habilidades ativas se aplicam quando a pergunta inclui uma frase de acionamento. Confira antes o escopo do espaço de trabalho.
Escolha um arquivo SKILL.md de até 64 KiB.
Não foi possível ler o arquivo da habilidade.
Não foi possível visualizar a habilidade.
Não foi possível importar a habilidade.
SKILL.md precisa de um nome de até 80 caracteres.
SKILL.md precisa de uma descrição em uma linha de até 400 caracteres.
As instruções devem ter de 1 a 40000 caracteres.
Já existe uma habilidade com esse nome. Renomeie antes de importar.
Abra a importação de habilidades no aplicativo local do Strata.
Primeiro confira a prévia e confirme a importação.
Adicione pelo menos uma frase de acionamento antes de importar.
DuckDuckGo / Bing funciona sem chave de API. Use a pesquisa web no chat para escolher uma consulta e anexar trechos.
Nenhuma chave foi salva. A pesquisa autônoma do agente usa DuckDuckGo / Bing como alternativa. A pesquisa web manual só aceita o provedor integrado.
Pesquisar`,
ru: `Поиск в интернете
В DuckDuckGo / Bing отправляется только введённый запрос. Учётная запись и ключ API не нужны. Переписка и личная память не используются для составления запроса.
Поисковый запрос
Назначение
Использовать результаты в вопросе
Открыть настройки инструментов
Идёт поиск…
Только фрагменты. Полные страницы не читались. Источники применяются к следующему ответу с отключёнными автономными инструментами.
Результатов нет. Попробуйте другой запрос.
Источники готовы · инструменты для этого ответа отключены
Убрать источники
Результаты поиска · только фрагменты
Открыть поиск, выбрать запрос и приложить фрагменты
Не удалось загрузить настройки поиска в интернете.
Поиск в интернете отключён. Включите его в Настройки → Инструменты.
Ручной поиск пока поддерживает только DuckDuckGo / Bing. Выберите его в Настройки → Инструменты.
Нужен действительный чат.
Подтвердите поиск перед отправкой.
Введите поисковый запрос длиной 1–400 символов.
Поиск уже идёт. Остановите его или подождите.
Поиск отменён.
Поиск в интернете сейчас ограничен. Попробуйте позже.
Не удалось связаться с поисковой службой. Проверьте соединение и повторите попытку.
Срок действия источников истёк или они относятся к другому чату. Выполните поиск заново.
Откройте поиск в локальном приложении Strata.
Для источников поиска нужен обычный вопрос.
Для автономного исследования нужна модель агента. С этой моделью используйте поиск, чтобы приложить источники.
Источники используются в обычном ответе с отключёнными инструментами.
Импорт SKILL.md
Из папки
Импортируются только инструкции. Связанные скрипты, источники и зависимости не устанавливаются и не запускаются. Проверьте инструкции перед импортом.
Выбрать SKILL.md
Или вставить SKILL.md
Предпросмотр навыка
Идёт импорт…
До 64 КиБ. В начальных метаданных нужны имя и описание, а инструкции должны содержать не более 40000 символов.
Разделяйте фразы запятыми. В обычном чате включённые навыки применяются, если вопрос содержит фразу активации.
Пример вопроса: {0}
пожалуйста, помогите с этой задачей.
Выбрать другие инструкции
Из папки добавляет локальную папку навыка. Совместимость агента охватывает инструкции; связанным ресурсам может потребоваться отдельная настройка.
В обычном чате включённые навыки применяются по фразам активации в вопросе. Перед использованием проверьте область рабочего пространства.
Выберите файл SKILL.md размером не более 64 КиБ.
Не удалось прочитать файл навыка.
Не удалось открыть предпросмотр навыка.
Не удалось импортировать навык.
В SKILL.md требуется имя длиной до 80 символов.
В SKILL.md требуется однострочное описание длиной до 400 символов.
Инструкции навыка должны содержать 1–40000 символов.
Навык с таким именем уже существует. Переименуйте его перед импортом.
Откройте импорт навыков в локальном приложении Strata.
Сначала просмотрите навык и подтвердите импорт.
Перед импортом добавьте хотя бы одну фразу активации.
DuckDuckGo / Bing работает без ключа API. Воспользуйтесь поиском в чате, чтобы выбрать запрос и приложить фрагменты.
Ключ не сохранён. Автономный поиск агента использует DuckDuckGo / Bing как запасной вариант. Ручной поиск поддерживает только встроенную службу.
Поиск`,
};
for (const [language, text] of Object.entries(translations)) {
  const values = text.split('\n'); assert.equal(values.length, ids.length, language);
  const file = resolve(directory, language + '.json');
  const catalog = JSON.parse(readFileSync(file, 'utf8'));
  for (const [index, id] of ids.entries()) { assert.ok(english[id]); catalog[id] = values[index]; }
  assert.deepEqual(Object.keys(catalog).sort(), Object.keys(english).sort());
  writeFileSync(file, JSON.stringify(Object.fromEntries(Object.keys(english).map(id=>[id,catalog[id]])),null,2)+'\n');
}
console.log(`Completed ${ids.length} existing interface keys in ${Object.keys(translations).length} languages.`);
