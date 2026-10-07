# Review 2 — feat/prefill-preempt, 29ef162

Historyczny zapis przeglądu REVIEW2. Poprawki kodu zostały następnie włączone przez
`a1a137a`; dalsze ustalenia i nowy patch opisuje `PREFILL-PREEMPT-REVIEW3.md`.

Przegląd kodu z 2026-10-03, po rebase na 99f3dbd. Poprawki są lokalne; nie wykonano push.
Raport użytkownika opisuje udane przebiegi GPU. Ten przegląd nie odtwarza ich: środowisko
nie ma kompilatora CUDA, GPU ani modelu. Zgodność tokenów w podanych scenariuszach nie
pokrywa niżej opisanych interakcji z cache i cyklem życia serwera.

## Znalezione błędy i zakres poprawki

| Priorytet | Miejsce | Błąd / warunek | Zmiana |
|---|---|---|---|
| P1 | generate.cpp, RESUME / ConversationCache | Bufory `retain()` po odtworzeniu B nie należą do A. Samo `limit_reuse(resume)` zachowuje prefiks B jako rzekomo niezmienione KV A. Błąd może wyjść dopiero przy następnym zapisie/restore rozmowy. | Parkowanie kończącego się B przed nadpisaniem; `limit_reuse(0)` przy zmianie właściciela; przywrócenie checkpointów zapisanych z A. |
| P1 | EngineRequest.drain | Bezterminowe `lines.get()` po STOP/CANCEL blokuje właściciela silnika, jeśli silnik nie odpowie. | Deadline oparty na istniejącym `engine_silence_s` i allowance prefillu; istniejąca procedura `_silent`. |
| P1 | Service.run_preemptable | Heartbeat podczas prefillu nie kończy pracy po cancel; request anulowany w kolejce nadal wysyła GEN. | Sprawdzanie cancel przed GEN i na heartbeat, potem STOP/drain. |
| P1 | Service.run | Włączenie preemption pomija cały mechanizm reasoning_budget_tokens. | Requesty z aktywnym budżetem myślenia korzystają z istniejącej ścieżki obsługującej budżet. Dla nich preemption pozostaje wyłączone — jawne ograniczenie, nie pełna integracja. |
| P1 / warunkowy | susp_save/susp_restore | Wektory snapshotu mają `qsa_alloc` elementów, ale były indeksowane globalnym `i=qsa_ord0+j`. Dla niezerowego ord0 grozi wyjście poza bufor. | Stan urządzenia nadal indeksowany `i`, kompaktowy snapshot indeksowany `j`. Domyślny single-GPU/ord0=0 nie wykrywa błędu; layer-split jest nadal wyłączony. |
| P2 | tools/prefill_preempt_test.py | Przy braku opcjonalnych dead/pooled_full skrócona lista kluczy była zipowana z nieskróconymi grupami regexu. `ple_prev0/1` stawały się None. | Stała mapa grup; pomijanie tylko brakujących wartości. Test pokazuje poprawne 123/456 zamiast None/None. |
| P2 | trace_state | STATE_EXTRA nie jest porównywane przez harness; bieżący STATE_HASH nie zawierał dead/pooled_full mimo opisu rozszerzonej bramki. | dead/pooled_full/block w głównym STATE_HASH i parserze; iteracja tylko po własnych QSA. |
| P2 | Service / EngineRequest | History/metrics pobierały współdzielone `engine.last`, więc mogły przypisać DONE od B do anulowanego A; finalizacja A mogła wyczyścić status B. | DONE przechowywane w obiekcie requestu, czas rozpoczęcia lokalny, czyszczenie statusu tylko przez właściciela request_id. |
| P2 | Service.unload / restart | Zaparkowany request nie blokował unload; ścieżka preempt omijała ensure_loaded i jego hooki/admisję. | parked_requests blokuje unload; wspólne ensure_loaded; powiązanie EngineRequest z konkretnym obiektem procesu. |
| P2 | prefill segment loop | `sp.suspended()` może być starym wynikiem poprzedniego `sp.run`, gdy teraz wykonano read_windows. Po nieudanej admisji krótką końcówkę można było policzyć inną ścieżką. | Odczyt suspended tylko dla wykonanego batched run; zachowanie ścieżki i layoutu do końca przerwanego segmentu również po odmowie parku. |
| P2 | lend | Bezwarunkowe lend pełnego chunka zmieniało małe prompty również przy wyłączonej funkcji. | Zapis oryginalnego `segment_chunk`; odtworzenie dokładnego layoutu tylko dla kontynuowanego segmentu. Pozostałe segmenty zachowują upstream request-sized lending. |

Nie zmieniono kerneli modelu ani algorytmu decode. Zmiany C++ wymagają kompilacji i bramek GPU.
Fingerprint nie jest pomiarem ULP ani dosłownym porównaniem wszystkich bajtów.

## Weryfikacja wykonana tutaj

- 11 nowych testów serwera: 11/11 po zmianie. Na czystym 29ef162 ten sam plik:
  8 failures, 2 errors, 1 pass. Testy nie wymagają modelu ani HTTP.
- 7 testów harnessu (w tym dwa nowe przypadki parsera) przechodzi.
- Istniejący `conversation_cache_test`, kompilacja C++20 na CPU: 4175 checks passed.
- Pełny zestaw server/preempt/lifecycle/detok/winjob oraz nowe regresje i harness:
  163 testy w 88,719 s, OK, 7 pominiętych. Wynik po uzupełnieniu zależności regex.
- Brak testu GPU, kompilacji silnika CUDA/HIP, ASAN snapshotu na CUDA i nowych pomiarów czasu.

## Przygotowanie testów GPU

1. Zapisz SHA, build flags, wersję sterownika, model/kwantyzację, typ KV, wszystkie argumenty
   i istotne STRATA_*; zanotuj kartę, RAM oraz faktyczny expert_slots.
2. Zbuduj trzy warianty: upstream 99f3dbd, gałąź z poprawką bez flagi i gałąź z flagą.
   Dwie ostatnie muszą używać tej samej binarki. Dla porównań liczbowych identyczne ustawienia.
3. Maszyna bez innych silników podczas pomiaru. Nie używaj blanket `pkill -x strata`.
   Zabijaj tylko PID-y procesów uruchomionych przez swój test.
4. Zachowaj pierwsze niepowodzenie i cały log. Nie zaliczaj testu po późniejszym udanym retry.
5. Do ścisłej parzystości: greedy, stała geometria cache, brak migracji ekspertów przez cały
   przebieg, suffix-draft off i spec-min-p 0. Ostatnia opcja NIE stabilizuje acceptance count.
6. Hashuj przy tych samych pozycjach i liczbie commitowanych tokenów. MTP poza aktywnym oknem
   oraz target stale cells nie są dowodem uszkodzenia aktywnego stanu. Loguj je osobno.
7. Ścieżkę produkcyjną z adaptacją i suffix drafterem przetestuj osobno: pełny wynik nie musi
   być bitowo taki sam jak w zamrożonym rozmieszczeniu CPU/GPU ekspertów.

## Szybkie komendy

Dopasuj plik JSON i ścieżkę silnika. Z repo:

```bash
python3 -m unittest serve.test_preempt_regressions tools.test_preempt_harness
python3 -m unittest serve.test_preempt serve.test_server serve.test_lifecycle serve.test_detok serve.test_winjob

python3 tools/prefill_preempt_test.py --config strata-iq3_xxs.json --trace --quick \
  --interim-max-new 0 --workdir /tmp/preempt2-B0
python3 tools/prefill_preempt_test.py --config strata-iq3_xxs.json --trace --quick \
  --interim-max-new 1 --workdir /tmp/preempt2-B1
python3 tools/prefill_preempt_test.py --config strata-iq3_xxs.json --trace --quick \
  --interim-max-new 32 --workdir /tmp/preempt2-B32
python3 tools/prefill_preempt_test.py --config strata-iq3_xxs.json --trace \
  --tail-tokens 33 -k park-only-last --workdir /tmp/preempt2-short-tail
python3 tools/prefill_preempt_test.py --config strata-iq3_xxs.json \
  --max-new 128 --workdir /tmp/preempt2-full
```

`--tail-tokens 33` jest nowym parametrem: ostatni token promptu rozpoczyna decode,
więc zostaje 32-tokenowy batched ogon. Zweryfikuj rzeczywistą pozycję SUSPENDED;
YIELD wysyłany po PP może zostać zauważony dopiero na następnej granicy.

## Scenariusze regresji — bramki przed merge

Testy T01–T06 mają pierwszeństwo: sprawdzają przypadki niepokryte prostym A→B→A.
Poniższe to specyfikacja testów modelowych, nie deklaracja ich wykonania przez ten przegląd.

| ID | Przebieg | Co musi zostać sprawdzone |
|---|---|---|
| T01 | B1 → C → B2 z rzeczywistym restore B z conversation cache; A1 długi park → B3 ze wspólnym prefiksem B2 → resume A1 → D → A2 | W logu musi wystąpić restore/retain B. A2 ponownie montuje zapis A. Logits i aktywny KV A2 zgodne z kontrolą; bez pomieszania KV B. Testuj różne prefiksy A/B i wspólny system prompt. |
| T02 | A z cvec on i checkpointami → park → B cvec off → resume A → D → następny turn A; potem odwrotne cvec | Checkpointy i identity należą do A. Nie wystarczy sprawdzić pierwszego outputu A po resume. |
| T03 | Test layoutu QSA z qsa_ord0>0 i qsa_alloc mniejszym od całkowitej liczby warstw | Kopie obsługują własne ordinals; strażnicy pamięci/ASAN bez naruszeń. To unit/integration snapshot helpera; nie włączaj nieobsługiwanego layer split jako obejścia. |
| T04 | Wielochunkowy A, pozostały ogon 1/32/64/65/255/256/490 tokenów prefillu; park przed ogonem | Porównaj z uninterrupted A: layout, batched vs windows, stan przed decode, logits. Wymuś granicę w harnessie silnika, jeśli timing YIELD ją przeskakuje. |
| T05 | Jak T04, ale odmowa snapshotu/alokacji po chunku | A kończy się tą samą ścieżką batched; brak ponownego użycia stale suspended=true, zapętlenia albo podwójnego parkowania. |
| T06 | Serial: B1→B2→A. Candidate: B1→A park→B2→A resume. Potem C i odtworzenie obu rozmów | Cache B nie znika bez potrzeby, A odzyskuje własne checkpointy; oba późniejsze cache hits poprawne. |
| T07 | Anuluj B przed zdobyciem silnika, A nadal prefills | Ani jeden GEN dla anulowanego B. Request C działa. Mierz niepotrzebne parki A osobno. |
| T08 | Anuluj A przy heartbeat prefillu, bez żadnego T | STOP zostaje wysłany; obsłużony po bezpiecznej granicy, a nie dopiero po całym decode. Po DONE anulowania C działa. |
| T09 | Engine nie odpowiada na STOP/CANCEL, silnik bez stdout, śmierć tuż przed DONE | Konfigurowany timeout zwalnia request i kończy wadliwy proces. Brak deadlocku FIFO. Oddzielny test `engine_silence_s=0` świadomie wyłącza deadline. |
| T10 | A zaparkowany; B umiera; C zdąży restartować silnik zanim A wróci | A odrzucony przez identity procesu; żaden RESUME/CANCEL A nie trafia do nowego procesu. |
| T11 | A zaparkowany; POST /unload i idle-unload między okresami własności | busy, proces pozostaje żywy, A wznawia się poprawnie. Wymuszony administracyjny reset ma kończyć A jawnie. |
| T12 | Thinking budget z config/request; OpenAI stream/nonstream i Anthropic | Identyczne wrap-up i odpowiedź z feature off/on; patch używa dla aktywnego budżetu ścieżki bez preemption. Jawne budget=0 nadal może korzystać z preemption. |
| T13 | Barrier: A kończy i opuszcza lock; B już aktualizuje status; dopiero potem finalizacja A | /status nadal opisuje B i busy=true. History A nie zawiera timings/tokenów B. Sprawdź też cancel A w czasie B. |
| T14 | A parkuje wielokrotnie przy --prefill-preempt-max 1/2/16; co najmniej 10 cykli dla wariantu 16 | Nigdy ponad limit; po resume co najmniej jeden chunk; brak narastającego dryfu. Nie ograniczaj się do obecnego repeat-3. |
| T15 | KV int8/q4_0/FP16; resident i streamed; A przekracza resident window; MTP ring wrap | Weryfikacja aktywnych stron, map residency, idx_dead/tail/pooled/spare/block i PLE. First-token logits po resume zgodne. |
| T16 | A tekst → park → B obraz → A tekst; osobno B steering | M-RoPE wraca do identity A, brak zapożyczonych embeddings. A obraz nie parkuje w MVP. |
| T17 | Nieobsługiwane GPU/layer split | Flaga skutecznie wyłączona lub startup refused, INFO zgodne, serwer nie czeka na SUSPENDED. |
| T18 | Mały limit RAM snapshotów i telemetry admission failure | Wymagane: odmowa przed dużą alokacją, A działa dalej, B czeka. Nadal bramka otwarta — obecny SuspReq nie ma wspólnej admisji/budżetu. |
| T19 | A długi, B/C/D stale nadchodzą | Wymagany mierzalny limit odwlekania A. Nadal bramka otwarta: obecny wait-until-queue-empty może głodzić A, a threading.Lock nie zapewnia FIFO. |
| T20 | 100 park/resume, następnie minimum kilkaset losowych requestów/cancel | Każdy request ma terminalny wynik; snapshot bytes i parked_requests wracają do zera; RSS/VRAM nie rosną liniowo. Odróżnij high-water allocatorów. |

## Benchmarki bez diagnostyki

Nie używaj --trace, STATE_HASH ani profilerów do pomiaru czasu.

| Pomiar | Konfiguracja i metryki | Kryterium |
|---|---|---|
| P01 feature OFF | upstream vs nowa binarka, bez flagi; 32/64/220/512/2048-tokenowe i długie prompty | Nie zmieniają się kształty/lending ani odpowiedzi; brak systematycznej regresji czasu. |
| P02 feature ON, brak kolejki | Te same prompty, ta sama binarka, flaga off/on; przeplatane ABBA, >=5 prób | 0 parków; cel <1% regresji prefillu, decode w szumie. Jeżeli szum >1%, wynik nierozstrzygający, nie automatyczny PASS. |
| P03 200K + 220 tokenów | B po potwierdzonym pierwszym chunku A; max_new A=32, B=32; off/on | Mierz B queue wait/TTFT/total, A total, snapshot/restore, faktyczny chunk. Cel >=80% redukcji B queue wait i <10% narzutu A dla jednego parku. |
| P04 50K na mniejszym GPU | Identycznie do P03, oznacz rozmiar i sprzęt | Osobny wynik; nie ekstrapoluj liniowo do 200K/1M — rozmiar i koszt snapshotu rosną. |
| P05 metryki wznowienia | Wymuś 2 parki A; porównaj logi faz i DONE/timings API | Czas i liczba świeżych tokenów obejmują wszystkie segmenty A; tokeny wyliczone wcześniej przez A nie są cache hit z innej rozmowy. Obecne resetowanie zegara/reused wymaga osobnej poprawki. |

## Nadal otwarte, nieprzykryte tym patchem

- Sprawiedliwy scheduler: wait-until-empty i zwykły Lock nie dają gwarancji FIFO/starvation.
- Wspólny budżet snapshotów/admisja RAM i konsolidacja SuspReq ze SavedConversation.
  Parkowanie B może mieć dodatkowy koszt i zużycie RAM; należy je zmierzyć.
- Agregacja prompt_ms/prompt_read/reused przez wiele zawieszeń; obecne engine timings
  opisują głównie ostatni segment. History w Pythonie nie kradnie już DONE od B, ale to
  nie naprawia definicji liczników emitowanych przez silnik.
- Pełna integracja reasoning budget z preemption — aktualnie bezpieczny fallback.
- Ochrona wariantu z dynamiczną migracją CPU/GPU ekspertów wymaga osobnej definicji
  numerycznej tolerancji i testów; nie udowadnia jej adapt-every=100000.

Nie oznaczaj zadania jako w pełni zakończonego, dopóki obowiązkowe T18/T19 i wyniki GPU
nie zostaną rozstrzygnięte. Nowy patch jest kandydatem do lokalnego build/test, nie
potwierdzonym wydaniem produkcyjnym.
