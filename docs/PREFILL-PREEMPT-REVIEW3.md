# Review 3 — 2ab3103, T18/T19 i diagnostyka awarii

Przegląd gałęzi `feat/prefill-preempt` z 2026-10-03: HEAD `2ab3103`, poprawka REVIEW2
`a1a137a`, nowa admisja/scheduler `638437f`. Zmiany z tego przeglądu są lokalne.
Nie wykonano push, kompilacji silnika CUDA ani przebiegu modelowego. Nie ma tu GPU,
nvcc ani cmake; CPU C++ kompilowano bezpośrednio przez g++.

## Ocena przekazanego raportu

Pierwszy fail B0 i `repeat-3` pozostają niepowodzeniami. Przekazany RUNNOTES je
zachowuje, co jest właściwe. Rerun B0 jest diagnostyką, a nie zaliczeniem pierwszego
przebiegu. Brak stderr nie wyjaśnia śmierci procesu.

Harness nie wymaga dokładnie max_new tokenów: porównuje tokeny z referencją.
Wczesny EOS jest poprawny, gdy referencja kończy się tak samo. Komunikat 5 vs 128
wskazuje różne odpowiedzi obu ramion; nie można skasować faila jako zwykłego EOS.
Bez tokenów, pierwszej różnicy, finish reasons i trace obu ramion przyczyna nie
jest ustalona. Ten patch nie deklaruje naprawy numerycznej przyczyny tego faila.

P01/P02-lite są smoke testami. Mediany 1.2 s zaokrąglone do 0.1 s nie rozstrzygają
progu 1%. RUNNOTES nazywa 6 przebiegów ABBA x3; pełne trzy bloki ABBA to 12
przebiegów. Przed wnioskami o <1% trzeba wyjaśnić kolejność i zachować surowe czasy.
50K i 200K to odrębne pomiary. B wait w starym benchmarku jest czasem do DONE,
nie samym oczekiwaniem w kolejce ani TTFT.

## Potwierdzone problemy kodu i poprawki

| Priorytet | Warunek w 2ab3103 | Poprawka |
|---|---|---|
| P1 | T19 kończy tylko czekanie na pustą kolejkę; następnie A konkuruje o zwykły threading.Lock. Brak gwarancji FIFO lub wznowienia przed nowymi requestami. | EngineGate: zwykłe okresy mają FIFO, A po deadline dostaje priorytet następnego okresu. B nadal kończy aktualny okres. Feature OFF zachowuje dotychczasowy lock. |
| P1 | T18 pomija kopię całego checks i run.ids; pos*8 nie liczy pełnego promptu. Dla długiego ogona i checkpointów estymata może być za niska mimo 1 MiB slack. | Liczone są pełne ids, running ids, checkpointy, dead/block dwa razy zgodnie z kopią, katalogi KV, pooled i rzeczywisty zakres ringa. Arytmetyka z kontrolą overflow; przed kopią admisja fizycznego RAM z istniejącej conversation_memory. |
| P2 | pooled rośnie wektorem po każdej warstwie; pojemność może przewyższać rozmiar payloadu. | Jeden resize do łącznego rozmiaru; sprawdzenie rzeczywistych capacities i floor RAM po capture, zanim ogłosimy SUSPENDED. |
| P2 | pr nadal trzyma duże bufory po restore i przez dalszy prefill/decode. Przy następnym parku obok nowego rec żyje poprzedni snapshot. | Zwolnienie payloadu pr od razu po restore; zachowane skalarne metadane segmentu. |
| P2 | should_suspend liczy estymatę na każdym chunku także bez YIELD; komunikat odmowy jest static raz na cały proces. | Najpierw tani test YIELD. Odmowa i log raz na request; dalsze chunk boundaries nie powtarzają kosztownego admission. |
| P2 | atoll literówek/ujemnego snapshot-mib przechodziło do 0, czyli brak limitu; przesunięcie MiB mogło overflowować. | Istniejący parser from_chars z zakresem dla opcji pamięci; przeliczenie budżetu nie wrapuje. Limit czekania musi być finite i >=0. |
| P2 | collect(1) po CANCEL potrafiło wyrzucić T/DONE B; docstring obiecywał zachowanie innych requestów. | Bufor linii per rid; cancel test porównuje też całe B. |
| P2 | B finish reason nie był bramkowany, repeated parks zapisywały tylko pierwszą granicę. | Finish i pierwszy różniący się token w raporcie; wszystkie granice i postęp minimum jednego chunka sprawdzane. |
| P2 | close() ignorował exit przed QUIT oraz niezerowy kod po QUIT. Śmierć po DONE mogła zostać zaliczona. | Nieoczekiwany exit/EOF i wadliwe zamknięcie to fail; zapis returncode i nazwy sygnału. |
| P2 | Benchmark nazywał kolejką czas do DONE, nie porównywał odpowiedzi i nie przypinał expert_slots obu ramion. | Osobno queue wait, TTFT i total; czasy z pumpu, pin geometrii, token/finish parity przed interpretacją wydajności. |

EngineGate nie daje absolutnego limitu 30 s. Granica to: czas do deadline i
zgłoszenia ticketu A + pozostała długość aktualnego okresu B. Nie wywłaszcza decode.
Jeśli B dekoduje 120 s, A czeka na jego koniec. Nowe requesty nie mogą wyprzedzać
już zarezerwowanego okresu A. Anulowani/time-out waiters są usuwani z kolejki gate.

snapshot-mib ogranicza własny obraz parku, nie całe RSS silnika ani wspólny budżet
conversation cache. Admisja MemAvailable/floor jest kontrolą punktową; nie jest
rezerwacją pamięci w systemie i nie może zapobiec każdej zewnętrznej presji RAM.
trace też alokuje tymczasowe bufory. Ograniczenie dotyczy logicznych capacities
rekordu; narzuty alokatora pozostają osobnym pomiarem RSS.

Nie dodano nowego formatu snapshotu ani zmian kerneli/decode.

## Testy wykonane tutaj

- Końcowa wersja: 183 testy Python, OK, 7 pominiętych (99.232 s):
  engine_gate, preempt_regressions, preempt, server, lifecycle, detok, winjob,
  preempt_harness i preempt_bench. Zależności Jinja2/regex zainstalowane osobno.
- 6 testów gate, w tym rzeczywisty Service/FakeEngine: B -> RESUME A -> C po deadline.
- Harness: 18 testów CPU, w tym prawdziwe pipe'y subprocess do modelowego fake'a,
  zachowanie innego rid, EOS parity, wszystkie repeat granice, exit i sygnał.
- Benchmark: 3 testy odróżniające queue wait od TTFT i total.
- prefill_preempt_budget_test: 11 checks, g++ C++20 -Wall -Wextra -Werror.
- conversation_memory_test: 23 checks; conversation_cache_test: 4175 checks.
- git diff --check oraz py_compile: OK.
- Nie uruchomiono całego ctest, GPU, ASAN na CUDA ani benchmarku. Te wyniki nie
  potwierdzają parzystości modelu po zmianach C++ ani progu <1% throughput.

## Kolejność dalszej weryfikacji

1. Zbuduj nowy silnik. Zapisz nowy SHA, dirty diff, build flags, exe hash, config,
   model, args, STRATA_*, faktyczne expert_slots i sterownik. RUNNOTES opisuje
   przebiegi a1a137a; nie przypisuj ich nowej binarce T18/T19 lub temu patchowi.
2. Powtórz B0 i repeat-3 w świeżych katalogach, bez kasowania starych logów.
3. Po wyjaśnieniu awarii: T01–T06, potem T18/T19 na realnym silniku.
4. Dopiero dalej: P02 z pełną precyzją, P03 200K, T20 stress.

```bash
python3 -m unittest serve.test_engine_gate serve.test_preempt_regressions \
  tools.test_preempt_harness tools.test_preempt_bench
python3 tools/prefill_preempt_test.py --config strata-iq3_xxs.json \
  --quick --trace --interim-max-new 0 -k boundary-2048 --workdir /tmp/preempt3-B0-first
python3 tools/prefill_preempt_test.py --config strata-iq3_xxs.json \
  --trace --max-new 128 -k repeat-3 --workdir /tmp/preempt3-repeat128-first
```

Harness zapisuje `*.protocol.jsonl` (in/out, monotonic timestamp, PID, args,
STRATA_*, zebrane tokeny, expected/actual przy failu, EOF/QUIT/kill i returncode),
`reference-*.json` oraz istniejący stderr `*.log`. Te dane zachowaj w komplecie.
Nowa diagnostyka nie potrafi odzyskać returncode starego B0 z samych RUNNOTES.

Dla B0: odczytaj `returncode`. -9 to SIGKILL, -11 SIGSEGV, -15 SIGTERM.
SIGKILL nie dowodzi OOM: trzeba skorelować czas/PID z logiem systemowym i RSS.
Exit 0 przed QUIT jest naruszeniem protokołu procesu resident i też failuje.
Po failu sprawdź lokalnie log kernela z odpowiedniego przedziału (OOM, NVIDIA Xid),
a nie uruchamiaj automatycznie retry-to-green. Zewnętrzny kill i timeout harnessu
muszą być odróżnione od samoistnego exit; nowy dziennik zapisuje kill harnessu.

Dla repeat-3:

- Ustal pierwszy różniący się token B oraz finish obu ramion. Nie porównuj tylko
  długości ani nie zwiększaj max_new w nadziei, że EOS zniknie.
- Porównaj odpowiednie B w control/preempt: STATE_POINT prefill_done, potem
  VERIFY_POINT i WINDOW po tej samej pozycji. Różnica już po B prefill wskazuje
  ścieżkę resetu/prefillu/cache; pierwsza różnica w verify oknie wskazuje decode,
  geometrię lub niewłaściwą długość okna. Hash nie lokalizuje sam kernela.
- Loguj lend/layout, reuse, cvec, prompt length draftera, commitowany L, offered T
  i accepted count. spec-min-p=0 nie gwarantuje jednakowego accepted count.
- Dopiero z tym śladem wybierz kernel/bufor do dokładnego porównania. Nie maskuj
  różnicy wyłączeniem gatingu MTP lub tolerancją dobraną do tego faila.

## Konkretnie T18 na GPU

W obu ramionach pin geometry i deterministic args jak REVIEW2. Dla refusal nie
wpuszczaj B do silnika przed zakończeniem A: samo YIELD symuluje ofertę kolejki.
To pozwala przechwycić końcowy hash A zanim B go zastąpi.

1. Control bez preemption: warm -> A -> hash A -> B. Zachowaj oba wyniki.
2. Candidate: w engine_args(preempt=True) zamień JEDNĄ wartość
   --prefill-preempt-snapshot-mib na 1. Podczas A wyślij YIELD po PP; żadnego GEN B
   przed DONE A. Oczekuj 0 SUSPENDED, jednego logu odmowy, A/hash identycznego.
   Następnie B ma być zdrowe i zgodne z control.
3. Powtórz z długim promptem i wieloma prompt_cache_every checkpointami: to
   przypadek, który pomijała stara estymata. Sprawdź log budżetu i RSS.
4. Z większym limitem żądaj jednego realnego parku. Zalogowane snapshot MiB
   obejmuje capacities całego rekordu, a nie tylko główne payloady KV.
5. Powtórz po RESUME A i następnym parku. Sprawdź, czy RAM poprzedniego obrazu
   jest zwolniony przed kopią następnego (uwzględnij high-water mark alokatora).
6. Osobno odmowa po physical RAM admission / po capture floor: brak SUSPENDED,
   stan A dalej poprawny. Nie wywołuj celowo globalnego OOM na roboczej maszynie.

Uwaga: zwykły parity harness przypina snapshot-mib=4096; niski limit z configu
nie steruje jego boundary testami. T18 potrzebuje powyższego dedykowanego armu.

## Konkretnie T19 na GPU/HTTP

Włącz flagę w silniku (config args) i serwerze, max-wait-s=1. Rozgrzej serwer.
A musi mieć kilka pełnych chunków pozostałych. Kolejkuj B i C przed parkiem A;
B powinno trwać >1 s. Utrzymuj dalszy napływ krótkich requestów przez 10–30 s.

Oczekuj: po logu rezerwacji i DONE aktywnego B następny engine request to RESUME A,
przed C i nowymi arrivals. Zmierz deadline, ticket i rzeczywiste RESUME. A musi
przetworzyć kolejny chunk; nie wymagaj zakończenia całego A po jednym wznowieniu.
Powtórz z anulowanym C, disconnect A i błędem B. Z limitami 0/1/30 sprawdź odpowiednio
dotychczasowe oczekiwanie na drain albo nową rezerwację. Sprawdź FIFO zwykłych B/C/D.

## Benchmark i pozostałe otwarte bramki

Nowy benchmark zapisuje `benchmark.json`, pin slots po probe na tym samym ctx,
porównuje tokeny/finish A+B i publikuje trzy osobne latencje. To nadal jeden pomiar
OFF/ON; do rozkładu/noise potrzebne są niezależne świeże katalogi i bloki ABBA.

```bash
python3 tools/prefill_preempt_bench.py --config strata-iq3_xxs.json \
  --tokens 200000 --max-context 262144 --kv-resident 8192 \
  --chunk 2048 --max-new 32 --workdir /tmp/preempt3-200k-first
```

Ta konfiguracja streaming KV jest kandydatem do sprawdzenia na 16 GB, nie
obietnicą, że mieści się na karcie. Geometry/VRAM admission musi przejść probe.
Nie zmieniaj między ramionami expert_slots ani nie zaliczaj 50K jako 200K.
P02 wymaga rzeczywiście długiego niekonkurencyjnego prefillu i surowych czasów;
logging trace ma być wyłączony w pomiarach wydajności modelu.

Pozostają T01–T06 i T20 z checklisty REVIEW2, GPU T18/T19, oba zgłoszone faile,
oraz poprawna agregacja prompt_ms/reused/read przez okresy wznowienia.
Zauważono też istniejące ograniczenie lazy_load: can_preempt=False przed READY
powoduje wyłączenie Service.preempt na stałe. Na razie nie łącz --lazy z tym
feature; potrzebne jest osobne rozwiązanie odroczonego capability handshake.

W HEAD 2ab3103 brakowało śledzonych docs/PREFILL-PREEMPT*.md mimo odwołań w kodzie
i raporcie. Patch przywraca historyczną checklistę REVIEW2 i dodaje ten dokument.
Oryginalny dokument projektowy/audyt granicy nadal trzeba dołączyć przed PR.
