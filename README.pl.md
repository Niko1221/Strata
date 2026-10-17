<h1 align="center">Strata</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · **Polski**

<p align="center"><b>Uruchom model AI ze 125 miliardami parametrów na swoim komputerze do gier</b><br>
Karta graficzna NVIDIA lub AMD (od 12 GB) · Windows lub Linux · bezpłatny i z otwartym kodem źródłowym</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="Wokselowy ogród z pagodą, którego kod model napisał w Strata, działa w przeglądarce"></a><br>
<sub>Wokselowy ogród z pagodą stworzony na podstawie jednego polecenia na RTX 5070 ze Strata (IQ3_S, kontekst 128K) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">cały film (49 s)</a></sub></p>

Strata uruchamia **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** na zwykłym komputerze.
To duży, inteligentny model AI, który zazwyczaj potrzebuje serwera. Rozmawia, pisze kod, rozumie obrazy
oraz współpracuje z twoimi aplikacjami i agentami programistycznymi. Nic nie opuszcza twojego komputera.

## Jak szybko działa?

Zmierzyliśmy wydajność na dwóch zwykłych komputerach do gier. Jeden token to około ¾ słowa.

- **Generowanie odpowiedzi:** jak szybko pojawia się odpowiedź w krótkiej rozmowie. 60 tokenów na sekundę to
  szybciej, niż jesteś w stanie czytać.
- **Odczyt polecenia:** jak szybko model przetwarza przesłaną treść (tutaj dokument, kod lub historię rozmowy
  o długości 32K tokenów).

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM</th></tr>
<tr><td>

| Wariant | Generowanie odpowiedzi | Odczyt polecenia |
| --- | ---: | ---: |
| **Q2_0** | 94 tokeny/s | 2 650 tokenów/s |
| **IQ2_XS** | 79 tokenów/s | 2 090 tokenów/s |
| **IQ3_XXS** | 62 tokeny/s | 1 750 tokenów/s |
| **IQ3_S** | 53 tokeny/s | 1 620 tokenów/s |
| **Coder** | 55 tokenów/s | 2 180 tokenów/s |

</td><td>

| Wariant | Generowanie odpowiedzi | Odczyt polecenia |
| --- | ---: | ---: |
| **Q2_0** | 60 tokenów/s | 1 160 tokenów/s |
| **IQ2_XS** | 52 tokeny/s | 1 110 tokenów/s |
| **Coder** | 44 tokeny/s | 1 420 tokenów/s |

</td></tr>
</table>

NVIDIA: Q2_0 z silnikiem 0.1.36, pozostałe wiersze z 0.1.26 (odpowiedzi 4K, polecenia 32K). Pełne
zestawienia znajdziesz w [DETAILS.md](docs/DETAILS.md#speed-measured). Karta z większą ilością VRAM jest szybsza:
RTX 3090 (24 GB) powinna generować około 100–140 tokenów na sekundę. Długie rozmowy i inne karty:
[szybkość poszczególnych wariantów modelu](docs/MODELS.md#how-fast-is-each-size), [wyniki społeczności](docs/COMMUNITY_BENCHMARKS.md).

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Postaw mi kawę" height="50"></a><br>
<sub>Strata jest bezpłatna. Jeśli dobrze działa na twoim komputerze, kawa pomoże w dalszym rozwoju.</sub></p>

## Czego potrzebujesz

| | |
| --- | --- |
| **Karta graficzna** | **NVIDIA** GeForce RTX z serii 20, 30, 40 lub 50 albo **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 lub z serii RX 6800 / 6900. Potrzebujesz **co najmniej 12 GB VRAM**. |
| **RAM** | Co najmniej 32 GB. Ilość RAM decyduje, [który model](#który-model-wybrać) się zmieści. Przy 64 GB działają wszystkie warianty. |
| **Dysk** | Około 80 GB wolnego miejsca. Jeśli możesz, użyj SSD: pierwsze uruchomienie będzie znacznie szybsze. |
| **System** | Windows 10 / 11 lub Linux oraz aktualny sterownik karty graficznej NVIDIA lub AMD. |

Resztę skonfiguruje instalator. Dwie lub trzy karty mogą współdzielić model ([wiele GPU](docs/MULTI_GPU.md)).

Obsługa eksperymentalna, opracowana i przetestowana przez członków społeczności na własnych komputerach:

- **Starsze karty graficzne** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [starsze GPU](docs/OLDER_GPUS.md).
- **Intel Arc**, kompilacja ze źródeł pod Linuksem: [Intel Arc](docs/INTEL_ARC.md).
- **Maki z Apple Silicon (od M1 do M5)**, kompilacja ze źródeł przez `./setup.sh` (silnik Metal): [Mac](docs/MAC.md).
- **AMD Ryzen AI Max (Strix Halo)**, kompilacja ze źródeł pod Linuksem: [Strix Halo](docs/STRIX_HALO.md).
- **Starsze procesory bez AVX2**: działają, ale wolno. [Starsze CPU](docs/INSTALL.md#older-cpus-experimental).

Pełna lista: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Instalacja

### Zleć konfigurację swojemu asystentowi AI

Korzystasz z asystenta AI do programowania (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Wklej do niego ten tekst:

```text
Skonfiguruj Strata na tym komputerze: https://github.com/Niko1221/Strata — postępuj zgodnie z docs/AI_SETUP.md w tym repozytorium.
```

Sprawdzi kartę graficzną, RAM i dysk oraz wybierze odpowiedni model. Następnie go zainstaluje,
uruchomi i wyjaśni, jak podłączyć aplikacje. Narzędzia AI mogą też instalować, uruchamiać i zatrzymywać Strata
przez jego [serwer MCP](docs/MCP_SERVER.md).

### Albo zrób to samodzielnie

[Pobierz Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) i rozpakuj archiwum (lub użyj
`git clone`). **Windows:** kliknij dwukrotnie **`START-HERE.bat`**. **Linux:** uruchom **`./setup.sh`** w folderze Strata.

**Mac (Apple Silicon, od M1 do M5, macOS 15 lub nowszy):** jednorazowo zainstaluj narzędzia wiersza poleceń Apple (`xcode-select --install` w terminalu; sam Xcode nie jest potrzebny). Następnie otwórz terminal w folderze Strata i uruchom **`./setup.sh`**. Skrypt skompiluje silnik dla Maca na twoim komputerze (około 7 minut), zada te same pytania, pobierze model i go uruchomi. Przy kolejnym uruchomieniu `./setup.sh` wystartuje od razu. Dobór wariantu do twojego Maca i jego wydajność: [docs/MAC.md](docs/MAC.md).

Kroki są takie same dla NVIDIA i AMD. Instalator rozpoznaje kartę i konfiguruje odpowiedni silnik.
Zada ci kilka pytań:

- jaki model i wariant wybrać,
- jak duży ma być kontekst (ile tekstu model zapamiętuje),
- czy model ma rozpoznawać obrazy.

Za każdym razem naciśnij Enter, aby wybrać zalecaną odpowiedź. Instalator pobierze model (około 70 GB)
i go uruchomi. Jeśli pobieranie zostanie przerwane, po prostu uruchom go ponownie: wznowi pracę od miejsca,
w którym skończył. Przeglądarka otworzy aplikację Strata pod adresem `http://127.0.0.1:8080`.

> **Podczas uruchamiania modelu komputer może zwolnić lub nie odpowiadać przez 1–3 minuty** (najdłużej za
> pierwszym razem). Strata ładuje 35–55 GB do RAM i blokuje część tej pamięci na potrzeby karty graficznej. To normalne.
> Poczekaj i nie zamykaj okna. W oknie zobaczysz, co Strata właśnie robi.

**Przy kolejnym uruchomieniu** ponownie otwórz `START-HERE.bat` (lub `./setup.sh`). Program wystartuje od razu,
bez ponownego pobierania plików. Zamknij okno, aby zatrzymać model. `UPDATE.bat` (`./update.sh`) aktualizuje Strata
bez uruchamiania. Aktualizacje, Docker, wiele kart, lokalizacje plików i wszystkie opcje:
[docs/INSTALL.md](docs/INSTALL.md).

**Jak szybko działa na twoim komputerze?** Uruchom `BENCH.bat` (`./bench.sh`): w około 5–10 minut zmierzy wydajność czatu, poleceń 4K/16K i sesji zbliżonej do pracy agenta przy twoich ustawieniach oraz zapisze raport, którym możesz się podzielić ([jak](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)).

## Który model wybrać?

Instalator zaleci model odpowiedni do ilości RAM. Ten sam model jest dostępny w kilku wariantach o różnym
stopniu kompresji. Mniejsze warianty są szybsze. Większe są nieco inteligentniejsze.

| Twój RAM | Wybierz | Dlaczego |
| --- | --- | --- |
| **32 GB** | **Coder** | mieści się w 32 GB i jest przeznaczony do kodu (z kartą 24 GB działają też Q2_0 i IQ2_XS) |
| **48 GB** | **IQ2_XS** (lub Q2_0, najszybszy) | większe warianty się nie zmieszczą |
| **64 GB** | **IQ2_XS** (zalecany) lub IQ3_XXS / IQ3_S | mieszczą się wszystkie warianty; IQ3_S jest najlepszy i najwolniejszy |
| **96 GB lub więcej** | **IQ3_S** lub UD-IQ4_XS od Unsloth (~4 bity) | miejsce na największe warianty, nawet gdy inne programy są otwarte |

- **[Coder](docs/MODELS.md#coder):** wersja do programowania, z której usunięto połowę ekspertów. Osiąga
  91% wyniku pełnego modelu w SWE-bench Verified (pomiar autorów) i mieści się w 32 GB RAM.
  Poza kodem radzi sobie gorzej, także z tekstem chińskim i innymi tekstami CJK (#438). Do tego wybierz Q2_0, IQ2_XS
  lub IQ3_S, które zachowują wszystkich ekspertów.
- **[Swift 1.5](docs/MODELS.md#swift-15):** model dostrojony tak, by znacznie krócej rozumował przed odpowiedzią.
  Otrzymasz odpowiedź wcześniej, przy zbliżonej jakości.
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs):** około 4-bitowa wersja od Unsloth, o jakości
  między IQ3_S a UD-Q4_K_XL. Do pobrania 94 GB. Przy mniej niż ~80 GB RAM Strata podczas generowania odpowiedzi
  odczytuje część modelu z SSD, więc działa wolniej (pomaga dysk NVMe).
- **[Unsloth UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL i Q8_0](docs/MODELS.md#unsloths-files-ud-iq4_xs-to-q8_0):** od 4 do 8 bitów, bliżej pełnego modelu
  (ten sam kolejny token co w Q8_0 w 93–95% przypadków). Szybkie, gdy wszyscy eksperci mieszczą się w RAM (~88 GB dla UD-Q4_K_XL
  do ~136 GB dla Q8_0); przy mniejszej ilości RAM Strata odczytuje część modelu z SSD podczas generowania odpowiedzi,
  co jest znacznie wolniejsze (7–8,5 tokena/s dla UD-Q4_K_XL na komputerze z 64 GB).
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs):** wymaga ręcznej
  konfiguracji. Nie ma go w menu instalatora.

Warianty, pliki do pobrania i wymagania pamięci: [docs/MODELS.md](docs/MODELS.md). Aby później dodać inny model,
uruchom `SETUP.bat` (Linux: `./setup.sh --setup`).

## Jak korzystać

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="Karta Monitor aplikacji Strata obok agenta programistycznego"><br>
<sub><b>Monitor</b> aplikacji Strata (po lewej), gdy agent programistyczny pisze kod ogrodu z pagodą z filmu (po prawej)</sub></p>

- **W przeglądarce:** otwórz `http://127.0.0.1:8080`. Znajdziesz tam **Chat**, **Monitor** pokazujący na bieżąco
  stan modelu oraz GPU/CPU/RAM, a także **About** z ustawieniami i adresami.
- **Twoje aplikacje i agenci programistyczni:** dodaj dostawcę „zgodnego z OpenAI” z bazowym adresem URL
  **`http://127.0.0.1:8080/v1`**. Działa dowolny klucz API i dowolna nazwa modelu.
  - Aplikacje korzystające z API Anthropic: `http://127.0.0.1:8080/v1/messages` (Claude Code:
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).
  - Codex CLI i inne aplikacje korzystające z OpenAI Responses API: `/v1/responses`
    ([konfiguracja](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **Rozumowanie:** wybierz **wyłączone, niskie, średnie lub wysokie** w menu czatu albo w ustawieniu „Reasoning Effort”
  w swojej aplikacji. Wyłączone jest najszybsze. Wysokie sprawdza się najlepiej przy trudnych pytaniach.
- **Obrazy:** podczas konfiguracji odpowiedz Tak na „Images?”. Następnie kliknij **Picture** w czacie lub załącz
  obrazy w swojej aplikacji. Karty AMD pod Linuksem odczytują obrazy za pośrednictwem procesora; pod Windowsem ta funkcja jeszcze nie działa.
- **Z telefonu lub innego komputera:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Zawsze ustawiaj
  klucz.
- **Jedno żądanie naraz:** domyślnie Strata odpowiada na jedno żądanie, a pozostałe czekają. Aby obsługiwać kilka
  jednocześnie, ustaw `"parallel": 2` ([BATCHING.md](docs/BATCHING.md)). Na karcie 12 GB każda odpowiedź będzie
  przez to wolniejsza.
- **Długie polecenia:** Strata odczytuje całą pierwszą wiadomość rozmowy, około 1 minuty na 30 000 tokenów.
  Przy kolejnych wiadomościach zaczyna odpowiadać w ciągu kilku sekund.

Więcej: [gdzie zapisywane są rozmowy](docs/INSTALL.md#where-things-are-stored), [API](docs/DETAILS.md#using-it).

### Opcjonalnie: większa szybkość na jednej karcie, gdy cały model mieści się w RAM

Dwa ustawienia włączane opcjonalnie (autor: RubenTorney, #1904 i #1905). Pomagają, gdy korzystasz z jednej karty graficznej, a RAM mieści wszystkich ekspertów (tryb arena w Strata, na przykład Q2_0 lub IQ3_XXS na komputerze z 64 GB). Otwórz konfigurację `strata-<model>.json` w folderze danych Strata ([lokalizacja](docs/INSTALL.md#where-things-are-stored)) i dodaj:

- **Szybsze generowanie:** `"--adapt-async", "1"` do listy `"args"`. Pamięć podręczna ekspertów będzie wtedy aktualizowana w pomocniczym wątku, bez wstrzymywania odpowiedzi. Pomiar na RTX 5070 (Windows, Q2_0): +4–5% szybkości generowania, +7–10% przy kodzie i długich dokumentach. Odpowiedzi mogą nieznacznie różnić się od domyślnych (pozostają takie same między uruchomieniami).
- **Szybszy odczyt poleceń pod Windowsem:** `"STRATA_PIN_BY_NEED": "1"` w bloku `"env"`. Windows pozwala zablokować na potrzeby karty graficznej tylko około połowy RAM; to ustawienie w pierwszej kolejności blokuje pamięć ekspertów, których karta faktycznie potrzebuje. Pomiar na RTX 5070: +4,4% szybkości odczytu poleceń.

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

Następnie uruchom Strata ponownie (`START-HERE.bat` lub `./setup.sh`). Aby cofnąć zmianę, usuń oba wpisy.

## Coś poszło nie tak?

- **Komputer zawiesił się przy pierwszym uruchomieniu Strata.** To normalne podczas ładowania modelu. Poczekaj
  i nie zamykaj okna. Nadal nie odpowiada po 10 minutach? Uruchom komputer ponownie, zamknij inne programy
  i spróbuj jeszcze raz lub wybierz mniejszy wariant.
- **Pobieranie lub instalacja zostały przerwane.** Uruchom ponownie `START-HERE.bat` (lub `./setup.sh`).
  Program wznowi pracę od miejsca, w którym skończył.
- **Działa bardzo wolno, a kontrolka dysku ciągle miga albo pojawia się komunikat „the engine stopped
  unexpectedly”.** Komputer ma za mało wolnego RAM. Zamknij inne programy (przeglądarki zużywają dużo pamięci)
  lub wybierz mniejszy wariant (Q2_0 lub IQ2_XS).
- **Pojawia się komunikat, że port 8080 jest już zajęty.** Strata już działa. Poszukaj jego okna.

Więcej problemów i rozwiązań: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Nadal potrzebujesz pomocy?
Otwórz [zgłoszenie](https://github.com/Niko1221/Strata/issues) i dołącz `strata-<model>.log` z folderu Strata.
Udało ci się znaleźć problem z bezpieczeństwem? Zgłoś go poufnie: [SECURITY.md](SECURITY.md).

## Jak to działa?

Takie modele zwykle działają na serwerach z setkami gigabajtów pamięci graficznej. Twoja karta ma
12–24 GB. Strata mimo to mieści model, **rozdzielając pracę na cały komputer**. Wyobraź sobie kuchnię:
to, czego stale potrzebujesz, zostaje na blacie, a reszta czeka w spiżarni.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="24 576 ekspertów modelu: najczęściej używani na karcie graficznej, wszyscy w RAM, tablica wyszukiwania na SSD"></p>

- **Model to zespół 24 576 małych specjalistów („ekspertów”).** Każde słowo potrzebuje tylko 10 z nich.
- **Karta graficzna** przechowuje kilka tysięcy najczęściej używanych ekspertów. **RAM** przechowuje wszystkich,
  a **procesor** równolegle wykonuje pozostałe obliczenia. **SSD** przechowuje dużą tablicę wyszukiwania.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="Mały model pomocniczy przewiduje kolejne słowa; duży model sprawdza je wszystkie jednocześnie i zachowuje poprawne"></p>

- **Najpierw przewidywanie, potem sprawdzanie:** mały model pomocniczy przewiduje kilka kolejnych słów.
  Duży model sprawdza je wszystkie jednocześnie. Otrzymujesz tę samą odpowiedź, tylko 1,6–1,8 raza szybciej.
- **Długie teksty są odczytywane dużymi fragmentami** (do 8 192 tokenów naraz), z szybkością ponad 1 000 tokenów
  na sekundę.

Dokładniejsze wyjaśnienie: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Każdy element i jego wyniki:
[szczegóły](docs/DETAILS.md#how-it-works) oraz [publikacja](docs/paper/Strata-Paper.pdf).

## Podziękowania i licencja

Model to [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) zespołu Qwen. Skompresowali
go [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5)
i Unsloth. Strata wykorzystuje części [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Pełne podziękowania:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata ma otwarty kod źródłowy na [licencji MIT](LICENSE).
Niektóre części i wszystkie modele mają własne licencje ([jakie](docs/HOW_IT_WORKS.md#license)).

## Wesprzyj Strata

Strata jest bezpłatna i ma otwarty kod źródłowy. Jeśli ci się przydaje, możesz wesprzeć rozwój:

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Postaw mi kawę" height="50"></a></p>
