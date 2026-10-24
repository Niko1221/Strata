<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="380">
  </picture>
</p>

# Strata po polsku

<p align="center"><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <b>Polski</b></p>

<h3 align="center">Uruchom nowoczesny model AI na swoim komputerze do gier</h3>

<p align="center">NVIDIA, AMD i Intel · Windows, Linux i Mac · bezpłatny i otwarty kod źródłowy</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?include_prereleases&label=release" alt="Najnowsze wydanie"></a> <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="Licencja MIT"></a> <a href="https://github.com/Niko1221/Strata/stargazers"><img src="https://img.shields.io/github/stars/Niko1221/Strata?style=flat" alt="Gwiazdy GitHuba"></a></p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="Ogród z wokselową pagodą napisany przez model Strata i działający w przeglądarce"></a><br><sub>Ogród z wokselową pagodą utworzony jednym poleceniem na RTX 5070 ze Strata (IQ3_S, kontekst 128K) · <a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">cały film (49 s)</a></sub></p>

Strata uruchamia **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** — model ze 125 miliardami parametrów,
który zwykle wymaga serwera — na zwykłym komputerze. Prowadzi rozmowy, pisze kod, odczytuje obrazy i współpracuje z twoimi
aplikacjami oraz agentami programistycznymi. Rozmowy i pliki pozostają na twoim komputerze.

**Zaczynasz?** Sprawdź [wymagania](#czego-potrzebujesz), a potem [zainstaluj aplikację](#opcja-a-aplikacja-strata).

- **Działa na komputerze do gier.** Na początek wystarczy łącznie 32 GB RAM i pamięci karty graficznej.
- **Współpracuje z twoimi narzędziami.** Obsługuje API OpenAI i Anthropic; podłączysz Claude Code, Codex CLI, OpenCode i inne agenty.
- **Odczytuje obrazy.** Ta funkcja jest domyślnie włączona.
- **Bezpłatny.** Licencja MIT, bez konta i bez chmury.
- **Prywatny.** Bez telemetrii. Informacje o pobieranych plikach, połączeniach z internetem i weryfikacji pobrań: [docs/SAFETY.md](docs/SAFETY.md).

Wersja 0.2.0-beta · silnik 0.1.45.3

## Czego potrzebujesz

**Łącznie potrzebujesz co najmniej 32 GB RAM i pamięci karty graficznej (VRAM).** Mac i Ryzen AI Max używają pamięci
współdzielonej. Więcej pamięci pozwala uruchomić większy lub szybszy model.

- Komputer z 32 GB pamięci i kartą 12–16 GB uruchomi wariant Coder.
- 64 GB RAM wystarczy dla każdego rozmiaru modelu.

| Komputer | Szczegóły |
| --- | --- |
| **NVIDIA** GeForce RTX serii 20, 30, 40 lub 50 (co najmniej 12 GB) | Zainstaluj aplikację lub Terminal. |
| **AMD** Radeon RX 6800 lub nowsza, Radeon AI PRO R9700 (co najmniej 12 GB) | [Karty AMD](docs/AMD_HIP.md) |
| **Intel** Arc | [Intel Arc](docs/INTEL_ARC.md), [silnik Intel](docs/INTEL.md) |
| **Mac z Apple Silicon** (M1 lub nowszy, macOS 15 lub nowszy) | [Mac](docs/MAC.md) |
| **AMD Ryzen AI Max (Strix Halo)** i inne systemy ze współdzieloną pamięcią | [Strix Halo](docs/STRIX_HALO.md) |
| Starsze karty i procesory | [Starsze GPU](docs/OLDER_GPUS.md), [starsze CPU](docs/INSTALL.md#older-cpus-experimental) |

Potrzebujesz też około 80 GB wolnego miejsca na dysku (SSD przyspiesza pierwsze uruchomienie), Windows 10 / 11, Linux
lub macOS oraz aktualnego sterownika karty graficznej. Dwie lub trzy karty tego samego producenta mogą współdzielić model
([wiele GPU](docs/MULTI_GPU.md)). Strata nie dzieli modelu między karty NVIDIA i AMD. Pełna lista: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Instalacja

Wybierz jedną metodę. Wszystkie uruchamiają ten sam silnik i używają tych samych modeli.

### Opcja A: aplikacja Strata

Najprostsza metoda: aplikacja zawiera silnik. Zainstaluj ją, wybierz model i rozpocznij rozmowę. Pobierz plik dla swojego
systemu i karty ze [strony wydań](https://github.com/Niko1221/Strata/releases/latest).

| System | Karta graficzna | Plik do pobrania |
| --- | --- | --- |
| **Windows 10 / 11** | NVIDIA RTX, sterownik 580 lub nowszy | [cuda13](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.0-beta-windows-cuda13.exe) |
| | NVIDIA, starsza karta lub sterownik starszy niż 580 | [cuda12](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.0-beta-windows-cuda12.exe) |
| | AMD Radeon RX serii 9000 lub Radeon AI PRO R9700 | [hip](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.0-beta-windows-hip.exe) |
| | AMD Radeon RX serii 7000 lub Ryzen AI Max (Strix Halo) | [hip-rdna3](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.0-beta-windows-hip-rdna3.exe) |
| **Linux** | NVIDIA RTX, sterownik 580 lub nowszy | [cuda13](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.0-beta-linux-nvidia-cuda13.AppImage) |
| | NVIDIA, starsza karta lub sterownik starszy niż 580 | [cuda12](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.0-beta-linux-nvidia-cuda12.AppImage) |
| | AMD Radeon RX serii 9000 lub Radeon AI PRO R9700 | [amd-rx9070](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.0-beta-linux-amd-rx9070.AppImage) |
| | AMD Radeon RX serii 7000 | [amd-rx7900](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.0-beta-linux-amd-rx7900.AppImage) |
| | AMD Ryzen AI Max (Strix Halo) | [amd-strixhalo](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.0-beta-linux-amd-strixhalo.AppImage) |
| **Mac** (Apple Silicon, macOS 15 lub nowszy) | wbudowana obsługa | [dmg](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.0-beta-mac.dmg) |

Dla NVIDIA wybierz cuda13 przy sterowniku 580 lub nowszym, w pozostałych przypadkach cuda12. Pliki hip i amd są przeznaczone
dla AMD. Dla innych kart AMD (RX 6000) nie ma jeszcze gotowej aplikacji: [karty AMD](docs/AMD_HIP.md).

- **Windows:** uruchom plik exe. Instaluje się dla bieżącego użytkownika i nie wymaga praw administratora. Jeśli Windows
  wyświetli ostrzeżenie, wybierz **Więcej informacji**, a potem **Uruchom mimo to** (instalator nie ma jeszcze podpisu kodu).
- **Linux:** nadaj plikowi prawa uruchamiania poleceniem chmod +x. Wymaga FUSE 2: sudo apt install libfuse2 (Ubuntu 22.04)
  albo libfuse2t64 (Ubuntu 24.04 i nowszy).
- **Mac:** otwórz dmg i przeciągnij Stratę do folderu Applications.

Aplikacja sprawdzi komputer i poleci model, który się zmieści. Model ma około 40–70 GB; po przerwaniu pobieranie jest wznawiane.
Zamknięcie okna zatrzymuje Stratę i silnik.

### Opcja B: Terminal

Terminal to polecenie strata, przeznaczone do skryptów, serwerów i agentów AI.

Linux i Mac:

~~~sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh | sh
~~~

Windows (PowerShell):

~~~powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
~~~

Instalator sprawdza komputer, pobiera Terminal, weryfikuje SHA-256 i rozpakowuje pliki w folderze domowym. Nie wymaga
praw administratora. W Windows otwórz potem nowe okno terminala. Uruchom:

~~~sh
strata launch
~~~

Program pyta kolejno o model, kontekst, obrazy i dostęp dla innych urządzeń. Pobiera podpisany silnik dla karty, sprawdza
jego uruchomienie, pobiera model i rozpoczyna serwowanie. W skryptach i agentach użyj:

~~~sh
strata launch --model q2_0 --yes --no-tui
~~~

Aby sprawdzić skrypt instalacyjny przed uruchomieniem, pobierz go przez curl -fsSL <adres> -o install.sh, przeczytaj,
a potem uruchom sh install.sh. Wszystkie ekrany i opcje: [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md).

### Opcja C: konfiguracja przez asystenta AI

Wklej poniższe polecenie do Claude Code, Cursor, Codex, GitHub Copilot lub innego asystenta:

~~~text
Skonfiguruj Strata na tym komputerze: https://github.com/Niko1221/Strata — postępuj zgodnie z docs/AI_SETUP.md w tym repozytorium.
~~~

Asystent sprawdzi kartę, RAM i dysk, dobierze model, zainstaluje go i uruchomi oraz wyjaśni, jak podłączyć aplikacje.
Narzędzia AI mogą też zarządzać Stratą przez [serwer MCP](docs/MCP_SERVER.md). Instrukcje budowania: [Budowanie ze źródeł](#budowanie-ze-źródeł).

## Modele

Aplikacja i strata launch polecają rozmiar dopasowany do dostępnej pamięci. Mniejsze warianty są szybsze, większe — nieco
inteligentniejsze.

### Mało pamięci? Wypróbuj Gyro

[Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) od agentionai to najmniejszy wariant modelu.

| Rozmiar | Potrzebna pamięć (RAM + VRAM) | Pobieranie | Opis |
| --- | ---: | ---: | --- |
| **Gyro-S** | 27,6 GB | 58,5 GB | Dla komputerów, na których nie mieści się nic innego. Jakość niższa niż IQ2_XS. |
| **Gyro-M** | 35 GB | 92,0 GB | Jakość zbliżona do IQ3_XXS przy ok. 20% mniejszym użyciu pamięci GPU. Odpowiada o 17–21% wolniej niż Q2_0 na karcie 12 GB i R9700. |

Gyro działa na kartach NVIDIA i AMD (AMD jest eksperymentalne), ale nie na Intel Arc ani Macu. Szczegóły i pomiary:
[docs/GYRO.md](docs/GYRO.md). Uruchom: strata launch --model gyro-s.

### Pozostałe rozmiary

| Pamięć (RAM + VRAM) | Wybierz | Uwagi |
| --- | --- | --- |
| **32 GB** | **Gyro-S** albo **[Coder](docs/MODELS.md#coder)** do kodu | Coder potrzebuje 32 GB RAM i zachowuje połowę ekspertów, więc poza kodowaniem jest słabszy. |
| **40 GB** | **[Swift 1.5](docs/MODELS.md#swift-15)** IQ2_XS (39,2 GB) albo **Gyro-M** (35 GB) | Swift krócej rozumuje przed odpowiedzią. Gyro-M ma jakość zbliżoną do IQ3_XXS, ale odpowiada o 17–21% wolniej niż Q2_0. |
| **48 GB** | **IQ3_XXS** (47,0 GB) | Wolniejszy, lepsza jakość. |
| **64 GB** | **IQ3_S** (54,8 GB), model oryginalny lub Swift 1.5 | Najwolniejszy, ale najlepszy: w opublikowanych testach dorównuje pełnemu modelowi. |
| **96 GB lub więcej** | **UD-IQ4_XS** od Unsloth (ok. 72 GB RAM) do **Q8_0** (ok. 136 GB), 4–8 bitów | Bliżej pełnego modelu. Przy mniejszej ilości RAM Strata odczytuje część danych z SSD, co mocno spowalnia działanie. |

Model może działać przy mniejszej ilości RAM: Strata sama przełącza się na [tryb małej ilości RAM](docs/MODELS.md#a-big-graphics-card-and-little-ram).
Będzie wolniej, szczególnie przy małej karcie graficznej.

Inne opcje:

- **[Swift 1.5](docs/MODELS.md#swift-15)** to model dostrojony tak, by krócej rozumował, przy zbliżonej jakości i wymaganiach pamięci.
- **[NVFP4](docs/NVFP4.md)** to 4-bitowy wariant NVIDIA. Potrzebuje ok. 80 GB RAM, by trzymać wszystkich ekspertów w pamięci.
- **[Qwen3.6-35B-A3B](docs/QWEN36.md)** to mniejszy model, który odczytuje obrazy i obsługuje kilka żądań naraz.
- **[Uncensored IQ3_XXS od OrcaRouter](docs/MODELS.md#orcarouter-uncensored-iq3_xxs)** wymaga ręcznej konfiguracji.

Wszystkie rozmiary i pliki do pobrania: [docs/MODELS.md](docs/MODELS.md).

## Jak szybko działa?

Pomiar na karcie 12 GB (RTX 5070, Ryzen 5 7600, 64 GB RAM). Wyniki dla innych kart: [szybkość modeli](docs/MODELS.md#how-fast-is-each-size),
[wyniki społeczności](docs/COMMUNITY_BENCHMARKS.md).

| Model | Generowanie odpowiedzi | Odczyt polecenia |
| --- | --- | --- |
| **Q2_0** | ████████████ 94 tokeny/s | ████████████ 2 650 tokenów/s |
| **IQ2_XS** | ██████████░░ 79 tokenów/s | █████████░░░ 2 090 tokenów/s |
| **IQ3_XXS** | ████████░░░░ 62 tokeny/s | ████████░░░░ 1 750 tokenów/s |
| **IQ3_S** | ███████░░░░░ 53 tokeny/s | ███████░░░░░ 1 620 tokenów/s |
| **Coder** | ███████░░░░░ 55 tokenów/s | ██████████░░ 2 180 tokenów/s |

Token to około ¾ słowa, więc 60 tokenów na sekundę to więcej, niż można przeczytać. „Odczyt polecenia” to szybkość
przetwarzania długiego dokumentu, kodu lub historii rozmowy.

Pomiar na karcie AMD 16 GB (RX 9070 XT, Ryzen 9 3900X, 47 GB RAM):

| Model | Generowanie odpowiedzi | Odczyt polecenia |
| --- | --- | --- |
| **Q2_0** | ████████░░░░ 60 tokenów/s | █████░░░░░░░ 1 160 tokenów/s |
| **IQ2_XS** | ███████░░░░░ 52 tokeny/s | █████░░░░░░░ 1 110 tokenów/s |
| **Coder** | ██████░░░░░░ 44 tokeny/s | ██████░░░░░░ 1 420 tokenów/s |

Obie tabele używają tej samej skali. RTX 3090 (24 GB) powinna generować około 100–140 tokenów/s. Wersję silnika dla
każdego pomiaru podano w [DETAILS.md](docs/DETAILS.md#speed-measured).

## Korzystanie ze Strata

### W aplikacji

Aplikacja ma **Czat**, **Monitor** modelu, GPU, CPU i RAM, stronę **Modele** oraz **Ustawienia**. Czat może pracować
w folderze na komputerze. W trybie **Manual** pyta przed zmianą pliku lub uruchomieniem polecenia. Wpisz
/loop 10m <zadanie>, aby powtarzać wiadomość co 10 minut, gdy czat jest otwarty.

Czat może przeszukiwać internet przez DuckDuckGo lub Bing, bez konta. Klucz wyszukiwania można dodać w **Ustawienia, Narzędzia**.
Jedyną funkcją wysyłającą coś poza komputer są wyszukiwane hasła. Aby uruchomić dwa modele, włącz **Uruchom wiele modeli**
i przydziel każdemu osobną kartę tego samego producenta.

<p align="center"><img src="docs/media/app-chat.png" width="900" alt="Aplikacja Strata z czatem agenta tworzącego ogród z pagodą i podglądem strony"></p>

### W Terminalu

<p align="center"><img src="docs/media/terminal-monitor.png" width="900" alt="Monitor Terminala Strata pokazujący szybkość, GPU, pamięć i ostatnie żądania"></p>

~~~sh
strata launch                                  # konfiguracja: model, kontekst, obrazy, dostęp, potem serwowanie
strata launch --model q2_0 --yes --no-tui     # bez pytań: uruchom serwer, wyświetl adres i zakończ
strata launch --plan --model q2_0              # pokaż plan; niczego nie zmieniaj
strata launch --status                         # co jest teraz serwowane
strata launch --stop                           # zatrzymaj
strata monitor --url http://127.0.0.1:8080     # obserwuj szybkość, GPU, pamięć i żądania
strata update --check                          # sprawdź, czy jest nowy silnik
strata doctor                                  # sprawdź pliki zainstalowanego silnika
~~~

strata --help pokazuje wszystkie opcje. Uruchomienie serwera niczego nie pobiera. Pobieranie odbywa się podczas instalacji,
strata launch (po wyborze modelu) i strata update. Aby połączyć się z telefonu lub innego komputera, użyj
strata launch --access lan. Polecenie tworzy klucz API i nie uruchomi serwera w sieci bez klucza. Przeglądarka otwiera się
tylko po dodaniu --open. Instrukcja: [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md), [docs/TERMINAL.md](docs/TERMINAL.md).

### Z agentami AI i innymi aplikacjami

Agent instaluje Stratę poleceniem z opcji B i uruchamia strata launch --model NAZWA --yes --no-tui.
Kroki i opcje: [docs/AI_SETUP.md](docs/AI_SETUP.md); zasady repozytorium: [AGENTS.md](AGENTS.md).

| Rodzaj aplikacji | Adres bazowy |
| --- | --- |
| Zgodne z OpenAI | http://127.0.0.1:8080/v1 (dowolny klucz API i nazwa modelu) |
| API Anthropic (Claude Code i inne) | http://127.0.0.1:8080, bez /v1. Aplikacje dodają /v1/messages. Claude Code: ANTHROPIC_BASE_URL=http://127.0.0.1:8080 |
| OpenAI Responses API (Codex CLI) | http://127.0.0.1:8080/v1/responses ([konfiguracja](docs/DETAILS.md#the-responses-api-and-codex-cli)) |

strata connect wyświetla ustawienia dla OpenCode, Codex CLI, Claude Code, oh-my-pi, pi, Hermes Agent i OpenClaw.
strata connect opencode --write zapisuje plik po uzyskaniu zgody i zachowuje kopię zapasową.

### Obrazy i pozostałe wskazówki

Obsługa obrazów jest domyślnie włączona w modelach, które potrafią je odczytywać. Załącz lub wklej obraz do czatu albo wyślij
go z aplikacji. Gdy karta ma mniej niż 24 GB, obraz analizuje procesor (kilka sekund, bez zajmowania pamięci karty);
przy 24 GB lub więcej robi to karta. Dotyczy to NVIDIA i AMD. Czytnik wybierzesz opcjami --vision off, --vision gpu
lub --vision cpu. Mac nie ma jeszcze czytnika obrazów.

Pomiary: na Radeon AI PRO R9700 czytnik karty analizował obraz o rozmiarze 260 tokenów w 57 ms, a procesorowy w 3,1 s.
Na kartach 16 GB (Tesla P100 i RTX A4000, Linux) czytnik karty obniżał szybkość tekstu o 3–5%, bo zajmował ok. 1,4 GB
pamięci potrzebnej modelowi.

- Poziom rozumowania (**wyłączone, niski, średni lub wysoki**) wybierzesz w menu czatu albo ustawieniu reasoning effort. Wyłączone jest najszybsze.
- Domyślnie Strata obsługuje jedno żądanie naraz. Użyj --parallel N albo opcji parallel w konfiguracji, by obsługiwać kilka ([BATCHING.md](docs/BATCHING.md)). Na karcie 12 GB odpowiedzi będą wolniejsze.
- Pierwszą wiadomość w czacie Strata odczytuje w całości; 30 000 tokenów zajmuje około minuty. Kolejne wiadomości przetwarza w kilka sekund.
- Lokalizacja zapisanych rozmów i API: [docs/INSTALL.md](docs/INSTALL.md#where-things-are-stored), [docs/DETAILS.md](docs/DETAILS.md#using-it).

## Budowanie ze źródeł

<details>
<summary>Windows, Linux i Mac z repozytorium</summary>

[Pobierz Stratę](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) i rozpakuj archiwum lub użyj git clone.
**Windows:** uruchom START-HERE.bat. **Linux:** uruchom ./setup.sh w folderze Strata.

**Mac (Apple Silicon, macOS 15 lub nowszy):** zainstaluj narzędzia wiersza poleceń Apple poleceniem xcode-select --install
(sam Xcode nie jest potrzebny). W folderze Strata uruchom ./setup.sh. Kompilacja silnika trwa około 7 minut; program zada
pytania, pobierze model i go uruchomi. Przy kolejnym uruchomieniu startuje od razu. Szczegóły: [docs/MAC.md](docs/MAC.md).

Instalator wykryje kartę i skonfiguruje silnik, a następnie zapyta o model, kontekst i obrazy. Naciśnij Enter, aby wybrać
zalecaną odpowiedź. Pobrany model ma około 70 GB. Jeśli pobieranie zostanie przerwane, uruchom skrypt ponownie. Przeglądarka
otworzy http://127.0.0.1:8080.

> Przy pierwszym uruchomieniu komputer może działać wolno lub przestać odpowiadać na 1–3 minuty. Strata ładuje 35–55 GB
> do RAM i blokuje część pamięci dla GPU. Poczekaj i nie zamykaj okna.

Później uruchom ponownie START-HERE.bat lub ./setup.sh. Zamknięcie okna zatrzymuje model. UPDATE.bat (./update.sh)
aktualizuje Stratę bez uruchamiania. Aktualizacje, Docker, wiele kart i lokalizacje plików: [docs/INSTALL.md](docs/INSTALL.md).

Pomiar wydajności: BENCH.bat (./bench.sh). Trwa 5–10 minut i zapisuje raport ([instrukcja](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)).

**Opcjonalnie: szybsze działanie na jednej karcie, gdy cały model mieści się w RAM.** Dwa ustawienia RubenaTorneya (#1904, #1905)
pomagają, gdy pamięć pomieści wszystkich ekspertów, np. Q2_0 lub IQ3_XXS przy 64 GB. W pliku strata-<model>.json
([lokalizacja](docs/INSTALL.md#where-things-are-stored)) można dodać:

- "--adapt-async", "1" do listy args. Pamięć podręczna ekspertów aktualizuje się w osobnym wątku bez zatrzymywania odpowiedzi.
  Pomiar na RTX 5070 (Windows, Q2_0): +4–5% generowania, +7–10% dla kodu i długich dokumentów. Wyniki mogą nieznacznie
  różnić się od domyślnych, ale są stałe między uruchomieniami.
- "STRATA_PIN_BY_NEED": "1" do bloku env w Windows. System blokuje dla GPU tylko około połowy RAM; ustawienie najpierw
  rezerwuje potrzebnych ekspertów. Pomiar na RTX 5070: +4,4% szybkości odczytu polecenia.

~~~json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
~~~

Po zmianie uruchom Stratę ponownie. Aby cofnąć ustawienia, usuń oba wpisy.

</details>

## Jak to działa?

Podobne modele zwykle działają na serwerach z setkami gigabajtów pamięci GPU. Twoja karta ma 12–24 GB. Strata rozdziela
pracę na cały komputer. Wyobraź sobie bibliotekę.

- Model to zespół 24 576 małych specjalistów („ekspertów”), jak 24 576 książek. Każde słowo potrzebuje tylko 10.
- **Karta graficzna to biurko.** Jest mała i szybka; leży na nim około 4 000 najczęściej używanych książek.
- **RAM to półki czytelni.** Jest na nich wszystkich 24 576 książek. Procesor odczytuje potrzebną książkę z półki, gdy nie ma jej na biurku.
- **SSD to archiwum.** Przechowuje tablicę wyszukiwania 29 GB; model odczytuje kilka jej wierszy na słowo.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="24 576 ekspertów modelu: najczęściej używani na karcie, wszyscy w RAM, tablica wyszukiwania na SSD"></p>

Mały model pomocniczy przewiduje następne słowa. Duży sprawdza je jednocześnie i zachowuje poprawne. Otrzymujesz tę samą
odpowiedź 1,6–1,8 raza szybciej. Długie teksty są odczytywane fragmentami do 8 192 tokenów, z szybkością ponad 1 000 tokenów/s.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="Mały model przewiduje kolejne słowa, duży sprawdza je naraz i zachowuje poprawne"></p>

Więcej: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md), [szczegóły](docs/DETAILS.md#how-it-works) i [publikacja](docs/paper/Strata-Paper.pdf).

## Coś poszło nie tak?

- **Komputer zawiesił się przy pierwszym uruchomieniu.** Program ładuje model do RAM. Poczekaj; jeśli po 10 minutach nadal nie odpowiada,
  uruchom komputer ponownie, zamknij inne programy albo wybierz mniejszy model.
- **Pobieranie lub instalacja zostały przerwane.** Uruchom ponownie aplikację, strata launch, START-HERE.bat lub ./setup.sh;
  program wznowi pracę.
- **Działa bardzo wolno, miga kontrolka dysku lub pojawia się „the engine stopped unexpectedly”.** Brakuje wolnego RAM.
  Zamknij programy (przeglądarki zużywają dużo pamięci) lub wybierz mniejszy model, np. Q2_0 albo IQ2_XS.
- **Brak silnika dla komputera.** W aplikacji wybierz Ustawienia, Silnik, Aktualizuj silnik. W Terminalu uruchom strata update.
  strata doctor pokazuje zainstalowane pliki.
- **Port 8080 jest zajęty.** Strata już działa. Poszukaj jej okna.

Rozwiązania: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Zgłoś problem przez [GitHub Issues](https://github.com/Niko1221/Strata/issues).
W aplikacji dołącz logi przez Ustawienia, Silnik, Diagnostyka. W Terminalu wklej wynik strata doctor. Przy instalacji ze źródeł
dołącz strata-<model>.log. Problem bezpieczeństwa zgłoś poufnie: [SECURITY.md](SECURITY.md).

## Podziękowania i licencja

Model [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) stworzył zespół Qwen. Skompresowali go
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5), agentionai (Gyro)
i Unsloth. Strata wykorzystuje części [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Pełne podziękowania:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata ma otwarty kod na [licencji MIT](LICENSE). Niektóre części
i każdy model mają własne licencje ([lista](docs/HOW_IT_WORKS.md#license)).

## Współtworzenie

Znalazłeś błąd, chcesz przesłać zmianę albo wyniki testów? Najpierw przeczytaj [CONTRIBUTING.md](CONTRIBUTING.md).
Znajdziesz tam informacje, co dołączyć, aby inni mogli sprawdzić twoją pracę.

## Wesprzyj Stratę

Strata jest bezpłatna i objęta licencją MIT. Tworzę ją i testuję na dostępnym mi sprzęcie. Jeśli dobrze działa na twoim
komputerze, kawa pomoże mi testować nowe karty i naprawiać zgłaszane problemy. Nie ma presji — samo używanie i zgłaszanie
uwag też pomaga.

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Postaw mi kawę" height="50"></a>

Chcesz wesprzeć Stratę jako sponsor? Napisz przez [GitHub](https://github.com/Niko1221).
