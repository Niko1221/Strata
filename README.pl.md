# Hakua / Strata

[English](README.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Polski](README.pl.md) · [Português](README.pt-BR.md) · [简体中文](README.zh-CN.md)

Lokalna AI w Windows: rozmowy, zadania agentów i badania ze źródłami. Hakua to niezależnie utrzymywany przez zapabob fork Niko1221/Strata, bez deklarowanego poparcia oryginalnego projektu.

## Wersja i pliki

<!-- release-060-candidate -->
Ten kod jest kandydatem integracji 0.6.0. Aplikacja, Terminal, Control i źródła silnika natywnego mają wersję 0.6.0. Weryfikacja kompilacji, instalacji i publikacji trwa; numer wersji nie dowodzi dostępności plików. Zachowano zweryfikowane [wydanie 0.5.9](https://github.com/zapabob/Strata/releases/tag/v0.5.9). Faktyczny stan 0.6.0 podaje zapis dostarczenia.

[0.6.0](https://github.com/zapabob/Strata/releases/tag/v0.6.0) · [Strata-0.6.0-windows-x64-setup.exe](https://github.com/zapabob/Strata/releases/download/v0.6.0/Strata-0.6.0-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.6.0/Strata-0.6.0-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.6.0/SHA256SUMS.txt)

## Instalacja i aktualizacja

Użyj instalatora NSIS Windows x64 tego forka. Instaluje się dla użytkownika i zawiera własne Node, Python i Control; narzędzia programistyczne nie są potrzebne. Podpisany silnik CUDA to osobny komponent. tar.7z ma zawierać ten sam instalator, komponenty, instrukcje i sumy kontrolne, bez wag modeli i danych osobistych. Rozpakuj najpierw 7z, potem tar, i sprawdź BUNDLE-SHA256SUMS.txt.

Zakończ żądania, zwolnij modele z pamięci, zrób kopię danych i zamknij aplikację również w zasobniku. Zainstaluj w istniejącym katalogu. Zachowaj modele, rozmowy, pamięć, narzędzia, klucze i zgody. Zaimportuj skompresowany silnik opcją posiadanego pliku, wybierz zgodny model lub lokalny GGUF i załaduj go. Nazwa modelu nie dowodzi zgodności z silnikiem. [Instalacja i powrót do poprzedniej wersji](docs/HAKUA_INSTALL.md).

## Zachowane funkcje

`/btw` uruchamia rzeczywistego podagenta bez narzędzi do pytania pobocznego, z osobnym zatrzymaniem. `/research` i `/deepresearch` gromadzą źródła. Wyszukiwanie może używać dołączonego RustXNG Full. Zapytania trafiają do włączonych dostawców zewnętrznych; wyszukiwanie nie jest całkowicie offline. Liczba źródeł nie gwarantuje poprawności.

Uczenie umiejętności i automatyczne stosowanie to osobne opcje, obie wyłączone w nowych profilach. Zachowane są ustawienia, prywatne edytowalne umiejętności, wersje i cofanie zmian. Pozostają operacje plikowe, ograniczony zakres historii i pamięci, zgodne ścieżki PrismML/TurboQuant, ograniczona pamięć ekspertów, integralność pakietów oraz diagnostyka. [Umiejętności](docs/WORKFLOW_SKILLS.md) · [Wyszukiwanie](docs/BROWSER_SEARCH.md).

## Integracja oficjalnego projektu

Punktem odniesienia jest oficjalny commit [09b6038869](https://github.com/Niko1221/Strata/commit/09b603886925994d0d4461857b01321ef3aa431c), generacja 0.2.14 / silnik 0.1.60: sieć, wybór modelu pomocniczego, jądra promptów, pamięć i poprawki platform. Numeracja forka jest niezależna. Usunięte ścieżki DeepSeek pozostają rozszerzeniem; brak dedykowanego środowiska ma blokować ładowanie. Pakiet Windows nie potwierdza natywnego wejścia obrazu, dystrybucji silnika DeepSeek ani sprawdzonej inferencji Linux/Mac.

## Ochrona i ograniczenia pomiarów

Zachowane minima to 3 GiB dostępnego RAM, 4 GiB zapasu pamięci zadeklarowanej, co najmniej 768 MiB rezerwy GPU i wolne miejsce na dyskach. Ochrona RAM w Windows obserwuje też ładowanie i generowanie. 32 GB RAM i 16 GB VRAM nie gwarantują dowolnego modelu lub kontekstu. Pełne okno, ogólna jakość modeli i współbieżność GPU pozostają niezakwalifikowane: releaseQualified=false. Instalator nie ma podpisu Windows Authenticode; podpisy komponentów i kanału aktualizacji są odrębne. [Dawne pomiary](docs/HAKUA_RELEASE.md) zachowują pierwotne wersje.

## Dokumentacja i rozwój

[Install](docs/HAKUA_INSTALL.md) · [Release evidence](docs/HAKUA_RELEASE.md) · [API and hardware](docs/AI_REFERENCE.md) · [Terminal](docs/TERMINAL.md) · [Source build](docs/SOURCE_INSTALL.md) · [Packaging](packaging/README.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [License](LICENSE)

```sh
python -m pytest tests/terminal
```

```sh
cd apps/host
npm test
npm run build
```

### Niko1221/Strata — upstream Terminal reference

```sh
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

[Niko1221/Strata](https://github.com/Niko1221/Strata) · [Hermes Agent](https://github.com/NousResearch/hermes-agent) · [Źródła i informacje stron trzecich](third_party/)
