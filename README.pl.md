# Hakua / Strata

[English](README.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Polski](README.pl.md) · [Português](README.pt-BR.md) · [简体中文](README.zh-CN.md)

Lokalne AI dla systemu Windows z czatem, pracą agentów, wyszukiwaniem źródeł i wyborem kompatybilnych modeli. Hakua to fork projektu Niko1221/Strata rozwijany niezależnie przez zapabob i nie jest oficjalnie zatwierdzoną dystrybucją upstream.

## Co si? zmieni?o w 0.6.9

Ta wersja przenosi prace Q38 nad szybkoµcią, formatami i wektorami sterowania na nowy układ `terminal/` dla oficjalnego Strata 0.2.16. Qwen3.8-27B używa teraz połączonej uwagi GQA: test A/B w odwrotnej kolejnoµci zmierzył **dekodowanie o 6,0 % szybsze przy identycznych tokenach wyjµciowych** (baza 555,8 ms -> GQA 522,6 ms, mediany, monit 10 tokenów, ctx 512, KV F16). Ścieżka 9B dodaje pogrupowaną uwagę FP16 ze wspólnym cache softmax oraz usunięcie nadmiarowych memset/RoPE (5,4 % szybciej, tokeny identyczne bitowo). Obsługa formatów FP4/FP8-NVFP4 i wektory sterowania są włączone. Wczeµniej: pogrupowane zapisy KV F16 zmniejszyły koszt KV prefill o 27 % przy bajtowo identycznych logitach, dekodowanie jest na poziomie Prism llama.cpp (oba DP4A), a 32-gigabajtowy squeeze RAM z ostrzeżeniem o limicie commit z Sync-Strata. Zbudowano i przetestowano z CUDA 13.5 / SM120. Pełne okno generowania, semantyka rzadkiej uwagi QSA i ogólna jakoµć modelu pozostają niezweryfikowane (`releaseQualified=false`).

## Wersja i pobieranie

<!-- release-062-stable -->
Bieżące wydanie aplikacji: [0.6.9](https://github.com/zapabob/Strata/releases/tag/v0.6.9) (Aplikacja `0.6.9`; Terminal, Control i silnik natywny `0.6.3`). To wydanie aplikacji dla Windows celowo nie jest GitHub Latest; Latest pozostaje `0.6.3`. Silnik GPU pozostaje osobno podpisanim komponentem [`v0.6.4-preview.1`](https://github.com/zapabob/Strata/releases/tag/v0.6.4-preview.1). Status `releaseQualified=false` pozostaje jawny; pełne okno kontekstu i ogólna jakość modeli nie są certyfikowane.

[0.7.2](https://github.com/zapabab/Strata/releases/tag/v0.7.2) · [EXE instalacyjny](https://github.com/zapabob/Strata/releases/download/v0.7.2/Strata-0.7.2-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.7.2/Strata-0.7.2-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.7.2/SHA256SUMS.txt) · [BUNDLE-SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.7.2/BUNDLE-SHA256SUMS.txt)

![Hakua / Strata real-model Japanese chat on Qwen3.8-27B](docs/media/native-qwen38-japanese-chat.png)
*Zainstalowana aplikacja Hakua / Strata uruchamiająca `qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf` przez `strata-q38.exe` (CUDA 13 / RTX 5060 Ti 16 GB).*

## Zalecany GGUF dla prototypu Qwen3.8

Do lokalnego czatu tekstowego na RTX 5060 Ti z 16 GB VRAM polecamy przetestowany [GGUF o mieszanej kwantyzacji 3,69 bpw od soyaakinohara](https://huggingface.co/soyaakinohara/qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf). Dziękujemy **soyaakinohara** za kwantyzację i publikację, a **AEON-7** za wagi źródłowe. Model należy pobrać osobno. Zacznij od kontekstu 4096 tokenów; nazwa pliku nie gwarantuje działania z 12 GB VRAM.

[Podpisana wersja demonstracyjna natywnego silnika CUDA 13.5](https://github.com/zapabob/Strata/releases/tag/v0.6.4-preview.1) przeszła w aplikacji 0.6.3 dwie tury po japońsku, zatrzymanie odpowiedzi i zwolnienie modelu z pamięci. W dwóch próbach pierwszego żądania czas krótkiego prefill F16 spadł o 35–43% względem dotychczasowej ścieżki natywnej. Prism (llama.cpp) nadal dekodował szybciej. [Konfiguracja, klucz publiczny i ograniczenia](docs/Q38_NATIVE_PREFILL_PREVIEW.md).

## Instalacja i aktualizacja

Użyj instalatora NSIS Windows x64 dla tego forka. Instaluje się na użytkownika i zawiera prywatne Node, Python oraz Control; systemowe narzędzia deweloperskie nie są wymagane. Natywny silnik CUDA jest osobnym komponentem. Archiwum `tar.7z` zawiera instalator, pakiet silnika CUDA, pakiet Control, instrukcje i sumy kontrolne, bez wag modeli i danych osobowych.

Zakończ bieżące żądania, odładuj modele, wykonaj kopię zapasową danych i zamknij aplikację wraz z ikoną w zasobniku. Uruchom instalator w istniejącym folderze instalacji, aby nadpisać pliki z zachowaniem modeli, czatów, pamięci, narzędzi, kluczy i ustawień uprawnień. Zaimportuj skompresowany komponent CUDA przez opcję *Mam już plik*, wybierz zgodny model lub lokalny plik GGUF i załaduj go. Zobacz [instalacja i wycofanie zmian](docs/HAKUA_INSTALL.md).

## Nowości w 0.6.2 i funkcje forka

Wersja 0.6.2 rozbudowuje [lokalny runner GGUF dla rodziny Qwen3.8-27B (`strata-q38.exe`)](docs/QWEN38_DENSE_NATIVE.md) na Windows CUDA 13 / Blackwell (SM120):
- **Polimorfizm kwantyzacji na poziomie tensora (`StrataQuantTensor`)**: obsługuje 11 typów GGML/Strata (`F32`, `F16`, `BF16`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ3_S`, `IQ2_S`, `IQ4_XS` oraz PrismML `PQ2_0=142`) osobno dla każdego tensora.
- **Wsadowy prefill Tensor Core + wielotokenowy GDN + połączona uwaga Spin8 + dekodowanie CUDA Graph (CUDA 13.5 SM120)**:
  - Prefill 140 tokenów (`--kv spin8_k8v4 --ctx-size 4096`): z `36 760,7 ms` (`3,81 tok/s`, `--conservative-sync`) do **`1 558,3 ms` (`89,8 tok/s` na ciepło, przyspieszenie 23,6x)**.
  - Stabilne dekodowanie (`GEN 32`, 31 kroków, pakowane GEMV + CUDA Graph z barierą co 8 warstw): z `206,4 ms/krok` do **`85,13 ms/krok` (`11,75 kroków/s`, przyspieszenie 2,42x)**.
  - **Pomiary długiego kontekstu (`--kv spin8_k8v4`, RTX 5060 Ti 16 GB)**: **65K (`65 536`)** zaliczone (prefill 1 024 tokenów `10 813,5–11 220,7 ms`, `91,3–94,7 tok/s`; dekodowanie `670,4 ms` / 7 kroków); **96K (`98 304`)** zaliczone (`required_with_reserve=15,09 GB`, prefill 140 tokenów `1 968,7 ms`); **131K (`131 072`)** zablokowane przez rezerwę 768 MiB GPU (`required=15,97 GB > available=15,91 GB`).
- **Rzeczywista weryfikacja szkiców MTP `K=3` (`blk.64.nextn.*`)**: obsługuje `--spec-draft-n-max 0..3` z `KvCache::rewind` i przywracaniem stanu GDN.
- **Pamięć podręczna KV i profil japoński**: tryby `F16`, `Spin8 K8V4` i `F16-K / Turbo2/3/4-V` (kolejność `auto`: `f16` → `f16_turbo4` → `spin8_k8v4`) oraz opcjonalny profil wyjścia japońskiego.

`/btw` uruchamia podagenta bez narzędzi do pytań pobocznych z własnym przyciskiem zatrzymania. `/research` i `/deepresearch` zbierają źródła przez wbudowany RustXNG Full (zapytania trafiają do włączonych dostawców zewnętrznych). Nauka i automatyczne stosowanie umiejętności workflow są osobnymi opcjami, domyślnie wyłączonymi w nowych profilach. Zobacz [umiejętności](docs/WORKFLOW_SKILLS.md) oraz [wyszukiwanie](docs/BROWSER_SEARCH.md).

## Integracja z upstreamem i zgodność

Przypięte źródło oficjalne to [c775728069](https://github.com/Niko1221/Strata/commit/c77572806996349b49995e4f98d41b5294d36a21) (odpowiednik 0.2.15 / silnik 0.1.60). Wersje forka są niezależne. Zgodność z DeepSeek jest utrzymywana jako rozszerzenie i blokuje ładowanie bez dedykowanego środowiska uruchomieniowego. Ten pakiet Windows nie oznacza natywnego wejścia obrazu, dystrybuowanego natywnego silnika DeepSeek ani certyfikowanej inferencji na Linux/Mac.

## Ochrona i zmierzone limity

Zachowaj minimalne progi: 3 GiB wolnej pamięci RAM, 4 GiB zapasu commit, co najmniej 768 MiB rezerwy GPU oraz limity miejsca na dysku. 32 GB RAM i 16 GB VRAM nie gwarantują dopasowania każdego modelu lub kontekstu. `releaseQualified=false` pozostaje w mocy, a instalator nie ma podpisu Windows Authenticode. Historyczne pomiary znajdują się w [dowodach wydań](docs/HAKUA_RELEASE.md).

## Dokumentacja i rozwój

[Instalacja](docs/HAKUA_INSTALL.md) · [Dowody wydań](docs/HAKUA_RELEASE.md) · [API i sprzęt](docs/AI_REFERENCE.md) · [Terminal](docs/TERMINAL.md) · [Budowanie ze źródeł](docs/SOURCE_INSTALL.md) · [Pakowanie](packaging/README.md) · [Wkład](CONTRIBUTING.md) · [Bezpieczeństwo](SECURITY.md) · [Licencja](LICENSE)

```sh
python -m pytest tests/terminal
```

```sh
cd apps/host
pnpm test
pnpm run build
```

### Niko1221/Strata — oficjalne odniesienie Terminal

```sh
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

[Niko1221/Strata](https://github.com/Niko1221/Strata) · [Hermes Agent](https://github.com/NousResearch/hermes-agent) · [Źródła i noty stron trzecich](third_party/)

## Zmiany w wersji 0.6.3

- Natywne prefill Q38 wykorzystuje GEMM Tensor Core cuBLAS z aktywacjami FP16 skalowanymi dla każdego tokena oraz 20 MiB trwałym blokiem wag w L2. Dekodowanie używa wyrównanego GEMV dp4a Q8_1 zależnego od formatu oraz CUDA Graph z LayerFence(8).
- Przyjmowanie długiego kontekstu wybiera rozmiar fragmentu prefill na podstawie zmierzonej drabiny VRAM. Rezerwa GPU 768 MiB oraz progi 3 GiB RAM i 4 GiB commit pozostają niezmienione.
- Fixtures numeryczne normalizują syntetyczne spakowane wagi do realistycznej amplitudy bez osłabiania ścisłych tolerancji `_exact_q8`. Niewyrównany GEMV przełącza się na jądro skalarne ładujące bajty.
- Weryfikacja: 17/17 testów natywnych przechodzi, a compute-sanitizer raportuje 0 błędów i 0 zagrożeń dla memcheck/synccheck/racecheck. Ogólna jakość modelu, inferencja w pełnym oknie i współbieżność GPU pozostają niekwalifikowane (`releaseQualified=false`).
- Podstawa badawcza: informacje o wydaniu NVIDIA CUDA 13.5 (poprawka algorytmu 66 cuBLASLt przy Tensor Memory) oraz raport architektury Qwen3.8-Next (hybryda GDN/uwaga globalna 3:1, QSA z kompresją r=4 i budżetem K=2048, gated residual, osadzanie n-gramów wczytywane z hosta).
