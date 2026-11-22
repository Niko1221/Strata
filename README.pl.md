## Co si? zmieni?o w 0.7.3

To wydanie przejmuje wszystkie zmiany oficjalnego upstreamu **Niko1221/Strata v0.2.18**, zachowując wszystkie funkcje forkowe. Natywny silnik CUDA to teraz **rodzina 0.6.6** z dwoma opcjonalnymi komponentami, a wersja produktu stosuje wersjonowanie semantyczne 0.7.3. Zawiera obsługę Hy4-Flash 220B MoE (`strata-hy4` 0.1.64, Tree-8 CUDA Graphs, VNNI 1,5x), oficjalne API/ustawienia (Control i silnik 0.1.64) oraz poprawki upstreamu. Przebudowany silnik Q38 obsługuje pliki GGUF 27B (64 warstwy) i 9B (32 warstwy) z wektorami sterowania; oficjalny silnik Hy4-Flash jest rozpowszechniany jako osobny opcjonalny komponent. Decyzją opiekuna usunięto weryfikację podpisu ed25519 (brak klucza podpisu); hashe SHA-256 pozostają kontrolą integralności. Wszystkie 16 punktów kotwicznych forku (Q38, DS4, RAM32, Hakua, 8 języków) są nienaruszone, a optymalizacja RAM 32 GB została zachowana. Pełnookienkowa inferencja i ogólna jakość modelu pozostają niekwalifikowane (`releaseQualified=false`). Trzeci opcjonalny silnik, `strata-inkling2` (Inkling-2 Mini, arch GGUF `inkling2`), zostal przeniesiony z kuraneko1/Strata (MIT) autorstwa kuraneko1 (_ryu15_) z podaniem zrodla.

## Wersja i pobieranie

<!-- release-062-stable -->
Bieżące wydanie aplikacji: [0.7.4](https://github.com/zapabob/Strata/releases/tag/v0.7.4) (Aplikacja `0.7.3`; Terminal, Control `0.6.3`; rodzina silnika natywnego `0.6.6`) — to jest GitHub Latest. Aplikacja Windows dostarcza rodzinę silników 0.6.6 (przebudowany Q38 27B+9B, oficjalny Hy4-Flash 0.1.64) jako osobne opcjonalne komponenty. Weryfikacja podpisu została usunięta (brak klucza); hashe SHA-256 pozostają kontrolą integralności. `releaseQualified=false` pozostaje; pełnookienkowa inferencja i jakość modelu nie są zakwalifikowane. Trzeci opcjonalny silnik, `strata-inkling2` (Inkling-2 Mini, arch GGUF `inkling2`), zostal przeniesiony z kuraneko1/Strata (MIT) autorstwa kuraneko1 (_ryu15_) z podaniem zrodla.

[0.7.4](https://github.com/zapabab/Strata/releases/tag/v0.7.4) · [EXE instalacyjny](https://github.com/zapabob/Strata/releases/download/v0.7.4/Strata-0.7.4-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.7.4/Strata-0.7.4-windows-x64.tar.7z) · [SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.7.4/SHA256SUMS.txt) · [BUNDLE-SHA256SUMS](https://github.com/zapabob/Strata/releases/download/v0.7.4/BUNDLE-SHA256SUMS.txt)

![Hakua / Strata real-model Japanese chat on Qwen3.8-27B](docs/media/native-qwen38-japanese-chat.png)
*Zainstalowana aplikacja Hakua / Strata uruchamiająca `qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf` przez `strata-q38.exe` (CUDA 13 / RTX 5060 Ti 16 GB).*

## Zalecany GGUF dla prototypu Qwen3.8

Do lokalnego czatu tekstowego na RTX 5060 Ti z 16 GB VRAM polecamy przetestowany [GGUF o mieszanej kwantyzacji 3,69 bpw od soyaakinohara](https://huggingface.co/soyaakinohara/qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf). Dziękujemy **soyaakinohara** za kwantyzację i publikację, a **AEON-7** za wagi źródłowe. Model należy pobrać osobno. Zacznij od kontekstu 4096 tokenów; nazwa pliku nie gwarantuje działania z 12 GB VRAM.

[Podpisana wersja demonstracyjna natywnego silnika CUDA 13.5](https://github.com/zapabob/Strata/releases/tag/v0.7.2) przeszła w aplikacji 0.6.3 dwie tury po japońsku, zatrzymanie odpowiedzi i zwolnienie modelu z pamięci. W dwóch próbach pierwszego żądania czas krótkiego prefill F16 spadł o 35–43% względem dotychczasowej ścieżki natywnej. Prism (llama.cpp) nadal dekodował szybciej. [Konfiguracja, klucz publiczny i ograniczenia](docs/Q38_NATIVE_PREFILL_PREVIEW.md).

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

Przypięte źródło oficjalne to [6e38f588](https://github.com/Niko1221/Strata/commit/6e38f588a646c0ee6948a74afbe670eb75896106) (odpowiednik 0.2.15 / silnik 0.1.60). Wersje forka są niezależne. Zgodność z DeepSeek jest utrzymywana jako rozszerzenie i blokuje ładowanie bez dedykowanego środowiska uruchomieniowego. Ten pakiet Windows nie oznacza natywnego wejścia obrazu, dystrybuowanego natywnego silnika DeepSeek ani certyfikowanej inferencji na Linux/Mac.

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