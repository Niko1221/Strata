# Hakua / Strata

[English](README.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Polski](README.pl.md) · [Português](README.pt-BR.md) · [简体中文](README.zh-CN.md)

Lokalna sztuczna inteligencja dla Windows, z czatem, pracą agentów, badaniami i wyborem kompatybilnych modeli. Hakua to fork zapabob projektu Niko1221/Strata, utrzymywany niezależnie.

<!-- upstream v0.2.19 -->
## Co zmieniło się w 0.7.5

Uwaga: wartości DS4 2,84x / FLA 4,86x / Sakura 22,44 t/s pochodzą z dzienników rozwoju Antigravity i nie zostały niezależnie odtworzone w tej sesji budowy. Ta sesja zmierzyła 9,09 t/s prefill / 11,78 t/s decode na kanonicznym GGUF przy 32K.

To wydanie zachowuje wszystkie funkcje forkowe i dodaje natywne optymalizacje z 22/23-11. Produkt to **0.7.5**, rodzina silników natywnych **0.6.7**, źródło upstream **Niko1221/Strata v0.2.19** (Control/engines 0.1.65). Optymalizacje RTX 5060 Ti (sm_120a) / CUDA 13.5: DS4-9B prefill 2,84x szybciej, decode +13,4%, E2E +22,7%; FLA GDN + Spin(8) exp2f prefill 4,86x i kompresja KV 59,4%; branchless router Inkling-2 (1,16x); Sakura Inkling-2-Mini 65K–98K (`--expert-vram 7.8` daje decode 22,44 t/s, 12,7x, 32/32 identyczne tokeny). Warstwa serve czyta tokenizator i szablon czatu osadzone w GGUF oraz udostępnia parser wyjścia Inkling; Hermes Agent ma providera `inkling2`. Tablice profili SM × VRAM Ryu (kuraneko1, MIT) zostały włączone do oficjalnego CUDA Inkling-2 i odesłane upstream jako PR #6571. Weryfikacja podpisu usunięta (brak klucza); hashe SHA-256 pozostają. `releaseQualified=false`.

## Wersja i pobieranie

Bieżące wydanie: [0.7.5](https://github.com/zapabob/Strata/releases/tag/v0.7.5) (Aplikacja `0.7.5`; Terminal, Control `0.6.3`; rodzina silnika natywnego `0.6.7`) — GitHub Latest. Instalator Windows zawiera prywatny Node/Python/Control; silniki GPU są osobnymi komponentami opt-in na stronie wydania: przebudowany Q38 (27B + 9B) oraz oficjalny Hy4-Flash `strata-hy4` 0.1.64. `strata-inkling2` (Inkling-2 Mini, arch `inkling2`) przeniesiony z kuraneko1/Strata (MIT) przez kuraneko1 (_ryu15_) z podaniem źródła. `releaseQualified=false`.

[0.7.5](https://github.com/zapabob/Strata/releases/tag/v0.7.5) · [setup EXE](https://github.com/zapabob/Strata/releases/download/v0.7.5/Strata-0.7.5-windows-x64-setup.exe) · [tar.7z](https://github.com/zapabob/Strata/releases/download/v0.7.5/Strata-0.7.5-windows-x64.tar.7z)

## Zachowane funkcje forkowe

Q38 27B/9B dense, DS4/DeepSeek, RAM32, Hakua UI/pamięć, workflow skills, 8 języków README. Szczegóły: [README.md](README.md).
