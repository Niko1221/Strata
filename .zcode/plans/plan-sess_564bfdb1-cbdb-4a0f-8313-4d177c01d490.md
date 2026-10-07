# RoPE scaling dla Strata — none / linear / YaRN (1:1 z llama.cpp)

## Stan obecny (ustalone eksploracją)

- theta=1e7 i n_rot=64 są **zahardkodowane**: `qsa_freq_base()` (`include/strata/kernels/qsa.hpp:121-122`), `qsa_real_shapes()` (qsa.hpp:109-117). Komentarze wprost mówią „no rope.scaling keys, so freq_scale = 1 and no YaRN". Żadnych flag `--rope-*` nie ma.
- RoPE istnieje w **5 miejscach**, w dwóch formach:
  - **Tabela cos/sin** (host, float64): `build_rope_table` (`src/kernels/cuda/rope.cu:34-45`), budowana raz w `qsa_state_init` (`src/core/layer.cpp:632-641`), wspólna dla 12 warstw QSA. Czytają ją kernele `rope_neox_kernel` (rope.cu:56-72) i `indexer_key_append` (`src/kernels/cuda/qsa.cu:155-238`).
  - **Analitycznie w kernelu**: `native_rope.cu:54,82` (`theta_scale` liczony na hoście), prefill `rope_kernel` + wrapper `rope()` (`src/prefill/kernels.cu:314-325, ~501`, hardcode `-2.0f/64.0f`), `native_qsa_indexer.cu:110-128`.
- **Call sites** przekazujące `qsa_freq_base()`: layer.cpp:637 (budowa tabeli), 818-819/872-873 (decode), verify.cpp:438,467-469 (okno verify), mtp.cpp:334 (drafter), prefill.cpp:964,984,987,~993-999 (prefill).
- **CUDA graphs**: argumenty per-token wyłącznie z pamięci urządzenia (`st.pos_dev`); stałe procesu mogą być zwykłymi argumentami kernela — są bake'owane przy capture i nigdy się nie zmieniają. To jest centralny niezmiennik.
- **Kluczowa przewaga architektury**: dla domyślnej ścieżki tabelarycznej skalowanie = **inna zawartość tabeli**. Kernele `rope_neox_kernel` i `indexer_key_append` nie zmieniają się wcale (mscale wchodzi przez przemnożenie cos/sin w tabeli).
- `--prompt-cache` trzyma checkpointy tylko w RAM procesu (generate.cpp:259-262) — brak ryzyka niespójności między uruchomieniami.

## Krok 1 — nowy nagłówek `include/strata/kernels/rope_scaling.hpp`

```
enum class RopeScalingType { None, Linear, YaRN };
struct RopeScaling {
    RopeScalingType type = RopeScalingType::None;
    double freq_base = 1e7;      // qwen4exp.rope.freq_base (wartość z qsa_freq_base())
    double factor = 1.0;         // --rope-scale; freq_scale = 1/factor
    double orig_ctx = 262144;    // --yarn-orig-ctx (natywny kontekst treningowy — potwierdzić w kroku 7)
    double ext_factor = 1.0;     // --yarn-ext-factor (0 wyłącza korekcję; none/linear ignorują)
    double attn_factor = 1.0;    // --yarn-attn-factor
    double beta_fast = 32.0;     // --yarn-beta-fast
    double beta_slow = 1.0;      // --yarn-beta-slow
    double freq_scale() const;   // 1 dla None; 1/factor dla Linear/YaRN
    double mscale() const;       // attn_factor dla None/Linear; attn_factor*(1+0.1*ln(factor)) dla YaRN (konwencja ggml)
    void corr_dims(int n_rot, double out[2]) const;  // ggml_rope_yarn_corr_dims
};
void rope_scaling_set(const RopeScaling&);   // wzorzec domu jak native_rope_set_enabled / mrope_table_set
const RopeScaling& rope_scaling();
```

Pomocnicze `rope_yarn_ramp` i `corr_dims` przepisane **z vendored `third_party/llama.cpp`** (ggml/src/ggml.c, ggml-cuda/rope.cu) z atrybucją MIT dokładnie jak w nagłówku `native_rope.cu:1-24` — nie z pamięci. Semantyka kierunkowa (niskie pary = wysokie częstotliwości ekstrapolują, wysokie pary interpolują) zostanie potwierdzona fixturą strukturalną (F5), nie wywodem.

## Krok 2 — skalowana tabela + fixtury (`rope.cu`, `rope_parity.cpp`)

- `build_rope_table` zyskuje parametr `const RopeScaling&`; float64 w tej samej kolejności co dziś: `inv → theta_extrap = p*inv → mix z theta_interp = freq_scale*theta_extrap wg ramp(pair) → cos/sin → *mscale` (mnożenie przed castem do float32). Dla `None` wartości **bitowo identyczne** z dzisiejszymi (patrz Krok 5).
- Wywołanie w layer.cpp:637 przekazuje `rope_scaling()`. Kernele tabelaryczne bez zmian.
- Nowe fixtury w `src/kernels/rope_parity.cpp` (styl domu: float64 referencja bit-exact po cast, tolerancja względna 1e-6 z mianownikiem = magnitude wiersza, check strukturalny, asercja „observability"):
  - **F1**: tabela `None` = referencja bit-exact (istniejący check zostaje).
  - **F2**: tabela `Linear(4)` vs float64 referencja `ang = p*inv/4` — bit-exact.
  - **F3**: równoważność: `Linear(4)` na pozycji p == `None` na p/4 (identyczne bity wiersza).
  - **F4**: tabela `YaRN` vs float64 referencja (ramp + corr_dims + mscale, transkrypcja z ggml z atrybucją) — bit-exact.
  - **F5** (strukturalny, jak check parowania NEOX): przy `YaRN(4)` na dalekiej pozycji najniższa para ma kąt == theta_extrap, najwyższa == theta_interp; dodatkowo `cos_tab[0] == mscale` (na pozycji 0 cos(0)=1, więc mscale jest bezpośrednio widoczny).
  - **F6**: rotacja kernela z przeskalowaną tabelą YaRN — istniejący wzorzec host-referencji.
  - **Observability**: asercja, że fixtura odróżnia scaled od unscaled (różnica kątów None vs Linear(2) na pozycji 262143 > próg) — konwencja z sampler_parity fixture 9, żeby test nie był pusty.

## Krok 3 — ścieżki analityczne i wszystkie call sites

Każdy wrapper analityczny zamiast samego `freq_base` dostaje rozdzielczone stałe procesu: `freq_base, freq_scale, corr_dims[2], ext_factor, mscale` (host → float; legalne w grafach, bo per-token pozycje nadal lecą z device):

- `native_rope.cu`: kernel `apply` (linia 54) dostaje mix rampowy + `*mscale`; wrapper (66-87) — nowa sygnatura; walidacje zostają. Komentarz-kontrakt „no YaRN/frequency factors" w `native_rope.hpp:7-8` do zaktualizowania.
- `native_qsa_indexer.cu`: `append` + wrapper (110-128) — to samo (spare key na pozycji 0 dostanie mscale — spójne, bo q_idx też jest skalowany).
- `prefill/kernels.cu`: `rope_kernel` (314-325) + wrapper `rope()` (~501) — to samo; hardcode n_rot=64 (`pair+32`, `-2.0f/64.0f`) zostaje z asercją.
- Call sites do przepięcia na `rope_scaling()`: layer.cpp:818-819, 872-873; verify.cpp:438, 467-469; mtp.cpp:334; prefill.cpp:964, 984, 987, ~993-999. (Ścieżki tabelaryczne tych samych miejsc działają automatycznie.)
- Nowa fixtura: zgoda **native vs tabela** na tym samym wejściu przy włączonym YaRN (tolerancja jak w istniejących testach native; `--use_fast_math`). Opcjonalnie oracle vs pinned llama.cpp przez istniejącą ścieżkę `STRATA_ORACLE_*` — nice-to-have.

## Krok 4 — CLI, GGUF, spięcie w `generate.cpp`

- `Options` (~linia 91/119): `rope_scaling` (string, „none"), `rope_scale` (1.0), `rope_freq_base` (0 = default 1e7), `rope_freq_scale` (0 = z factor, pełna parity z llama.cpp), `yarn_orig_ctx`, `yarn_ext_factor`, `yarn_attn_factor`, `yarn_beta_fast`, `yarn_beta_slow`.
- Pętla argv (~775-952) + `usage()` (~288-405): blok flag `--rope-scaling none|linear|yarn`, `--rope-scale F`, `--rope-freq-base N`, `--rope-freq-scale F`, `--yarn-orig-ctx N`, `--yarn-ext-factor F`, `--yarn-attn-factor F`, `--yarn-beta-fast F`, `--yarn-beta-slow F`. (Nieznana flaga nadal twardy błąd.)
- Walidacja (~1033-1049): typ z {none,linear,yarn}; `factor >= 1`; yarn: `orig_ctx >= 1`, betas > 0; yarn z factor==1 → ostrzeżenie (identity); przy factor>1 i `max_context <= orig_ctx` → linia informacyjna.
- Budowa `RopeScaling` i `rope_scaling_set()` **musi być przed `session_init` (1244) i przed capture** — miejsce obok `native_rope_set_enabled` (1176) / `mrope_table_set` (1192), z komentarzem „before any CUDA graph captures a rope kernel".
- Defaults z GGUF w istniejącym bloku (1197-1209, obok `expert_count`): `qwen4exp.rope.freq_base`, `qwen4exp.rope.scaling.type|factor|original_context_length` — tylko gdy klucz obecny (dziś nie jest → no-op); CLI zawsze ma priorytet.
- Linia podsumowania na stderr przy starcie gdy scaling != none: „rope scaling: yarn factor 2, orig ctx 262144 → 524288" (widoczna w logach bench).

## Krok 5 — brak regresji na default

- Unit: F1 (bit-exact tabeli None).
- End-to-end: `--dump-logits` na stałym prompcie, greedy, `--rope-scaling none` — **identyczne logity** przed i po zmianie (wzorzec bench/results/2026-09-27-cache-parity). To jest warunek zielonego mdMerge dla kroków 2-4.

## Krok 6 — setup.py (serwer bez zmian)

- Passthrough flag `--rope-*`/`--yarn-*` do `cfg["args"]` (miejsce: ~1297-1300, gdzie `--context` → `--max-context`); `CONTEXTS` (linia 77) rozszerzone o 393216/524288, **dostępne tylko przy aktywnym skalowaniu** (walidacja z czytelnym komunikatem); RAM-fallback (1149-1151) objęty nowymi wartościami.
- `serve/server.py` i `tools/needle_bench.py`: zero zmian — flagi płyną przez config args, `READY <max_context>` już propaguje kontekst, needle_bench czyta `/metrics`.

## Krok 7 — ustalenie natywnego kontekstu, bench jakościowy, docs, wersja

- **Probe**: na `none` recall z needle_bench przy 262144 → 327680 → 393216 ustala punkt degradacji = natywny kontekst treningowy; to domyka default `--yarn-orig-ctx` (wynik zapisany w README bench).
- **Matrix** (temperature 0): baseline none@262144; linear/yarn @393216 (factor 1.5) i @524288 (factor 2.0); głębokości 0/50/100%. Wyniki wg konwencji: `bench/results/<data>-rope-scaling/` z README.md (sprzęt, wersja, flagi, metoda, zastrzeżenia, tabela), matrix.json, config.json (recipe serwera), engine.log. Exit code 1 przy chybieniu.
- `docs/DETAILS.md`: sekcja „Extending the context (rope scaling)" w „Using it" (tabela flag, domyślne, mapowanie na llama.cpp, wyniki needle bench, ograniczenia) + wiersz w Troubleshooting. README: jedno zdanie w „Using it".
- Wersja: `CMakeLists.txt:11` → **0.1.18**; `setup.py:60` `MIN_ENGINE = (0, 1, 18)` z komentarzem historii.
- Kolejność commitów (konwencja przedmiot + numer issue):
  1. `kernels: rope scaling config and the scaled cos/sin table (none/linear/yarn) + parity`
  2. `rope: thread scaling through the native, indexer and prefill rotation sites (#NN)`
  3. `engine: --rope-scaling/--rope-scale/--yarn-* flags; rope keys as GGUF defaults`
  4. `setup: --rope-* passthrough; contexts beyond 262144 require scaling` + `docs: context extension (rope scaling)`
  5. `Engine 0.1.18; setup requires it (rope scaling, #NN)` + `bench: needle recall at 393K/524K with rope scaling`

## Ryzyka i niezmienniki

1. **CUDA graphs**: wszystkie nowe argumenty kerneli to stałe procesu (bake przy capture — poprawne); pozycje nadal z device. Zmiana skalowania w trakcie procesu jest niemożliwa z założenia — dokumentujemy.
2. **K w cache przechowywane post-RoPE z mscale**: spójne, bo konfiguracja procesowa; prompt-cache wyłącznie w RAM, więc brak zagrożenia między uruchomieniami.
3. **Indekser**: q_idx i poodolny klucz skalowane mscale → iloczyny ×mscale²; top-k jest niezmienny na skalowanie; **do sprawdzenia w kodzie**, czy scoring indeksera nie ma progów bezwzględnych (gate_attn to inny gate — QSA output, nie indekser).
4. **VRAM tabeli**: rośnie liniowo — ~64 MiB @262K → ~128 MiB @524K (2 tablice × 4 B × 32 pary).
5. **mrope/wizja**: skalowanie działa na pozycjach czytanych z tabeli (t,h,w) — komponuje się naturalnie; bench objmuje tekst, GENI+scaling odnotujemy jako nieprzetestowane.
6. **Obie ścieżki (tabela i `--native-rope`) muszą się zgadzać** — nowa fixtura porównawcza; walidacje native (freq_base > 1) zostają.
7. Skalowanie jest **per-process, nie per-request** (bez klucza w linii GEN) — świadoma decyzja, odnotowana w docs.

Pliki: nowy `rope_scaling.hpp`; zmiany w rope.cu/.hpp, rope_parity.cpp, native_rope.cu/.hpp, native_qsa_indexer.cu/.hpp, prefill/kernels.cu/.hpp, prefill.cpp, layer.cpp, verify.cpp, mtp.cpp, generate.cpp, setup.py, docs/DETAILS.md, CMakeLists.txt.