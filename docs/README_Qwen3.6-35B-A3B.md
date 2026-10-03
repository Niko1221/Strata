# strata-qwen36

Motore nativo CUDA per **Qwen3.6-35B-A3B** dentro Strata, per RTX 4090 / CUDA 12.8 / Linux. Parla lo stesso protocollo
stdin/stdout del motore di Strata, quindi `serve/server.py`, tokenizer, API, web app e MCP restano **intatti**.
Progetto e motivazioni: `docs/QWEN36_NATIVE.md`.

## Stato onesto
Scritto e controllato in compilazione (g++ per l'host, clang CUDA per i kernel sm_89), con test CPU che passano
(`sampler_test`, `config_test` su un GGUF vero scritto da `gguf-py`). **Mai eseguito su GPU**: non avevo una GPU.
La correttezza numerica si decide con il selftest qui sotto.

## 1. Installare in Strata (tocca una sola riga di Strata)
```bash
./apply.sh /percorso/Strata
cd /percorso/Strata
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89  -DSTRATA_ENABLE_CUDA=ON -DGGML_CUDA=ON
cmake --build build -j --target strata-qwen36
```

## 2. Test, in quest'ordine
```bash
# CPU (subito)
g++ -std=c++17 -Itests/../src/qwen36 tests/sampler_test.cpp -o /tmp/st && /tmp/st
python tests/write_shape_gguf.py /tmp/shape.gguf && g++ -std=c++17 -Isrc/qwen36 -I$STRATA/include tests/config_test.cpp -o /tmp/ct && /tmp/ct /tmp/shape.gguf

# GPU: oracolo HuggingFace -> GGUF F32 minuscolo -> confronto dei logit (serve torch, transformers>=5, gguf)
python tests/make_tiny_model.py /tmp/tiny
build/src/qwen36/strata-qwen36 --native /tmp/tiny/tiny.gguf --selftest /tmp/tiny/expected.txt
```
`SELFTEST PASSED` = GDN, attention, RoPE parziale, router, shared expert, mapping delle head e forward completo
sono giusti senza quantizzazione. Se fallisce, l'output indica la prima posizione che diverge: partire da lì
(layer GDN prima dell'attention, perché i token 0-2 non attraversano l'attention a lungo contesto).

## 3. Uso con il vero modello
Un GGUF di Qwen3.6-35B-A3B prodotto dal convertitore di llama.cpp (Q4_K_M sta nei 24 GB). Tokenizer e template:
```bash
# python -c "import sys; sys.path.insert(0,'tools'); import strata_tokenizer as T; T.extract('Qwen3.6-35B-A3B-Q4_K_M.gguf','qwen36')"
tests/qwen3.6-35b-a3b_config_maker.py < path_to_your_Qwen3.6-35B-A3B-Q4_K_M.gguf >
```
Config per `serve/server.py` (schema di Strata; `exe` è l'eseguibile nuovo):
```json
{ "exe": "build/src/qwen36/strata-qwen36",
  "args": ["--native", "/m/Qwen3.6-35B-A3B-Q4_K_M.gguf", "--max-context", "32768", "--eos-ids", "<id di <|im_end|>>"],
  "tokenizer": "qwen36/tokenizer", "model_name": "qwen3.6-35b-a3b",
  "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5} }
```
`python serve/server.py --engine strata --config qwen36.json --port 8080`. L'id di `<|im_end|>` si ottiene dal
tokenizer estratto (`vocab.json`). Sampling consigliato dalla model card: thinking 1.0/0.95/20/presence 1.5,
coding 0.6/0.95/20/presence 0, instruct 0.7/0.8/20/presence 1.5.

## Cosa può ancora andare storto (da leggere prima di fidarsi)
- **Nomi tensori e metadati** sono quelli del convertitore llama.cpp letti dal suo sorgente; il loader dice quale
  tensore manca. Il tipo di quantizzazione di ogni tensore deve essere uno di quelli del GEMV nativo di Strata
  (Q4_0/Q5_0/Q8_0/Q3_K/Q4_K/Q5_K/Q6_K/IQ4_NL/IQ4_XS) o F32/F16/BF16; altri tipi (es. Q2_K, Q4_1) vengono rifiutati.
- **Mapping delle head V** `head % h_k`: vero per i GGUF dei convertitori recenti (che riordinano le head); un GGUF
  vecchio darebbe un output sbagliato ma plausibile. Il selftest non lo copre per un GGUF reale, solo per quello
  generato dallo script con la stessa trasformazione.
- **Prestazioni**: Fase 1, sequenziale, una sync per layer per gli id degli esperti, prefill token per token.
  Prevedi decine di token/s in decode e prefill lento; nessun numero misurato.
- Solo testo, niente MTP né visione, il riuso della cache vale solo se il nuovo prompt estende il precedente.
