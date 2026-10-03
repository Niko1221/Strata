# strata-qwen36 — porting nativo di Qwen3.6-35B-A3B in Strata

Un nuovo eseguibile CUDA, `strata-qwen36`, che parla lo **stesso protocollo stdin/stdout** del motore di Strata
(`--serve`, `GEN`, `T`, `PP`, `DONE`, `STOP`, `QUIT`). Il server Python (`serve/server.py`), il tokenizer, le API, la web
app e MCP restano **intatti**: cambia solo l'eseguibile. Nessun llama.cpp a runtime.

Target di verifica: RTX 4090 (sm_89, 24 GB), CUDA 12.8, Linux. Stato: scritto e controllato in compilazione, **mai eseguito
su GPU** (vedi "Verifica").

## 1. Perché un eseguibile nuovo e non una modifica del motore

Il motore di Strata è costruito su Qwen3.8-Flash-Next: residui a 4 stream (hyper-connections), attention QSA con indexer,
PLE e tabella n-gram, 512 esperti, hidden 2560. Circa 20.000 righe in 14 file (`session`, `layer`, `prefill`, `verify`,
`mtp`, `pinned`, `generate`…) dipendono dai residui a 4 stream. Qwen3.6 ha un residuo normale, quindi quel percorso andrebbe
riscritto. Un driver separato che **riusa solo i kernel già validati** non rischia regressioni su Qwen3.8.

## 2. Architettura di Qwen3.6-35B-A3B (verificata sul codice di riferimento `transformers` e sul convertitore llama.cpp)

| | |
|---|---|
| layer | 40, schema `3 × GDN + 1 × attention` ripetuto 10 volte |
| hidden | 2048, vocab 248.320 |
| blocco | `x += Mixer(norm(x))`, poi `x += MoE(norm(x))`: pre-norm, residuo normale |
| GDN | `attn_qkv` (q 2048 \| k 2048 \| v 4096), `attn_gate` = z, conv1d k=4 + SiLU, 16 head K, 32 head V, S=128, norm gated con **silu(z)** |
| attention | 16 head Q, 2 KV, head_dim 256, `attn_q` produce `[q \| gate]` per head, q/k-norm, RoPE parziale (64 dim, NeoX), uscita × sigmoid(gate) |
| MoE | router softmax su 256, top-8, pesi **rinormalizzati**; 8 esperti (ff 512) + 1 shared (ff 512) con gate `sigmoid(w·x)` |
| norm | i pesi nel GGUF hanno già il `+1` incorporato (convertitore): si usa `w` così com'è |

Dal GGUF (generato dal convertitore llama.cpp) valgono: `ssm_a = -exp(A_log)`, head V riordinate **a piastrella**
(`v_head % n_k_head`, come fa `native_gdn_step`), conv1d `[channels, 4]`. Tutte le dimensioni si ricavano dalle **forme dei
tensori**, non da costanti: i metadati servono solo per rope theta, rope dim, eps e `expert_used_count`.

## 3. Cosa si riusa e cosa è nuovo

Riusato da Strata, senza modifiche: `GgufModel` (lettura/mmap GGUF), `dequantize_*`, `native_mmvq` (GEMV su quantizzazioni
GGUF standard: Q4_0/Q5_0/Q8_0/Q3_K/Q4_K/Q5_K/Q6_K/IQ4_NL/IQ4_XS), `native_quantize_q8_1`, `native_gdn_conv_silu`,
`native_gdn_l2_norm`, `native_gdn_gate`, `native_gdn_beta_gate`, `native_gdn_step`.

Scritto da zero (`qwen36_kernels.cu`): RMSNorm, GEMV F32/F16/BF16, router softmax+top-k, SiLU·mul, combine MoE con shared
gate, norm gated con SiLU, prep q/k (norm + RoPE parziale + gate), attention decode flash-style su KV in F16.
`native_gdn_out_norm` di Strata **non** si usa: fa sigmoid(z), Qwen3.6 usa silu(z).

## 4. Flusso per token (Fase 1, sequenziale)

```
embed (dequant riga su host) -> per layer: rmsnorm -> {GDN | attention} -> +residuo -> rmsnorm -> MoE -> +residuo
-> output_norm -> output.weight -> logit su host -> sampler su host
```
MoE: router su GPU, 8 id in host (una sync per layer), poi 3 GEMV per esperto con puntatore `base + id * stride`
(`native_mmvq_weight_bytes`), poi combine. Pesi interi in VRAM (un Q4_K_M sta nei 24 GB; KV F16 = 20 KB/token, 32K = 0,7 GB).

## 5. Protocollo

`--serve --native <gguf> [--max-context N] [--eos-ids a,b]`. Righe: `GEN <max_new> k=v... <id,id,...>`, `STOP`, `QUIT`.
Risposte: `INFO ...`, `READY <ctx> stop`, `PP <done> <total> <ms> <tok/s>`, `T <id>`, `DONE <gen> <prompt> <prompt_ms> <decode_ms> <finish> 0 0 <reused>`, `ERR <msg>`.
Riuso della cache solo se il nuovo prompt **estende** quanto già calcolato (lo stato GDN non si riavvolge).

## 6. Verifica

1. `tests/sampler_test.cpp` (CPU): gira ovunque.
2. `tests/make_tiny_model.py`: crea con `transformers` un Qwen3.5-MoE minuscolo a pesi casuali, calcola i logit di
   riferimento con HF, applica le **stesse trasformazioni del convertitore** (head V a piastrella, norm+1, `-exp(A_log)`),
   scrive `tiny.gguf` (F32) e `expected.txt`.
3. `strata-qwen36 --native tiny.gguf --selftest expected.txt`: confronta i logit end-to-end (tolleranza 2e-3).
   Se passa, GDN, attention, RoPE, router, shared expert, mapping delle head e protocollo sono giusti **senza quantizzazione**;
   poi si prova il Q4_K_M reale.

## 7. Limiti della Fase 1 (dichiarati)

Solo testo, niente MTP, niente CUDA graph, prefill token per token (lento sui prompt lunghi), nessun offload su CPU/SSD,
nessun riuso di prefisso dopo il primo turno se il template riscrive la risposta precedente, `strata_tune` ignorato.
Prossime fasi: prefill a blocchi (GEMM), snapshot dello stato GDN per il riuso del prefisso, CUDA graph, MTP.

## 8. Prefill a blocchi (Fase 2)

`Model::forward_batch(tokens, n)` elabora fino a 128 token per passata; il decode (`forward`) è rimasto invariato.
Layout token-major `[B][larghezza]`.

- **Proiezioni**: `native_mmvq` a 8 colonne per volta (le colonne sono contigue, uscita `y + j*n_out`); la quantizzazione
  Q8_1 di N colonne è un solo lancio (un vettore di `n_in*N`). Pesi F32/F16/BF16: `gemv_float_cols`.
- **GDN**: conv causale + SiLU sequenziale per canale su B token, L2 su q/k, gate/beta, e la ricorrenza del delta rule
  con la **stessa struttura e lo stesso layout di stato di `native_gdn_step`** (una warp per colonna, 4 righe per lane),
  così lo stato passa senza conversioni dal prefill al decode.
- **Attention**: stessi algoritmi del decode con una query per blocco (`grid = (n_head, B)`); il token b vede `pos0+b+1`
  posizioni; K/V dell'intero blocco vengono scritti prima.
- **MoE**: router su B righe, una sola sync per layer, coppie (token, slot) ordinate per esperto (`moe_group.hpp`,
  testato su CPU); ogni esperto attivo gira una volta (a blocchi di 8 colonne) su tutti i suoi token, poi un combine
  che rilegge i risultati per posizione ordinata e aggiunge lo shared expert con il suo gate.
- **Verifica**: `--selftest-batch N` confronta prefill a blocchi e token per token sullo stesso modello (logit dopo il
  prefill, dopo 6 passi di decode, sequenza greedy) e stampa la velocità di entrambi.

## 9. GEMM tensor-core nelle proiezioni del prefill a blocchi (Fase 3)

Le proiezioni del percorso batch non girano più solo sulla GEMV: passano dai **tensor core** attraverso il
`strata::prefill::Gemm` di Strata ([`src/prefill/gemm.cu`](../src/prefill/gemm.cu)), lo stesso oggetto che usa il motore
di Qwen3.8 (cuBLAS su CUDA, hipBLAS/hipBLASLt su HIP). `gemm.cu` è compilato dentro `strata-qwen36`; non si linka
`strata_prefill`, che porterebbe dentro tutto il motore qwen3.8.

Cosa fa il percorso, per proiezione:

- pesi in blocchi GGUF a blocchi larghi (Q4_0, Q5_0, Q5_1, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, Q2_0) →
  `Gemm::native` li dequantizza in **F16** nello scratch e chiama `cublasGemmEx` (`dequant_bf16.cu`, trascritto da
  `ggml-quants.c`);
- i-quant stretti (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, **IQ3_S**, IQ1_M), che sono quelli con cui `UD-IQ4_NL` quantizza
  `ffn_gate_exps` e `ffn_up_exps`: li dequantizza `iq_dequant_f16` (`iq_kernels.cu`), lo stesso ramo a cui
  `dequant_f16` rimanda quando `dequant_bf16` non ha la geometria. Il gate del motore è l'unione dei due insiemi, che
  è esattamente l'insieme che la GEMV nativa sa leggere: tutto ciò che si carica può anche passare dal GEMM;
- pesi già F16 nel file → `Gemm::f16` diretto, senza dequantizzazione;
- pesi F32 (i modelli minuscoli dei test) → conversione F32→F16 nello scratch, solo se il tensore ci sta;
- pesi **BF16 → restano sulla GEMV**: il GEMM vorrebbe anche le attivazioni in BF16, e una seconda copia di ogni
  buffer di attivazione non vale la pena finché le quantizzazioni in uso sono tutte a blocchi.

Le attivazioni F32 vengono copiate in F16 con `q36::to_f16`, che usa `strata::kernels::f16_from_f32`
([`include/strata/kernels/f16_bits.hpp`](../include/strata/kernels/f16_bits.hpp)) e **non** `__float2half`, il cui
comportamento con bit sbagliati è documentato lì.

Soglie (`--gemm-min`, `--gemm-min-expert`):

| | default | perché |
|---|---|---|
| proiezioni dense (qkv, q/k/v/o, shared expert, `ssm_out`) | 16 colonne | sotto, il GEMM perde: la GEMV legge i blocchi senza dequantizzare |
| un singolo esperto instradato | 32 colonne | qui il GEMM paga anche la dequantizzazione dei blocchi di **quell'esperto** |

Il **router `ffn_gate_inp` non passa mai dal GEMM**, nemmeno quando è abbastanza largo: è F32 nel file, e arrotondare
pesi e logits a F16 può spostare un logit oltre il confine del top-8, cambiando gli esperti scelti e quindi tutta la
risposta, per una proiezione da 256 × 2048 che costa una frazione di millisecondo in entrambi i modi. Strata fa la
stessa cosa nel suo prefill, dove il router gira su una coppia di GEMM in BF16 con la parte bassa (≈16 bit di
mantissa, [`src/prefill/prefill.cpp`](../src/prefill/prefill.cpp)).

`--no-gemm` spegne tutto e riporta ogni proiezione sulla GEMV nativa. Se `Gemm::init` fallisce (cuBLAS assente o
VRAM finita) il motore stampa il motivo e continua sulla GEMV: non è un errore fatale. Lo scratch è grande quanto il
peso più grande che il percorso dequantizza in un colpo solo; `token_embd.weight` e `output.weight` sono esclusi,
perché i loro logit li produce la GEMV del decode e dimensionare lo scratch su di loro costerebbe ~1 GiB per un
percorso che non gira mai.

Su Qwen3.6-35B-A3B lo scratch resta piccolo, perché un tensore MoE non viene mai dequantizzato intero: `mat()`
dimensiona su un singolo esperto (`ffn_*_exps` è 2048 × 512 per esperto = 2 MB in F16) e il tensore più grande in
assoluto è `attn_qkv` (2048 × 8192 = 33,5 MB). Con i 32 MB di workspace di cuBLAS, il percorso GEMM costa ~66 MB di
VRAM su un modello da 18,5 GB.

Il **decode token-per-token non usa il GEMM**: una GEMV che legge i blocchi GGUF è più veloce di una moltiplicazione
tensor-core che prima deve dequantizzare. Il GEMM cambia solo come viene processato un prompt.

Verifica: `--selftest-batch N` stampa anche la riga `GEMV vs GEMM (stessi blocchi)` con coseno, KL, overlap top-20,
`maxrel`, argmax, i token greedy in comune e i millisecondi dei due percorsi a parità di chunking. Il confronto a
parità di blocchi isola l'effetto del percorso tensor-core dal rumore del chunking.
