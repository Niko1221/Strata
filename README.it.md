<h1 align="center">Strata</h1>

**Italiano** · [English] · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>Esegui un modello di IA da 125 miliardi di parametri sul tuo normale PC da gaming</b><br>

Scheda grafica NVIDIA o AMD (12 GB o più) · Windows o Linux · gratuito e open source</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="Un giardino con una pagoda voxel creato dal modello di Strata, in esecuzione nel browser"></a><br>

<sub>Un giardino con una pagoda voxel, prompt di una sola richiesta eseguito su una RTX 5070 con Strata (IQ3_S, contesto da 128K) ·

<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">video completo (49 s)</a></sub></p>

Strata esegue **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** su un normale PC. È un

modello di IA grande e intelligente che normalmente richiede un server. Chatta, scrive codice, legge immagini e lavora con le tue app

e i tuoi agenti di coding. Nulla lascia il tuo PC.

## Quanto è veloce?

Lo abbiamo misurato su due normali PC da gaming. Un token corrisponde a circa ¾ di una parola.

- **Scrive le risposte:** indica la velocità con cui compare la risposta in una breve chat. 60 token al secondo sono più veloci di quanto tu possa leggere.

- **Legge il tuo prompt:** indica la velocità con cui acquisisce ciò che invii (qui un documento, codice o cronologia della chat da 32K token).

<table>

<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM</th></tr>

<tr><td>

| Dimensione | Scrive le risposte | Legge il tuo prompt |

| --- | ---: | ---: |

| **Q2_0** | 94 token/s | 2.650 token/s |

| **IQ2_XS** | 79 token/s | 2.090 token/s |

| **IQ3_XXS** | 62 token/s | 1.750 token/s |

| **IQ3_S** | 53 token/s | 1.620 token/s |

| **Coder** | 55 token/s | 2.180 token/s |

</td><td>

| Dimensione | Scrive le risposte | Legge il tuo prompt |

| --- | ---: | ---: |

| **Q2_0** | 60 token/s | 1.160 token/s |

| **IQ2_XS** | 52 token/s | 1.110 token/s |

| **Coder** | 44 token/s | 1.420 token/s |

</td></tr>

</table>

NVIDIA: Q2_0 con engine 0.1.36, le altre righe con 0.1.26 (risposte da 4K, prompt da 32K). Le tabelle complete sono in

[DETAILS.md](docs/DETAILS.md#speed-measured). Una scheda con più VRAM è più veloce: una RTX 3090 (24 GB) dovrebbe scrivere

circa 100-140 token al secondo. Chat lunghe e altre schede: [velocità di ogni modello](docs/MODELS.md#how-fast-is-each-size),

[risultati della community](docs/COMMUNITY_BENCHMARKS.md).

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Offrimi un caffè" height="50"></a><br>

<sub>Strata è gratuito. Se funziona bene sul tuo PC, un caffè aiuta a continuare a svilupparlo.</sub></p>

## Cosa ti serve

| | |

| --- | --- |

| **Scheda grafica** | **NVIDIA** GeForce RTX serie 20, 30, 40 o 50, oppure **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 o RX 6800 / 6900. Servono **12 GB di VRAM o più**. |

| **RAM** | 32 GB o più. La tua RAM determina [quale modello](#which-model-should-i-pick) può essere utilizzato. Con 64 GB funzionano tutte le dimensioni. |

| **Disco** | Circa 80 GB liberi. Se puoi, usa un SSD: il primo avvio sarà molto più veloce. |

| **Sistema** | Windows 10 / 11 o Linux e un driver grafico aggiornato di NVIDIA o AMD. |

L'installer configura tutto il resto. Due o tre schede possono condividere il modello ([multi-GPU](docs/MULTI_GPU.md)).

Sperimentale, scritto e testato dai membri della community sulle proprie macchine:

- **Schede grafiche meno recenti** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [GPU meno recenti](docs/OLDER_GPUS.md).

- **Intel Arc**, compilato dai sorgenti su Linux: [Intel Arc](docs/INTEL_ARC.md).

- **Processori meno recenti senza AVX2**: funzionano, ma lentamente. [CPU meno recenti](docs/INSTALL.md#older-cpus-experimental).

L'elenco completo: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Installazione

### Lascia che sia la tua IA a configurarlo

Usi un assistente di coding basato sull'IA (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Incolla questo testo:

```text

Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.

```

Controlla la tua scheda grafica, la RAM e il disco e sceglie il modello adatto. Poi lo installa, lo avvia e ti spiega

come collegare le tue app. Gli strumenti di IA possono anche installare, avviare e arrestare Strata tramite il suo

[MCP server](docs/MCP_SERVER.md).

### Oppure fallo tu

[Scarica Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) e decomprimilo (oppure esegui `git clone`).

**Windows:** fai doppio clic su **`START-HERE.bat`**. **Linux:** esegui **`./setup.sh`** nella cartella di Strata.

I passaggi sono gli stessi per NVIDIA e AMD. L'installer rileva la tua scheda e configura l'engine corretto. Ti

pone alcune domande:

- quale modello e quale dimensione,

- quanto contesto (quanto testo il modello mantiene a mente),

- se deve leggere le immagini.

Premi Invio ogni volta per accettare la risposta consigliata. Poi scarica il modello (circa 70 GB) e lo avvia. Se il

download si interrompe, eseguilo di nuovo: continuerà da dove si era fermato. Il browser apre l'app Strata all'indirizzo

`http://127.0.0.1:8080`.

> **Mentre il modello si avvia, il PC può rallentare o smettere di rispondere per 1-3 minuti** (il periodo più lungo si verifica la prima volta).

> Strata carica 35-55 GB nella RAM e ne blocca una parte per la scheda grafica. È normale. Attendi e non

> chiudere la finestra. La finestra mostra cosa sta facendo Strata.

**La volta successiva**, esegui di nuovo `START-HERE.bat` (oppure `./setup.sh`). Si avvia subito e non scarica nulla due volte. Chiudi

la sua finestra per arrestare il modello. `UPDATE.bat` (`./update.sh`) aggiorna Strata senza avviarlo. Aggiornamento, Docker,

più schede, posizione dei file e tutte le opzioni: [docs/INSTALL.md](docs/INSTALL.md).

## Quale modello dovrei scegliere?

L'installer ne consiglia uno in base alla tua RAM. Lo stesso modello è disponibile in diverse dimensioni, compresso in misura maggiore o minore. Le dimensioni più piccole

sono più veloci. Le dimensioni più grandi sono un po' più intelligenti.

| La tua RAM | Scegli | Perché |

| --- | --- | --- |

| **32 GB** | **Coder** | entra in 32 GB ed è realizzato per il codice (con una scheda da 24 GB, funzionano anche Q2_0 e IQ2_XS) |

| **48 GB** | **IQ2_XS** (oppure Q2_0, il più veloce) | le dimensioni più grandi non entrano |

| **64 GB** | **IQ2_XS** (consigliato), oppure IQ3_XXS / IQ3_S | entrano tutte le dimensioni; IQ3_S è la migliore e la più lenta |

| **96 GB o più** | **IQ3_S**, oppure Unsloth's UD-IQ4_XS (~4-bit) | c'è spazio per le dimensioni più grandi lasciando aperto tutto il resto |

- **[Coder](docs/MODELS.md#coder):** una versione per il coding con metà degli esperti rimossi. Raggiunge il 91% del punteggio SWE-bench Verified del

  modello completo (misurato dai suoi autori) ed entra in 32 GB di RAM. È più debole al di fuori del codice,

  incluso il cinese e altro testo CJK (#438). Per questi casi, scegli Q2_0, IQ2_XS o IQ3_S, che mantengono tutti gli esperti.

- **[Swift 1.5](docs/MODELS.md#swift-15):** un fine-tuning che ragiona per un tempo molto più breve prima di rispondere. Ottieni

  la risposta prima, con una qualità pressoché identica.

- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs):** la versione ~4-bit di Unsloth, con qualità compresa tra IQ3_S e

  UD-Q4_K_XL. Il download è di 94 GB. Con meno di ~80 GB di RAM, Strata legge una parte dal SSD

  mentre risponde, quindi è più lento (un SSD NVMe aiuta).

- **[Unsloth UD-Q4_K_XL](docs/MODELS.md#unsloth-ud-q4_k_xl-experimental)** (sperimentale): è quello più vicino al modello completo.

  Ma Strata legge la maggior parte del modello dall'SSD mentre risponde, quindi scrive solo 7-8,5 token/s su un PC con 64 GB.

- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs):** devi configurarlo manualmente. Non

  è presente nel menu dell'installer.

Dimensioni, download e compatibilità: [docs/MODELS.md](docs/MODELS.md). Per aggiungere un altro modello in seguito, esegui

`SETUP.bat` (Linux: `./setup.sh --setup`).

## Utilizzo

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="La scheda Monitor dell'app Strata accanto a un agente di coding"><br>

<sub>Il <b>Monitor</b> dell'app Strata (a sinistra) mentre un agente di coding scrive il giardino con la pagoda del video (a destra)</sub></p>

- **Nel browser:** apri `http://127.0.0.1:8080`. Contiene **Chat**, un **Monitor** in tempo reale del modello e della tua

  GPU/CPU/RAM e **About**, con le impostazioni e gli indirizzi.

- **Le tue app e i tuoi agenti di coding:** aggiungi un provider "compatibile con OpenAI" con l'URL di base

  **`http://127.0.0.1:8080/v1`**. Qualsiasi chiave API e qualsiasi nome del modello funzionano.

  - App che usano l'API di Anthropic: `http://127.0.0.1:8080/v1/messages` (Claude Code:

    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).

  - Codex CLI e altre app che usano l'API OpenAI Responses: `/v1/responses`

    ([configurazione](docs/DETAILS.md#the-responses-api-and-codex-cli)).

- **Ragionamento:** scegli **disattivato, basso, medio o alto** nel menu della chat o nel parametro "reasoning effort" della tua app. Disattivato è il

  più veloce. Alto è il migliore per le domande difficili.

- **Immagini:** rispondi sì a "Images?" durante la configurazione. Poi fai clic su **Picture** nella chat oppure allega immagini nella tua app.

  Le schede AMD leggono le immagini su Linux tramite il processore; su Windows non è ancora possibile.

- **Dal telefono o da un altro PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Imposta sempre una chiave.

- **Una richiesta alla volta:** per impostazione predefinita Strata risponde a una richiesta e le altre attendono. Per rispondere a più richieste contemporaneamente,

  imposta `"parallel": 2` ([BATCHING.md](docs/BATCHING.md)). Su una scheda da 12 GB questo rende ogni risposta più lenta.

- **Prompt lunghi:** Strata legge per intero il primo messaggio di una chat, circa 1 minuto ogni 30.000 token. I messaggi successivi

  iniziano in pochi secondi.

Altro: [dove vengono archiviate le tue chat](docs/INSTALL.md#where-things-are-stored), [l'API](docs/DETAILS.md#using-it).

## Qualcosa è andato storto?

- **Il mio PC si è bloccato la prima volta che Strata si è avviato.** È normale durante il caricamento del modello. Attendi e non chiudere la

  finestra. È ancora bloccato dopo 10 minuti? Riavvia il PC, chiudi gli altri programmi e riprova, oppure scegli una dimensione più piccola.

- **Si è fermato durante il download o l'installazione.** Esegui di nuovo `START-HERE.bat` (oppure `./setup.sh`). Continua da

  dove si era fermato.

- **È molto lento e la spia del disco continua a lampeggiare, oppure dice "the engine stopped unexpectedly".** Il tuo PC non ha

  abbastanza RAM libera. Chiudi gli altri programmi (i browser ne usano molta) oppure scegli una dimensione più piccola (Q2_0 o IQ2_XS).

- **Dice che la porta 8080 è già in uso.** Strata è già in esecuzione. Cerca la sua finestra.

Altri problemi e relative soluzioni: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Sei ancora bloccato? Apri una

[issue](https://github.com/Niko1221/Strata/issues) e allega `strata-<model>.log` dalla cartella di Strata. Hai trovato un

problema di sicurezza? Segnalalo privatamente: [SECURITY.md](SECURITY.md).

## Come funziona?

Modelli come questo normalmente vengono eseguiti su server con centinaia di gigabyte di memoria grafica. La tua scheda grafica ha

12-24 GB. Strata fa entrare il modello **condividendo il lavoro con tutto il PC**. Pensa a una cucina: le cose

che usi continuamente restano sul bancone e il resto aspetta nella dispensa.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="I 24.576 esperti del modello: quelli più utilizzati sulla scheda grafica, tutti nella RAM, una tabella di ricerca sull'SSD"></p>

- **Il modello è un team di 24.576 piccoli specialisti ("esperti").** Ogni parola ne utilizza solo 10.

- **La tua scheda grafica** mantiene i poche migliaia di esperti utilizzati più spesso. **La tua RAM** li contiene tutti,

  e **il tuo processore** lavora contemporaneamente sul resto. **Il tuo SSD** contiene una grande tabella di ricerca.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="Un piccolo assistente indovina le parole successive; il modello grande le controlla tutte insieme e mantiene quelle corrette"></p>

- **Indovina, poi controlla:** un piccolo assistente indovina le prossime parole. Il modello grande le controlla tutte insieme. Ottieni

  la stessa risposta, 1,6-1,8 volte più velocemente.

- **I testi lunghi vengono letti in blocchi grandi** (fino a 8.192 token alla volta), a oltre 1.000 token al secondo.

La spiegazione più approfondita: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Ogni parte e i relativi numeri:

[i dettagli](docs/DETAILS.md#how-it-works) e il [paper](docs/paper/Strata-Paper.pdf).

## Crediti e licenza

Il modello è [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) del team Qwen. È stato

compresso da [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5)

e Unsloth. Strata utilizza parti di [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Tutti i crediti:

[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata è open source con [licenza MIT](LICENSE). Alcune

parti e ogni modello hanno licenze proprie ([quali](docs/HOW_IT_WORKS.md#license)).

## Supporta Strata

Strata è gratuito e open source. Se ti è utile, puoi sostenere il suo sviluppo:

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Offrimi un caffè" height="50"></a></p>
