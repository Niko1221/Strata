# strata-qwen36: MoE raggruppato + MTP

File interi (non patch): copiali sopra al tuo albero Strata mantenendo i percorsi.
  src/kernels/cuda/iq_kernels.cu      (UNA riga: IQ4_NL ammesso come gate/up, riga ~510)
  src/qwen36/{qwen36_kernels.cu,.cuh,qwen36_model.cpp,.hpp,main.cpp,moe_group.hpp}
  tests/{moe_group_test.cpp,compare_dumps.py}

Verificato qui: sintassi CUDA (clang, sm_89) e test CPU del raggruppamento. NON verificato: numerica e velocita' su GPU.

## 1. MoE raggruppato
  ./strata-qwen36 --native M.gguf --selftest-dump 300 g1.bin           # raggruppato (default)
  Q36_GROUPED=0 ./strata-qwen36 --native M.gguf --selftest-dump 300 g0.bin
  python3 tests/compare_dumps.py g0.bin g1.bin
Il log di caricamento dice quanti layer sono raggruppati (Q6_K down -> percorso per-esperto).

## 2. MTP
  python3 -c "from transformers import AutoTokenizer as T;t=T.from_pretrained('Qwen/Qwen3.6-35B-A3B');print(','.join(map(str,t('Scrivi una breve storia su un gatto che impara a volare, con dettagli.')['input_ids'])))" > prompt_ids.txt
  ./strata-qwen36 --native M.gguf --mtp --selftest-mtp 200 prompt_ids.txt   # stampa la variante migliore
  ./strata-qwen36 --serve --native M.gguf --mtp --mtp-k 3 --mtp-variant N
Se il GGUF non contiene blk.40.nextn.* il log lo dice e l'MTP resta spento (serve un GGUF che li includa).
Se nessuna variante supera ~20% mandami l'elenco dei tensori blk.40.*.
L'uscita resta corretta anche con una testa sbagliata: cambia solo la percentuale di accettazione (DONE riporta draft/accettati).
Costo VRAM MTP: ~60 MB per riga di verifica (checkpoint GDN) + testa (~0.5 GB).

## 3. KV cache: --kv e --kv-unified
  --kv fp16|int8   int8 = codici a 8 bit + una scala F16 ogni 32 valori (1,06 byte/valore invece di 2, ~47% in meno)
  --kv-unified     un'unica allocazione GPU per K e V di tutti i layer di attenzione (e dell'MTP)
Solo le 10 layer di attenzione (+1 MTP) hanno KV; le 30 GDN hanno uno stato di dimensione fissa.
Il log stampa "KV cache: ... GiB". int8 introduce un piccolo errore: confronta con --selftest-dump
(--kv fp16 vs --kv int8) con tests/compare_dumps.py prima di fidarti sui contesti lunghi.
Non implementati: q4_0, k8v4, --kv-resident di Strata.

## 4. Snapshot del prompt (prefix cache): --ckpt-max, --ckpt-every
Lo stato GDN non si puo' riavvolgere: se il template riscrive il turno precedente (es. toglie il "thinking") il vecchio riuso
falliva e rileggeva tutto il prompt. Ora il motore salva lo stato (GDN + conv + carry MTP, ~62 MB) in RAM pinned ogni
--ckpt-every token di prompt (default 8192) e alla fine di ogni prompt; alla richiesta successiva ripristina il piu' lungo
snapshot che e' prefisso del nuovo prompt e legge solo il resto. Si tengono al massimo --ckpt-max snapshot (default 12, ~750 MB di
RAM; 0 = spenti), LRU. Il log stampa "prefix cache: restored a snapshot at X of Y prompt tokens". Gli snapshot la cui KV
verrebbe sovrascritta da una conversazione diversa vengono scartati (la KV e' una sola).
Non verificato su GPU.
