#!/bin/bash
# empacotar.sh - faz os packs (tokenizer, pesos densos, indice dos experts; ~1,4 GB cada) dos 6 modelos com a ferramenta da
# 0.2.0 (tools/iq_pack.py, versao 0.1.45.3), para que todos sejam feitos pela mesma versao. Nao usa a GPU; le os GGUF sem alterar.
#   empacotar.sh [todos | q2 q3 q4 q5 q6 q8]
set -eu
T=<data>/testes-020; C=<data>/strata-020/control; PY=$C/python/bin/python3
declare -A G=(
  [ud-q2_k_xl]=$T/modelos/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf
  [ud-q3_k_xl]=$T/modelos/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf
  [ud-q4_k_xl]=<data>/Strata-data/models/unsloth-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
  [ud-q5_k_xl]=$T/modelos/UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00006.gguf
  [ud-q6_k_xl]=<data>/testes-q6-q8/modelos/UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf
  [q8_0]=<data>/testes-q6-q8/modelos/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf )
declare -A A=([q2]=ud-q2_k_xl [q3]=ud-q3_k_xl [q4]=ud-q4_k_xl [q5]=ud-q5_k_xl [q6]=ud-q6_k_xl [q8]=q8_0)
[ "${1:-todos}" = todos ] && lista="q2 q3 q4 q5 q6 q8" || lista="$*"
export PYTHONDONTWRITEBYTECODE=1 STRATA_GGUF_PY=${STRATA_GGUF_PY:-<data>/strata-0145/strata-src/third_party/llama.cpp/gguf-py}   # o pacote 0.2.0 nao traz o gguf-py; usa o da producao (so leitura)
for a in $lista; do
    m=${A[$a]}; [ -f $T/packs/$m/PRONTO ] && { echo "$m: pack ja pronto"; continue; }
    echo "$(date '+%F %T') pack $m <- ${G[$m]}"
    (cd $C && $PY tools/iq_pack.py --gguf "${G[$m]}" --out $T/packs/$m --compat-bf16) > $T/logs/pack-$m.log 2>&1 \
        || { echo "ERRO no pack de $m (ver logs/pack-$m.log)"; tail -5 $T/logs/pack-$m.log; continue; }
    awk 'NR>1 {print $2"/"$3}' $T/packs/$m/native_experts.txt | sort | uniq -c   # tipos gate-up/down dos experts
    date '+%F %T' > $T/packs/$m/PRONTO
done
