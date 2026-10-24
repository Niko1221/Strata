#!/bin/bash
# pipeline.sh - bateria Strata 0.2.0 x 6 quantizacoes (UD-Q2..Q6_K_XL, Q8_0) numa L40S-48Q, sem supervisao. Retomavel:
# um teste com resultados/<tag>/FEITO e pulado. Detalhes e tempos: ../PLANO.md
#   antes: a GPU livre (gpu-stop maya / gpu-stop strata). Depois:
#   setsid nohup <data>/testes-020/scripts/pipeline.sh > <data>/testes-020/logs/pipeline.log 2>&1 &
# Ordem: (1) os 6 modelos na 0.2.0, do mais leve (Q2) ao Q8_0, cada pack feito na hora (o Q5 espera o download);
# (0) referencia Q8_0 no Strata 0.1.45 de producao; (2) se LONGO=<modelo>: so confere que 1M de contexto sobe e responde
# (info + 1 pedido de 8K, nada longo); cada modelo e a referencia tambem fazem as provas (funcao provas, ~1-1,5 h cada); (3) KL de todos contra a referencia (e contra a 0.1.42 de 10/10) e RESUMO.md.
# Variaveis: BATERIA3D=1 espera, em cada modelo, a bateria do Windows (estado/3D-FEITO-<tag>, ver PLANO.md);
#            LONGO=q8_0 (ou outro) roda a fase 1M; VOLTAR=0 nao sobe a producao no fim (padrao: gpu-start strata).
set -u
T=<data>/testes-020; S=$T/scripts; R=$T/resultados; PY=<data>/strata/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 TESTES_DIR=$T STRATA_SRC=<data>/testes-q6-q8/strata-src
BENCH=<data>/strata-0145/strata-src/tools/strata_bench.py
log() { echo "$(date '+%F %T') [pipeline] $*" | tee -a $T/logs/pipeline-eventos.log; }
M() { timeout "${2:-7200}" $PY $S/medir.py $1 >> $R/$TAG/medir.log 2>&1 || log "AVISO: medir $1 falhou em $TAG (rc $?)"; }

volta() {
    $S/alternar.sh parar-teste >/dev/null 2>&1
    if [ "${VOLTAR:-1}" = 1 ]; then log "fim/interrupcao: subindo a producao (gpu-start strata)"; <data>/gpu-services/gpu-start strata >> $T/logs/pipeline-eventos.log 2>&1
    else log "fim/interrupcao: a producao NAO foi iniciada (VOLTAR=0)"; fi
}
trap volta EXIT
trap 'log "sinal recebido"; exit 1' INT TERM

espera_3d() {   # o servidor fica no ar ate a bateria do Windows terminar (ou 12 h)
    [ "${BATERIA3D:-0}" = 1 ] || return 0
    touch $T/estado/ESPERANDO-3D-$TAG; log "  $TAG: esperando a bateria 3D (crie estado/3D-FEITO-$TAG quando terminar)"
    local n=0; while [ ! -f $T/estado/3D-FEITO-$TAG ] && [ $n -lt 43200 ]; do sleep 60; n=$((n+60)); done
    rm -f $T/estado/ESPERANDO-3D-$TAG; [ -f $T/estado/3D-FEITO-$TAG ] && log "  $TAG: bateria 3D feita" || log "AVISO: $TAG: 3D nao terminou em 12 h"
}

roda() {   # roda <tag> <medidas...>   ("bench" e "3d" sao do pipeline; o resto vai para o medir.py)
    TAG=$1; shift
    [ -f $R/$TAG/FEITO ] && { log "pulando $TAG (ja feito)"; return 0; }
    mkdir -p $R/$TAG/logits
    log "=== $TAG"
    if ! STRATA_ENV="STRATA_DUMP_FIRST_LOGITS=$R/$TAG/logits/l" $S/alternar.sh teste $TAG >> $R/$TAG/partida.log 2>&1; then
        log "ERRO: $TAG nao subiu (ver $R/$TAG/partida.log e logs/servidor-$TAG.log)"
        echo "falhou na partida $(date '+%F %T')" > $R/$TAG/FALHOU; return 1
    fi
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > $R/$TAG/vram-pronto.txt; free -g | sed -n 2p > $R/$TAG/ram-pronto.txt
    local med; for med in "$@"; do
        log "  $TAG: $med"
        case "$med" in
            bench) STRATA_KEY=$(cat $T/configs/.api_key-teste) timeout 3600 $PY $BENCH --url http://127.0.0.1:8090 --runs 3 \
                       --out $R/$TAG/bench --quiet >> $R/$TAG/bench.log 2>&1 || log "AVISO: bench falhou em $TAG" ;;
            3d) espera_3d ;;
            *) set -- $med; local cmd=$1; shift; M "$cmd $TAG $*"; set -- ;;
        esac
    done
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > $R/$TAG/vram-final.txt; free -g | sed -n 2p > $R/$TAG/ram-final.txt
    $S/alternar.sh parar-teste >/dev/null; touch $R/$TAG/FEITO; log "  $TAG concluido"
}

provas() {   # provas <tag>: sobe de novo (sem gravar logits) e roda as partes que faltam (marcadores PROVA-<parte>):
             # HumanEval+ 164; GSM8K e IFEval 100 cada (como em 10/10); MATH geometry 100; BBH geometric_shapes e navigate inteiros (250 cada)
    TAG=$1; [ -f $R/$TAG/FEITO ] || return 0
    local p falta=""; for p in evalplus gsm8k-ifeval math bbh; do [ -f $R/$TAG/PROVA-$p ] || falta="$falta $p"; done
    [ -z "$falta" ] && { log "pulando as provas de $TAG (ja feitas)"; return 0; }
    log "=== provas $TAG:$falta"
    $S/alternar.sh teste $TAG >> $R/$TAG/partida-provas.log 2>&1 || { log "ERRO: $TAG nao subiu para as provas"; return 1; }
    for p in $falta; do
        log "  $TAG: prova $p"
        case $p in
            evalplus) M "evalplus $TAG" 10800 ;;
            gsm8k-ifeval) M "lmeval $TAG gsm8k,ifeval 100" 10800 ;;
            math) M "lmeval $TAG minerva_math_geometry 100 max_gen_toks=4096 lm-eval-espacial" 10800 ;;
            bbh) M "lmeval $TAG bbh_cot_zeroshot_geometric_shapes,bbh_cot_zeroshot_navigate 250 max_gen_toks=4096 lm-eval-bbh" 14400 ;;
        esac
        touch $R/$TAG/PROVA-$p
    done
    $S/alternar.sh parar-teste >/dev/null; log "  provas de $TAG concluidas"
}

log "inicio (BATERIA3D=${BATERIA3D:-0} LONGO=${LONGO:-nenhum} VOLTAR=${VOLTAR:-1})"; mkdir -p $R $T/estado
[ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2048 ] && { log "ERRO: a GPU nao esta livre; pare o Maya/Strata antes"; trap - EXIT; exit 1; }
$PY $S/gerar-configs.py ${LONGO:+--longo $LONGO} > /dev/null
[ -e $R/q8_0-fp16-256k-0142 ] || ln -s <data>/testes-q6-q8/resultados/q8_0-fp16-256k $R/q8_0-fp16-256k-0142

# (1) os 6 modelos na 0.2.0, do mais leve ao mais pesado
declare -A CURTO=([ud-q2_k_xl]=q2 [ud-q3_k_xl]=q3 [ud-q4_k_xl]=q4 [ud-q5_k_xl]=q5 [ud-q6_k_xl]=q6 [q8_0]=q8)
for m in ud-q2_k_xl ud-q3_k_xl ud-q4_k_xl ud-q5_k_xl ud-q6_k_xl q8_0; do
    if [ $m = ud-q5_k_xl ]; then   # o download do Q5 pode ainda estar rodando: espera os 6 .ok (ate 6 h)
        n=0; while [ $(ls $T/modelos/UD-Q5_K_XL/*.gguf.ok 2>/dev/null | wc -l) -lt 6 ] && [ $n -lt 21600 ]; do
            [ $n = 0 ] && log "  esperando o download do UD-Q5_K_XL"; sleep 60; n=$((n+60)); done
    fi
    [ -f $T/packs/$m/PRONTO ] || { log "  pack $m"; $S/empacotar.sh ${CURTO[$m]} >> $T/logs/pipeline-packs.log 2>&1; }
    [ -f $T/packs/$m/PRONTO ] || { log "AVISO: sem pack de $m (ver logs/pack-$m.log); pulando"; continue; }
    extra=""; case $m in q8_0|ud-q4_k_xl) extra="visao-detalhe" ;; esac
    roda 020-$m-256k "info" bench "velocidade 8k,32k,128k" "kl-coletar geral" "kl-coletar dominios" "agulhas 128k" $extra 3d
    provas 020-$m-256k
done
# (0) referencia: Q8_0 no 0.1.45 de producao
roda ref0145-q8_0-256k "info" bench "velocidade 8k,32k,128k" "kl-coletar geral" "kl-coletar dominios" "visao-detalhe"
provas ref0145-q8_0-256k
# (2) 1M de contexto (as opcoes da producao): so confere que sobe e responde
[ -n "${LONGO:-}" ] && roda 020-$LONGO-1m "info" "velocidade 8k"
# (3) KL e resumo
TAG=_kl; mkdir -p $R/_kl
M "kl-comparar ref0145-q8_0-256k $(cd $R && ls -d 020-*-256k 2>/dev/null | tr '\n' ' ')"
M "kl-comparar q8_0-fp16-256k-0142 ref0145-q8_0-256k 020-q8_0-256k"
TAG=_resumo; mkdir -p $R/_resumo; M "resumo"
log "fim"
