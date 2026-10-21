#!/bin/bash
# pipeline.sh - bateria Strata 0.2.1 (motor 0.1.46, control 0.1.46.2) x 6 quantizacoes (UD-Q2..Q6_K_XL, Q8_0) e a bateria 3D com o
# Blender da VM. Copia adaptada de testes-020/scripts/pipeline.sh: mesmos configs, mesmo metodo; sem BBH (nao separou os modelos
# no 0.2.0) e sem a referencia 0.1.45. Retomavel: resultados/<tag>/FEITO e PROVA-<parte> sao pulados.
#   antes: a GPU livre. Depois: setsid nohup <data>/testes-021/scripts/pipeline.sh > logs/pipeline.log 2>&1 &
# Ordem: (1) os 6 modelos do mais leve ao mais pesado: testes rapidos + 3D, depois as provas (HumanEval+, GSM8K 4096, IFEval,
# MATH geometry); (2) UD-Q4_K_XL em partidas alternadas 0.2.0/0.2.1 (3 pares, 8K e 32K); (3) KL e resumo; no fim, gpu-start strata.
# PRAZO=<epoch>: depois dele nao comeca nada novo (pula o resto e sobe a producao). VOLTAR=0 nao sobe a producao.
set -u
T=<data>/testes-021; S=$T/scripts; R=$T/resultados; PY=<data>/strata/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 TESTES_DIR=$T STRATA_SRC=<data>/testes-q6-q8/strata-src
BENCH=<data>/strata-0145/strata-src/tools/strata_bench.py
log() { echo "$(date '+%F %T') [pipeline-021] $*" | tee -a $T/logs/pipeline-eventos.log; }
M() { timeout "${2:-7200}" $PY $S/medir.py $1 >> $R/$TAG/medir.log 2>&1 || log "AVISO: medir $1 falhou em $TAG (rc $?)"; }
no_prazo() { [ "$(date +%s)" -lt "${PRAZO:-9999999999}" ] || { log "PRAZO: pulando $1"; return 1; }; }

volta() {
    $S/alternar.sh parar-teste >/dev/null 2>&1
    if [ "${VOLTAR:-1}" = 1 ]; then log "fim/interrupcao: subindo a producao (gpu-start strata)"; <data>/gpu-services/gpu-start strata >> $T/logs/pipeline-eventos.log 2>&1
    else log "fim/interrupcao: a producao NAO foi iniciada (VOLTAR=0)"; fi
}
trap volta EXIT
trap 'log "sinal recebido"; exit 1' INT TERM

sobe() {   # sobe <config> [env]   sobe o servidor de teste; 1 se falhar
    STRATA_ENV="${2:-}" $S/alternar.sh teste $1 >> $R/$TAG/partida.log 2>&1 && return 0
    log "ERRO: $1 nao subiu (ver $R/$TAG/partida.log e logs/servidor-$1.log)"; echo "falhou na partida $(date '+%F %T')" > $R/$TAG/FALHOU; return 1
}

roda() {   # roda <tag> <medidas...>   ("bench" e "3d" sao do pipeline; o resto vai para o medir.py)
    TAG=$1; shift
    [ -f $R/$TAG/FEITO ] && { log "pulando $TAG (ja feito)"; return 0; }
    no_prazo $TAG || return 1
    mkdir -p $R/$TAG/logits; log "=== $TAG"
    sobe $TAG "STRATA_DUMP_FIRST_LOGITS=$R/$TAG/logits/l" || return 1
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > $R/$TAG/vram-pronto.txt; free -g | sed -n 2p > $R/$TAG/ram-pronto.txt
    local med; for med in "$@"; do
        log "  $TAG: $med"
        case "$med" in
            bench) STRATA_KEY=$(cat $T/configs/.api_key-teste) timeout 3600 $PY $BENCH --url http://127.0.0.1:8090 --runs 3 \
                       --out $R/$TAG/bench --quiet >> $R/$TAG/bench.log 2>&1 || log "AVISO: bench falhou em $TAG" ;;
            3d) timeout 10800 $PY $S/bench3d.py rodar $TAG >> $R/$TAG/bench3d.log 2>&1 || log "AVISO: 3d falhou em $TAG"
                log "  $TAG: $(tail -1 $R/$TAG/bench3d.log)" ;;
            *) set -- $med; local cmd=$1; shift; M "$cmd $TAG $*"; set -- ;;
        esac
    done
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > $R/$TAG/vram-final.txt; free -g | sed -n 2p > $R/$TAG/ram-final.txt
    $S/alternar.sh parar-teste >/dev/null; touch $R/$TAG/FEITO; log "  $TAG concluido"
}

provas() {   # provas <tag>: sobe de novo (sem gravar logits) e roda as partes que faltam (marcadores PROVA-<parte>):
             # HumanEval+ 164; GSM8K 100 com 4096 tokens e IFEval 100 (o medir.py separa os dois); MATH geometry 100
    TAG=$1; [ -f $R/$TAG/FEITO ] || return 0
    local p falta=""; for p in evalplus gsm8k-ifeval math; do [ -f $R/$TAG/PROVA-$p ] || falta="$falta $p"; done
    [ -z "$falta" ] && { log "pulando as provas de $TAG (ja feitas)"; return 0; }
    no_prazo "provas $TAG" || return 1
    log "=== provas $TAG:$falta"
    $S/alternar.sh teste $TAG >> $R/$TAG/partida-provas.log 2>&1 || { log "ERRO: $TAG nao subiu para as provas"; return 1; }
    for p in $falta; do
        no_prazo "prova $p de $TAG" || break
        log "  $TAG: prova $p"
        case $p in
            evalplus) M "evalplus $TAG" 10800 ;;
            gsm8k-ifeval) M "lmeval $TAG gsm8k,ifeval 100" 10800 ;;
            math) M "lmeval $TAG minerva_math_geometry 100 max_gen_toks=4096 lm-eval-espacial" 10800 ;;
        esac
        touch $R/$TAG/PROVA-$p
    done
    $S/alternar.sh parar-teste >/dev/null; log "  provas de $TAG concluidas"
}

log "inicio (PRAZO=$(date -d @${PRAZO:-9999999999} '+%F %T' 2>/dev/null) VOLTAR=${VOLTAR:-1})"; mkdir -p $R $T/estado
[ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2048 ] && { log "ERRO: a GPU nao esta livre"; trap - EXIT; exit 1; }
$PY $S/gerar-configs.py > /dev/null
for d in <data>/testes-020/resultados/020-*; do [ -e $R/$(basename $d) ] || ln -s $d $R/; done   # para o KL contra o 0.2.0

# (1) os 6 modelos no 0.2.1
for m in ud-q2_k_xl ud-q3_k_xl ud-q4_k_xl ud-q5_k_xl ud-q6_k_xl q8_0; do
    extra=""; case $m in q8_0|ud-q4_k_xl) extra="visao-detalhe" ;; esac
    roda 021-$m-256k "info" bench "velocidade 8k,32k,128k" "kl-coletar geral" "kl-coletar dominios" "agulhas 128k" $extra 3d
    provas 021-$m-256k
done
# (2) partidas alternadas 0.2.0 / 0.2.1, UD-Q4_K_XL, 3 pares
for i in 1 2 3; do
    for v in 020 021; do
        TAG=ab$v-q4-r$i; mkdir -p $R/$TAG
        [ -f $R/$TAG/FEITO ] && continue
        no_prazo $TAG || break 2
        log "=== $TAG"; sobe ab$v-q4 || continue
        M "info $TAG"; M "velocidade $TAG 8k,32k"
        $S/alternar.sh parar-teste >/dev/null; touch $R/$TAG/FEITO
    done
done
# (3) KL e resumo
TAG=_kl; mkdir -p $R/_kl
M "kl-comparar 021-q8_0-256k 021-ud-q6_k_xl-256k 021-ud-q5_k_xl-256k 021-ud-q4_k_xl-256k 021-ud-q3_k_xl-256k 021-ud-q2_k_xl-256k"
KL_MAPA=kl-mapa-dominios.json M "kl-comparar 021-q8_0-256k 021-ud-q6_k_xl-256k 021-ud-q5_k_xl-256k 021-ud-q4_k_xl-256k 021-ud-q3_k_xl-256k 021-ud-q2_k_xl-256k"
M "kl-comparar 020-q8_0-256k 021-q8_0-256k 021-ud-q4_k_xl-256k 021-ud-q2_k_xl-256k"
KL_MAPA=kl-mapa-dominios.json M "kl-comparar 020-q8_0-256k 021-q8_0-256k"
log "fim"
