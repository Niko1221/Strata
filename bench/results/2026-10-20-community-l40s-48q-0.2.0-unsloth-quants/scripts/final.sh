#!/bin/bash
# final.sh - o que falta para o artigo (pedido do dono em 2026-10-20). Roda solto (setsid nohup), sobrevive a queda do SSH.
#  1) para a producao; 2) referencia Q8_0 no 0.1.45 (testes rapidos + provas), KL e resumo (pipeline.sh, VOLTAR=0);
#  3) A/B curto do STRATA_ROUTE_TAIL_SKIP: UD-Q4_K_XL no 0.2.0 com o padrao do motor (7) em vez do 0 que o setup grava
#     (velocidade, strata_bench, HumanEval+ e GSM8K 4096) - explica a diferenca de velocidade contra o 0.1.45 antigo;
#  4) KL dos 160 prompts por assunto; 5) sobe a producao.
set -u
T=<data>/testes-020; S=$T/scripts; R=$T/resultados; PY=<data>/strata/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 TESTES_DIR=$T STRATA_SRC=<data>/testes-q6-q8/strata-src
log() { echo "$(date '+%F %T') [final] $*" | tee -a $T/logs/pipeline-eventos.log; }
M() { timeout "${2:-7200}" $PY $S/medir.py $1 >> $R/$TAG/medir.log 2>&1 || log "AVISO: medir $1 falhou em $TAG (rc $?)"; }
volta() { $S/alternar.sh parar-teste >/dev/null 2>&1; log "subindo a producao (gpu-start strata)"; <data>/gpu-services/gpu-start strata >> $T/logs/pipeline-eventos.log 2>&1; }
trap volta EXIT; trap 'log "sinal recebido"; exit 1' INT TERM

log "inicio: parando a producao para a referencia 0.1.45"
<data>/gpu-services/gpu-stop strata >> $T/logs/pipeline-eventos.log 2>&1
n=0; while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2048 ] && [ $n -lt 600 ]; do sleep 5; n=$((n+5)); done
[ $n -ge 600 ] && { log "ERRO: a GPU nao ficou livre em 10 min"; exit 1; }
D=$R/ref0145-q8_0-256k; [ -f $D/ADIADO ] && rm -f $D/FEITO $D/PROVA-* $D/ADIADO
LONGO=q8_0 BATERIA3D=0 VOLTAR=0 $S/pipeline.sh >> $T/logs/pipeline.log 2>&1
log "pipeline (referencia) terminou com rc $?"

# (3) A/B do tail skip no Q4
TAG=020-ud-q4_k_xl-256k-tailskip7
if [ ! -f $R/$TAG/FEITO ]; then
    python3 - <<'PY'
import json
d="<data>/testes-020/configs/"; c=json.load(open(d+"strata-teste-020-ud-q4_k_xl-256k.json"))
c["env"].pop("STRATA_ROUTE_TAIL_SKIP", None); c["model_name"]+="-tailskip7"; c["log"]=c["log"].replace(".log","-tailskip7.log")
open(d+"strata-teste-020-ud-q4_k_xl-256k-tailskip7.json","w").write(json.dumps(c,indent=1)+"\n")
PY
    chmod 600 $T/configs/strata-teste-$TAG.json; mkdir -p $R/$TAG
    log "=== $TAG (STRATA_ROUTE_TAIL_SKIP no padrao do motor)"
    if $S/alternar.sh teste $TAG >> $R/$TAG/partida.log 2>&1; then
        M "info $TAG"; M "velocidade $TAG 8k,32k,128k"
        STRATA_KEY=$(cat $T/configs/.api_key-teste) timeout 3600 $PY <data>/strata-0145/strata-src/tools/strata_bench.py \
            --url http://127.0.0.1:8090 --runs 3 --out $R/$TAG/bench --quiet >> $R/$TAG/bench.log 2>&1 || log "AVISO: bench falhou em $TAG"
        M "evalplus $TAG" 10800; M "lmeval $TAG gsm8k 100 max_gen_toks=4096 lm-eval-gsm8k-4k" 7200
        grep -h "route tail skip\|TAIL_SKIP" $T/logs/engine-020-ud-q4_k_xl-256k-tailskip7.log | tail -3 > $R/$TAG/tailskip-log.txt
        $S/alternar.sh parar-teste >/dev/null; touch $R/$TAG/FEITO; log "  $TAG concluido"
    else log "ERRO: $TAG nao subiu"; fi
fi

# (4) KL por assunto (160 prompts) e as comparacoes contra o Q8 do 0.2.0
TAG=_kl
M "kl-comparar 020-q8_0-256k 020-ud-q6_k_xl-256k 020-ud-q5_k_xl-256k 020-ud-q4_k_xl-256k 020-ud-q3_k_xl-256k 020-ud-q2_k_xl-256k"
KL_MAPA=kl-mapa-dominios.json M "kl-comparar 020-q8_0-256k 020-ud-q6_k_xl-256k 020-ud-q5_k_xl-256k 020-ud-q4_k_xl-256k 020-ud-q3_k_xl-256k 020-ud-q2_k_xl-256k"
KL_MAPA=kl-mapa-dominios.json M "kl-comparar ref0145-q8_0-256k 020-q8_0-256k"
log "fim"
