#!/bin/bash
# noite.sh - (2026-10-20) espera a bateria 0.1.45/0.2.0 terminar (final.sh, PID exato), para a producao que ela sobe no fim,
# espera a GPU esvaziar e roda a bateria 0.2.1 (pipeline.sh) com prazo ate 2026-10-21 08:30. A producao volta no fim do pipeline.
set -u
T=<data>/testes-021; P=${1:?PID do final.sh}
log() { echo "$(date '+%F %T') [noite] $*" | tee -a $T/logs/pipeline-eventos.log; }
log "esperando o final.sh (PID $P) terminar"
while kill -0 $P 2>/dev/null; do sleep 60; done
log "final.sh terminou; parando a producao para a bateria 0.2.1"
<data>/gpu-services/gpu-stop strata >> $T/logs/pipeline-eventos.log 2>&1
n=0; while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2048 ] && [ $n -lt 600 ]; do sleep 5; n=$((n+5)); done
[ $n -ge 600 ] && { log "ERRO: a GPU nao ficou livre em 10 min; bateria 0.2.1 nao iniciada"; <data>/gpu-services/gpu-start strata >> $T/logs/pipeline-eventos.log 2>&1; exit 1; }
PRAZO=$(date -d "2026-10-21 08:30" +%s) VOLTAR=1 $T/scripts/pipeline.sh >> $T/logs/pipeline.log 2>&1
log "pipeline 0.2.1 terminou (rc $?)"
