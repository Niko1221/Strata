#!/bin/bash
# alternar.sh - sobe e para o servidor de TESTE (porta 8090) da bateria 0.2.0. Copia adaptada de strata-0145/scripts/alternar.sh.
#   alternar.sh status
#   alternar.sh teste <tag>      sobe configs/strata-teste-<tag>.json na 8090 e espera "ready" (a GPU precisa estar livre)
#   alternar.sh parar-teste      para o servidor de teste (PID exato, SIGTERM) e confere que a GPU ficou livre
# Variaveis do motor para um teste: STRATA_ENV='STRATA_DUMP_FIRST_LOGITS=...' alternar.sh teste <tag>
# O servidor certo vem do "cwd" do config: a 0.2.0 usa o Python dela; a referencia 0.1.45, o venv da producao.
set -u
T=<data>/testes-020
MOTORES="^(<data>/strata-020/engine|<data>/strata-0145/engine)/strata(-vision)? "
log() { echo "$(date '+%F %T') [alternar] $*" | tee -a $T/logs/eventos.log; }
pid_teste() { pgrep -f "serve/server.py .*--config $T/configs/" | head -1; }
gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
para() {   # PID exato, SIGTERM, espera ate 180 s; depois termina motores orfaos (PID exato)
    local p=$1 n=0; kill -TERM "$p"; while kill -0 "$p" 2>/dev/null && [ $n -lt 180 ]; do sleep 1; n=$((n+1)); done
    kill -0 "$p" 2>/dev/null && { log "ERRO: PID $p nao parou em 180 s"; return 1; }
    n=0; while pgrep -f "$MOTORES" >/dev/null && [ $n -lt 60 ]; do sleep 1; n=$((n+1)); done
    for e in $(pgrep -f "$MOTORES"); do log "motor orfao (PID $e): TERM"; kill -TERM $e; done
    sleep 5; log "GPU depois de parar: $(gpu_mib) MiB"; return 0
}
espera_ready() {   # $1 = log do servidor, $2 = byte onde esta partida comecou
    local n=0 novo; while [ $n -lt 1800 ]; do
        novo=$(tail -c +$(( $2 + 1 )) "$1" 2>/dev/null)
        grep -q "^ready:" <<< "$novo" && { grep -E "expert cache|MiB of VRAM free|KV streaming|^ready:" <<< "$novo" | tail -6 | cut -c1-220; return 0; }
        grep -q -E "Traceback|engine (died|exited|stopped)|out of memory" <<< "$novo" && { tail -25 <<< "$novo" | cut -c1-250; return 1; }
        sleep 5; n=$((n+5)); done
    log "ERRO: sem 'ready' em 30 min"; return 1
}
case "${1:-status}" in
status)
    echo "teste (8090): $(pid_teste || true)   GPU: $(gpu_mib) MiB"; free -g | sed -n 2p ;;
teste)
    tag=${2:?tag}; cfg=$T/configs/strata-teste-$tag.json; [ -f "$cfg" ] || { echo "nao existe $cfg"; exit 1; }
    [ -n "$(pid_teste)" ] && { echo "ja ha um teste rodando: 'alternar.sh parar-teste'"; exit 1; }
    [ "$(gpu_mib)" -gt 2048 ] && { log "ERRO: a GPU nao esta livre ($(gpu_mib) MiB): pare o Maya/Strata antes (gpu-stop ...)"; exit 1; }
    src=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['cwd'])" "$cfg")
    case "$src" in <data>/strata-020/*) PY=<data>/strata-020/control/python/bin/python3 ;;
                   *) PY=<data>/strata/.venv/bin/python ;; esac
    sl=$T/logs/servidor-$tag.log; echo "=== $(date '+%F %T') inicio $tag" >> "$sl"; off=$(stat -c %s "$sl")
    log "subindo $tag ($PY)"
    (cd "$src" && env PYTHONDONTWRITEBYTECODE=1 ${STRATA_ENV:-} setsid nohup $PY serve/server.py --engine strata --config "$cfg" --port 8090 >> "$sl" 2>&1 < /dev/null &)
    t0=$(date +%s); sleep 3
    if espera_ready "$sl" "$off"; then log "$tag pronto em $(( $(date +%s) - t0 )) s"; echo "$(( $(date +%s) - t0 ))" > $T/logs/partida-$tag.s
    else log "ERRO: $tag nao subiu"; p=$(pid_teste); [ -n "$p" ] && para "$p"; exit 1; fi ;;
parar-teste)
    p=$(pid_teste); [ -z "$p" ] && { log "nenhum teste rodando"; exit 0; }
    log "parando o teste (PID $p)"; para "$p" && log "teste parado" ;;
*) sed -n '2,7p' "$0" ;;
esac
