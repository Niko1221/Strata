#!/bin/bash
# baixar.sh - baixa UD-Q2_K_XL, UD-Q3_K_XL e UD-Q5_K_XL (Unsloth, revisao fixa 766911a, a mesma dos Q4/Q6/Q8 que ja temos)
# para <data>/testes-020/modelos/, com retomada e SHA-256 (manifesto.txt: sha, bytes, caminho).
# As partes 2 e 3 do UD-Q5_K_XL sao identicas (mesmo SHA-256) as do UD-Q6_K_XL ja baixado: viram hardlink (~51 GiB a menos).
# Nao usa a GPU. Rodar de novo e seguro: o que tem .ok e pulado.
#   setsid nohup <data>/testes-020/scripts/baixar.sh >/dev/null 2>&1 &      acompanhar: tail logs/download.log
set -u
T=<data>/testes-020; REV=766911a6b7369840a91dbcd95f9f997acaab6cd6
BASE="https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/$REV"
Q6=<data>/testes-q6-q8/modelos/UD-Q6_K_XL
LOG=$T/logs/download.log
exec 9>$T/logs/download.lock; flock -n 9 || { echo "$(date '+%F %T') outra copia ja esta rodando" >> $LOG; exit 0; }
log() { echo "$(date '+%F %T') $*" >> $LOG; }
log "inicio"
while read -r sha size path; do
    dst=$T/modelos/$path; mkdir -p "$(dirname "$dst")"
    [ -f "$dst.ok" ] && continue
    gemeo=$Q6/$(basename "$path" | sed 's/UD-Q5_K_XL/UD-Q6_K_XL/')
    if [ -f "$gemeo.ok" ] || { [ -f "$gemeo" ] && [ "$(stat -c %s "$gemeo")" = "$size" ]; }; then
        if [ "$(sha256sum "$gemeo" | cut -d' ' -f1)" = "$sha" ]; then
            ln -f "$gemeo" "$dst" && echo "$sha" > "$dst.ok" && log "OK (hardlink do UD-Q6_K_XL) $path" && continue
        fi
    fi
    for tent in 1 2 3 4 5 6 7 8 9 10; do
        log "baixando $path (tentativa $tent)"
        curl -fsSL -C - --retry 5 --retry-delay 30 -o "$dst.part" "$BASE/$path" 2>>$LOG
        [ "$(stat -c %s "$dst.part" 2>/dev/null || echo 0)" = "$size" ] && break
        sleep 60
    done
    if [ "$(sha256sum "$dst.part" | cut -d' ' -f1)" = "$sha" ]; then
        mv "$dst.part" "$dst"; echo "$sha" > "$dst.ok"; log "OK $path"
    else
        log "ERRO: SHA-256 nao confere em $path (arquivo .part mantido para conferir)"
    fi
done < $T/scripts/manifesto.txt
n=$(ls $T/modelos/*/*.ok 2>/dev/null | wc -l)
log "FIM: $n de $(wc -l < $T/scripts/manifesto.txt) arquivos conferidos; livre em /dados: $(df -h /dados | awk 'NR==2{print $4}')"
