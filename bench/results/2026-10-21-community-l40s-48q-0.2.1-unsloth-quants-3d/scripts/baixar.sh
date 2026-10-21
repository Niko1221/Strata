#!/bin/bash
# baixar.sh - baixa o Strata 0.2.1 (Terminal/servidor 0.1.46.2 e motor CUDA 12 0.1.46) para <data>/strata-021.
# Só baixa (curl, release pública), confere o SHA-256 contra o SHA256SUMS.txt da versão e descompacta. Não usa a GPU,
# não mexe na produção (<data>/strata, strata-0145), no 0.2.0 (strata-020) nem no Maya.
set -eu
T=<data>/strata-021; U=https://github.com/Niko1221/Strata/releases/download/v0.2.1
C=strata-control-0.1.46.2-linux-x86_64.tar.gz; E=strata-engine-0.1.46-linux-x86_64-cuda-cuda12.tar.gz
mkdir -p $T/download $T/control $T/engine $T/logs
cd $T/download
for f in SHA256SUMS.txt $C $E; do curl -fsSL --retry 5 -o $f "$U/$f"; done
grep -E "  ?\*?($C|$E)$" SHA256SUMS.txt | sed 's/ \*/  /' | sha256sum -c -
[ "$(grep -cE "($C|$E)$" SHA256SUMS.txt)" = 2 ] || { echo "ERRO: os dois pacotes nao estao no SHA256SUMS.txt"; exit 1; }
tar -xzf $C -C $T/control
tar -xzf $E -C $T/engine
echo "$(date '+%F %T') FIM: baixado, conferido e descompactado" | tee -a $T/logs/download.log
