#!/bin/bash
# The whole Signal suite: the hash check, then every run in order (see the folder's README.md)
S=<SCRIPTS>; C=<BENCH>/configs
A=$C/setup-ap-q4_k_xl.json; B=$C/setup-ap-iq4_xs.json; H=$C/docs-handbuilt.json; K=$C/docs-handbuilt-kvq4_0.json
ADAPT="adapt:xargs=--adapt-every;1;--adapt-swaps;80;--adapt-decay;0.97"
echo "== verify-hash $(date -Is)"; $S/verify_hash.sh
$S/chain.sh pass1-A-setup-ap-q4_k_xl:$A pass1-C-docs-handbuilt-ap-q4_k_xl:$H pass1-B-setup-ap-iq4_xs:$B \
            pass2-B-setup-ap-iq4_xs:$B pass2-C-docs-handbuilt-ap-q4_k_xl:$H pass2-A-setup-ap-q4_k_xl:$A || exit 2
sleep 8; echo "== ab-adapt-max100 $(date -Is)"; $S/ab_run.sh ab-adapt-max100 100 4 $H default: "$ADAPT" || exit 2
sleep 8; echo "== ab-adapt-max512 $(date -Is)"; $S/ab_run.sh ab-adapt-max512 512 4 $H default: "$ADAPT" || exit 2
sleep 8
$S/chain.sh kv-1-int8:$H kv-1-q4_0:$K kv-2-int8:$H kv-2-q4_0:$K \
            pass3-B-setup-ap-iq4_xs:$B pass3-C-docs-handbuilt-ap-q4_k_xl:$H pass3-C2-docs-handbuilt-ap-q4_k_xl:$H || exit 2
sleep 8; $S/chain.sh pass3-A-setup-ap-q4_k_xl:$A || exit 2
echo SUITE-DONE $(date -Is)
