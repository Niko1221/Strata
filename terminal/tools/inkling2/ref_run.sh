#!/bin/bash
# ref_run.sh PROMPT: run strata-ink2ref on an oracle.py prompt's ids and compare its per-layer outputs and logits.
# Environment: PY (the oracle venv's python), ORACLE (oracle.py --out directory), WORK (dump directory), REF (the
# strata-ink2ref binary), MODEL (the Inkling-2-Mini checkpoint directory); N generated tokens (16)
: "${PY:?}" "${ORACLE:?}" "${WORK:?}" "${REF:?}" "${MODEL:?}"
T=$(dirname "$0"); O=$ORACLE/$1.safetensors; D=$WORK; mkdir -p $D
ids=$($PY -I -c "from safetensors.torch import load_file; print(\",\".join(map(str, load_file(\"$O\")[\"ids\"].tolist())))")
$REF --model $MODEL --tokens $ids --n ${N:-16} --dump-layers $D/$1.layers --dump-logits $D/$1.logits
$PY -I $T/ref_cmp.py $O $D/$1.layers $D/$1.logits
grep -o "\"name\": \"$1\".*\"gen_ids\": \[[0-9, ]*\]" $ORACLE/summary.json | grep -o "gen_ids.*"
