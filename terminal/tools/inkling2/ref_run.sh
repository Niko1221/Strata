#!/bin/bash
# ref_run.sh PROMPT: run strata-ink2ref on the oracle prompt ids and compare. Log: /var/ramdisk/inkling/ref-PROMPT.log
P=/var/cheapssd/inkling-oracle/.venv/bin/python; O=/var/fastasfuck/inkling2/oracle/fp32/$1.safetensors; D=/var/fastasfuck/inkling2/refdump; mkdir -p $D
ids=$($P -I -c "from safetensors.torch import load_file; print(\",\".join(map(str, load_file(\"$O\")[\"ids\"].tolist())))")
/var/fastasfuck/inkling2/bin/strata-ink2ref --model /var/fastasfuck/inkling2/Inkling-2-Mini --tokens $ids --n ${N:-16} --dump-layers $D/$1.layers --dump-logits $D/$1.logits
$P -I /var/cheapssd/inkling-oracle/ref_cmp.py $O $D/$1.layers $D/$1.logits
grep -o "\"name\": \"$1\".*\"gen_ids\": \[[0-9, ]*\]" /var/fastasfuck/inkling2/oracle/fp32/summary.json | grep -o "gen_ids.*"
