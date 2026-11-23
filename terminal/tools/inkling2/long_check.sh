#!/bin/bash
# long_check.sh: long-context and LogScale checks of strata-ink2ref against the fp32 transformers oracle.
# Environment: PY (the oracle venv's python, see setup-venv.sh), WORK (a directory holding long.txt and text.txt, the
# outputs land there), REF (the strata-ink2ref binary), MODEL (the Inkling-2-Mini checkpoint directory)
set -x
: "${PY:?}" "${WORK:?}" "${REF:?}" "${MODEL:?}"
T=$(dirname "$0"); P="$PY -I"; D=$WORK
$P $T/kl_run.py --model $MODEL --dtype float32 --mode a16 --text $D/long.txt --max-tokens 1600 --out $D/long_fp32.pt --ids-out $D/long.ids
$P $T/kl_run.py --model $MODEL --dtype float32 --mode a16 --text $D/text.txt --max-tokens 48 --pos-offset 20000 --out $D/off_fp32.pt --ids-out $D/off.ids
$REF --model $MODEL --tokens-file $D/off.ids --n 0 --pos-offset 20000 --dump-logits $D/off.ref.logits
$P $T/cmp_logits.py $D/off_fp32.pt $D/off.ref.logits
$REF --model $MODEL --tokens-file $D/off.ids --n 0 --dump-logits $D/off0.ref.logits
$P $T/cmp_logits.py $D/off_fp32.pt $D/off0.ref.logits
$REF --model $MODEL --tokens-file $D/long.ids --n 0 --dump-logits $D/long.ref.logits
$P $T/cmp_logits.py $D/long_fp32.pt $D/long.ref.logits
echo LONG-CHECK-DONE
