#!/bin/bash
# Long-context and log-scale checks of strata-ink2ref against the fp32 oracle. Log: /var/ramdisk/inkling/long.log
set -x
V=/var/cheapssd/inkling-oracle; P="$V/.venv/bin/python -I"; D=/var/fastasfuck/inkling2/kl; R=/var/fastasfuck/inkling2/bin/strata-ink2ref; M=/var/fastasfuck/inkling2/Inkling-2-Mini
$P $V/kl_run.py --dtype float32 --mode a16 --text $D/long.txt --max-tokens 1600 --out $D/long_fp32.pt --ids-out $D/long.ids
$P $V/kl_run.py --dtype float32 --mode a16 --text $D/text.txt --max-tokens 48 --pos-offset 20000 --out $D/off_fp32.pt --ids-out $D/off.ids
$R --model $M --tokens-file $D/off.ids --n 0 --pos-offset 20000 --dump-logits $D/off.ref.logits
$P $V/cmp_logits.py $D/off_fp32.pt $D/off.ref.logits
$R --model $M --tokens-file $D/off.ids --n 0 --dump-logits $D/off0.ref.logits
$P $V/cmp_logits.py $D/off_fp32.pt $D/off0.ref.logits
$R --model $M --tokens-file $D/long.ids --n 0 --dump-logits $D/long.ref.logits
$P $V/cmp_logits.py $D/long_fp32.pt $D/long.ref.logits
echo LONG-CHECK-DONE
