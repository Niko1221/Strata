#!/bin/bash
# pair-run.sh TESTNAME - one test: BASE arm then one-knob VARIANT arm, sequential, one instance at a time.
set -u
T=$1
DIR=~/Strata/ab-dense-f32
echo "######## TEST $T: base -> $T (STRATA_DENSE_F32=1 in both; only the one knob differs) ########"
bash $DIR/arm-run.sh $DIR/configs/base.json "${T}-base" || exit 1
bash $DIR/arm-run.sh $DIR/configs/$T.json "${T}-var" || exit 1
echo "######## TEST $T done ########"
