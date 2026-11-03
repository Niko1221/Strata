#!/bin/bash
ulimit -c 0   # a 25 GB core dump on a no-swap box froze the host (10-28)
cd /home/des/test/strata || exit 1
export LD_LIBRARY_PATH=/opt/rocm/lib
export STRATA_ADAPT_OVERLAP=1
export STRATA_PREFILL_MMQ=1
export STRATA_HIPBLASLT_TUNING=/home/des/test/strata/tools/hip/gfx1100-hipblaslt-100500-20261016.txt
export STRATA_PREFILL_RING=96
export STRATA_IO_THREADS=32
set -a; . /tmp/ab_env; set +a
exec /home/des/test/strata/.venv/bin/python -m serve.server --engine strata --config ${AB_CFG:-strata-sc117-iq3_s.json} --port 8089
