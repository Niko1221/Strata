#!/bin/sh
export PATH=/run/current-system/sw/bin:$PATH
# Runs on mirfak as user strata: sampler + protocol pilot (+concurrency) + needles. Output in /var/lib/strata/pilot-out/
O=/var/lib/strata/pilot-out
set -a; . /run/secrets/rendered/strata.env; set +a
export STRATA_KEY="$STRATA_API_KEY"
/var/lib/strata/sample.sh $O/telemetry.csv &
SP=$!
echo "clients_at_start=$(ss -tn state established '( sport = :8080 )' | tail -n +2 | wc -l)" > $O/context.txt
/var/lib/strata/Strata/.venv/bin/python /var/lib/strata/pilot.py run $O/fixtures.json $O/pilot --concurrency > $O/pilot.log 2>&1
echo "clients_after_pilot=$(ss -tn state established '( sport = :8080 )' | tail -n +2 | wc -l)" >> $O/context.txt
cd /var/lib/strata/Strata
.venv/bin/python tools/needle_bench.py --url http://127.0.0.1:8080 --api-key "$STRATA_API_KEY" --lengths 32k,128k --depths 10,50,90 --out $O/needles.json > $O/needles.log 2>&1
kill $SP
echo done > $O/DONE
