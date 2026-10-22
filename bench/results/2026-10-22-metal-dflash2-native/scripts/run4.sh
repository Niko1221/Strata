#!/bin/zsh
# run4.sh: wait until no other engine/bench runs for 10 min, then matrix4.py, then greedy identity (MTP vs DFlash2, one binary)
cd <data>/dflash2/bench
busy() { pgrep -f "run.py|matrix-v2|matrix3|ladder.sh|qtool.py|train.py" >/dev/null || ps -Ao command | grep -v grep | grep -q -- "strata.* --serve"; }
q=0; while [ $q -lt 1 ]; do if busy; then q=0; else q=$((q+1)); fi; sleep 60; done
echo "idle, start $(date)"
caffeinate -i python3 matrix4.py > matrix4.log 2>&1
for c in v4-mtp-s3-a42 v4-df-s4-a42; do
  [ "$(ps -Ao command | grep -v grep | grep -c -- "strata.* --serve")" -le 1 ] || echo "WARN another engine running before $c"
  ./swap4.sh $PWD/cfg-$c.json && caffeinate -i ../../../Strata/.venv/bin/python run.py 8092 0 greedy4-$c.json > greedy4-$c.log 2>&1
done
kill $(cat swap4.pid) 2>/dev/null
python3 - <<'P'
import json
a=json.load(open('greedy4-v4-mtp-s3-a42.json')); b=json.load(open('greedy4-v4-df-s4-a42.json'))
same=[x==y for x,y in zip(a,b)]
print(f"greedy identical {sum(same)} / {len(same)}", [i for i,s in enumerate(same) if not s])
P
echo "done $(date)"; cat matrix-v4.tsv
# give the user's own server back (as it ran before the benchmark)
cd <strata> && nohup python3 -m serve.server --engine strata --config strata-dflash.json --port 8080 > <data>/dflash2/bench/user-server-restart.out 2>&1 &
echo "user server restarted on 8080 $(date)"
