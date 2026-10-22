"""Recompute paired percentage changes and 95% t intervals; standard library only."""
from pathlib import Path
import csv,statistics,json
R=Path(__file__).resolve().parent
rows=list(csv.DictReader((R/'requests.csv').open()))
summary=[]
for workload in ('english','repeat'):
 for temperature in (.6,1.):
  pairs={}
  for row in rows:
   if row['workload']==workload and float(row['temperature'])==temperature:
    assert int(row['tokens'])==256 and row['finish']=='length'
    pairs.setdefault(int(row['block']),{})[row['arm']]=row
  assert len(pairs)==6 and all(set(p)=={'U','A'} for p in pairs.values())
  changes=[100*(float(p['A']['decode_tokens_per_second'])/float(p['U']['decode_tokens_per_second'])-1) for p in pairs.values()]
  mean=statistics.mean(changes);half=2.570581835636305*statistics.stdev(changes)/(len(changes)**.5)
  summary.append(dict(workload=workload,temperature=temperature,pairs=len(changes),baseline_tps=statistics.mean(float(p['U']['decode_tokens_per_second']) for p in pairs.values()),candidate_tps=statistics.mean(float(p['A']['decode_tokens_per_second']) for p in pairs.values()),mean_change_percent=mean,ci95=[mean-half,mean+half],per_block_changes=changes))
print(json.dumps(summary,indent=2))
