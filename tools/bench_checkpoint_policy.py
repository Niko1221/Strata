"""Deterministic policy comparison. Modeled reconstruction seconds, not GPU timings."""
import argparse
import json
from pathlib import Path
import random
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from serve.checkpoint_store import CheckpointStore


def traces():
    rng=random.Random(731)
    hot=[0]*20 + [x for cold in range(1,31) for x in (cold,0,0)]
    phase=[0]*20 + [x for cold in range(1,21) for x in (cold,0)]
    phase += [40]*20+[x for cold in range(41,61) for x in (cold,40)]
    mixed=[rng.choice([0,0,0,1,1,2,3,4,5,6]) for _ in range(150)]
    return dict(hot_prefix_scan=hot, demand_shift=phase, held_out_mix=mixed)


def run(policy,trace):
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        now=[0]
        store=CheckpointStore(root/'cache',budget_bytes=3*4096,reserve_bytes=0,policy=policy,
                              clock=lambda:now[0],half_life_s=10)
        cost=0
        hits=0
        for q in trace:
            now[0]+=1
            tokens=[q]*32
            full_cost=10 if q in (0,40) else 1
            row,t=store.best('f',tokens,full_cost)
            cost+=t
            if row:
                hits+=1
                store.restored(row['id'],.1)
            store.observe('f',tokens,full_cost)
            source=root/'session'
            source.write_bytes(bytes([q])*4096)
            store.admit(source,'f',tokens,restore_s=.1)
        result=dict(policy=policy,requests=len(trace),modeled_reconstruction_s=round(cost,3),hits=hits,
                    physical_cache_bytes=store.used(),decisions=store.db.execute('SELECT count(*) FROM decisions').fetchone()[0])
        store.close()
        return result


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result={name:[run(policy,trace) for policy in ('fifo','lru','utility')] for name,trace in traces().items()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
