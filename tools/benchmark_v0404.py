"""Three paired c4 essay/coding runs against pristine upstream 6674a00.

Uses the unchanged PR #1209 fixture. Each arm starts fresh; engines run serially.
Run only after both builds finish. Raw reports include outputs and receipt times.
"""
from pathlib import Path
import json
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
out = root/'build-validation'
results = []
for workload in ['essay', 'agentic']:
    for pair,seed in enumerate([123,456,789],1):
        for arm in (['baseline','candidate'] if pair%2 else ['candidate','baseline']):
            name=f'{workload}-p{pair}-{arm}'
            cfg=json.loads((out/(arm+'.json')).read_text())
            print('START',name,flush=True)
            with (out/(name+'.runner.log')).open('w') as log:
                proc=subprocess.Popen([sys.executable,str(root/'tools/benchmark_pr_workloads.py'),
                    '--config',str(out/(arm+'.json')),'--exe',cfg['exe'],
                    '--report',str(out/(name+'.json')),'--arm',arm,'--workload',workload,
                    '--seed',str(seed),'--tokens','2048' if workload=='essay' else '3072'],
                    stdout=log,stderr=subprocess.STDOUT)
                try:
                    rc=proc.wait(timeout=780)
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT)
                    raise
            if rc: raise RuntimeError(f'{name} failed: {rc}; see log')
            r=json.loads((out/(name+'.json')).read_text())
            assert r['ok'] and r['exit_code']==0
            row=dict(name=name,arm=arm,workload=workload,pair=pair,seed=seed,
                decode_tps=r['c4_no_admission_tps'],whole_run_tps=r['workflow_tps'],
                output_tokens=r['output_tokens'],
                structure_passes=sum(a['checks']['ok'] for a in r['agents'].values()) if workload=='agentic' else None,
                turn_caps=sum(a['turn_limit_hit'] for a in r['agents'].values()))
            results.append(row)
            (out/'summary.json').write_text(json.dumps(results,indent=2))
            print('DONE',row,flush=True)
