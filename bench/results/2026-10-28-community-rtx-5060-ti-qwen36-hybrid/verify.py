"""Check package hashes and recompute all paired statistics without GPU access."""
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import median

HERE=Path(__file__).resolve().parent


def main():
    manifest=json.loads((HERE/'manifest.json').read_text())
    for filename,sha in manifest.items():assert hashlib.sha256((HERE/filename).read_bytes()).hexdigest()==sha,filename
    fixtures=json.loads((HERE/'fixtures/manifest.json').read_text())
    assert sorted(v['prompt_tokens'] for v in fixtures.values())==[1025,16385]
    for case,item in fixtures.items():
        for suffix in ['tokens','forced']:
            path=HERE/'fixtures'/(case+'.'+suffix)
            assert hashlib.sha256(path.read_bytes()).hexdigest()==item[suffix+'_file_sha256']
        assert len((HERE/'fixtures'/(case+'.forced')).read_text().split(','))==96
    tasks=json.loads((HERE/'fixtures/tasks.json').read_text(encoding='utf-8'))
    assert len(tasks)==12
    for task in tasks:
        assert len(task['input_ids'])==task['prompt_tokens']
        assert hashlib.sha256(json.dumps(task['input_ids'],separators=(',',':')).encode()).hexdigest()==task['input_sha256']
    for pr,filename,csvfile in [('pr4335','summary.json','pairs.csv'),('pr4347','fixed-history-summary.json','fixed-history-pairs.csv')]:
        summary=json.loads((HERE/'data'/pr/filename).read_text(encoding='utf-8'))
        runs=json.loads((HERE/'data'/pr/'recorded-runs.json').read_text(encoding='utf-8'))['rows']
        assert len(runs)==20 and all(row['passed'] for row in runs)
        assert all(len(row['output_ids'])==97 for row in runs)
        for row in runs:
            label=(row['label'] if pr=='pr4335' else f"pair-{row['pair']}-{row['case']}-{row['arm']}")
            phase='execution-01' if pr=='pr4335' else 'fixed-history-01'
            log=(HERE/'logs'/pr/phase/(label+'.log')).read_text(encoding='utf-8')
            timing=re.search(r'prompt (\d+) tokens in ([\d.]+) ms \([\d.]+ tok/s\), (\d+) generated in ([\d.]+) ms',log)
            assert timing and int(timing[3])==97
            assert float(timing[2])==row['prefill_ms'] and float(timing[4])==row['generation_ms']
            gen=re.search(r'^gen: ([0-9 ]+)$',log,re.M)
            assert gen and [int(x) for x in gen[1].split()]==row['output_ids']
        pairs=list(csv.DictReader((HERE/'data'/pr/csvfile).open(encoding='utf-8')))
        assert len(pairs)==10 and summary['passed']
        for case in fixtures:
            selected=[row for row in pairs if row['case']==case]
            assert len(selected)==5 and {int(v['pair']) for v in selected}==set(range(5))
            for metric in ['prefill_ms','generation_ms','request_ms']:
                ratios=[]
                for row in selected:
                    left,right=(('c2048_','c4096_') if pr=='pr4335' else ('A_','B_'))
                    ratio=float(row[left+metric])/float(row[right+metric]);ratios.append(ratio)
                    arm_keys=('requested_chunk',2048,4096) if pr=='pr4335' else ('arm','A','B')
                    for prefix,arm in [(left,arm_keys[1]),(right,arm_keys[2])]:
                        raw=next(x for x in runs if x['case']==case and x['pair']==int(row['pair']) and x[arm_keys[0]]==arm)
                        assert abs(raw[metric]-float(row[prefix+metric]))<1e-8
                name='paired_2048_over_4096' if pr=='pr4335' else 'paired_A_over_B'
                recorded=summary['cases'][case]['metrics'][metric][name]
                assert abs(median(ratios)-recorded['median'])<1e-12
                assert abs(min(ratios)-recorded['min'])<1e-12 and abs(max(ratios)-recorded['max'])<1e-12
    dumps=json.loads((HERE/'numeric-dump-manifest.json').read_text())
    for case in fixtures:
        assert dumps['pr4335'][case+'-main2048']['sha256']==dumps['pr4335'][case+'-pr2048']['sha256']
        assert dumps['pr4335'][case+'-main4096']['sha256']==dumps['pr4335'][case+'-pr4096']['sha256']
        assert dumps['pr4335'][case+'-prauto']['sha256']==dumps['pr4335'][case+'-main4096']['sha256']
        assert dumps['pr4347'][case+'-A']['sha256']==dumps['pr4347'][case+'-C']['sha256']
        assert dumps['pr4347'][case+'-B']['sha256']==dumps['pr4347'][case+'-D']['sha256']
    print('PASS: file hashes, fixtures, 40 measured runs/logs, 20 pairs, statistics and matching numeric-dump hashes')


if __name__=='__main__':main()
