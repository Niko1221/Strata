"""Archive completed profiles from run_full_screen.py; preserve all token IDs.

Run: python3 collect_full_screen.py /path/to/dflash2-opt-20261026
Only zero-exit completed profiles are copied. An independent check compares every
sequence against the common target-only control. Local paths are normalized;
original JSON hashes are recorded before that transformation.
"""
from pathlib import Path
import argparse,json,hashlib,re,statistics


def encode(value):
    text=json.dumps(value,indent=2)
    return re.sub(r'\[\s*((?:-?\d+,\s*)+-?\d+)\s*\]',
                  lambda m:'['+' '.join(m[1].split())+']',text)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    args=parser.parse_args()
    src=args.source.resolve()
    dst=Path(__file__).resolve().parent/'whole-model'
    dst.mkdir(exist_ok=True)
    manifest=json.loads((src/'full-screen-manifest.json').read_text())
    reference=json.loads((src/'control/mtp.json').read_text())
    controls={}
    for row in reference['rows']:
        assert row['control'] and row['offered']==row['accepted']==0
        assert row['tokens']==controls.setdefault(row['task'],row['tokens'])
    archived=[];summary={}
    for record in manifest:
        if record['returncode']!=0:continue
        label=record['name'];mode='mtp' if label.startswith('mtp') or label=='control' else 'dflash2-il'
        path=src/label/(mode+'.json');raw=path.read_bytes();data=json.loads(raw)
        assert data['prompt_sha256']==reference['prompt_sha256']
        assert data['expert_profile_sha256']==reference['expert_profile_sha256']
        for key in ('context','kv','kv_resident','expert_slots','expert_slots_primary','pool_workers','spec','pcie_frac'):
            assert data['info'][key]==reference['info'][key],(label,key)
        for row in data['rows']:
            assert row['tokens']==controls[row['task']],(label,row['task'])
        normalized=raw.decode().replace(str(src.parent),'${STRATA_DATA}')
        normalized=normalized.replace('/home/jakub/.codex/worktrees/dflash2-opt-20261026/Strata','${REPO}')
        normalized=normalized.replace('/home/jakub/workspace/deepseek-harness/harnes_workspace/Strata','${REPO}')
        (dst/(label+'.json')).write_text(encode(json.loads(normalized)))
        groups={}
        for row in data['rows']:
            if row['rep']<0 or row['control']:continue
            group=groups.setdefault(row['task'],{'tok_s':[],'decode_ms':[],'prompt_ms':[],'wall_ms':[],'accepted_offered':[]})
            group['tok_s'].append(row['tok_s']);group['decode_ms'].append(row['decode_ms'])
            group['prompt_ms'].append(float(row['done'].split()[3]));group['wall_ms'].append(row['wall_ms'])
            group['accepted_offered'].append([row['accepted'],row['offered']])
        for group in groups.values():
            group['median_tok_s']=statistics.median(group['tok_s'])
            group['range_tok_s']=[min(group['tok_s']),max(group['tok_s'])]
            group['median_prompt_ms']=statistics.median(group['prompt_ms'])
        summary[label]=groups
        archived.append({'label':label,'original_sha256':hashlib.sha256(raw).hexdigest(),
                         'engine_sha256':data['sha256'],'validation':data['validation'],
                         'start_utc':record['start_utc'],'end_utc':record['end_utc']})
    (dst/'summary.json').write_text(encode(summary))
    (dst/'manifest.json').write_text(encode(archived))
    print('PASS: common target-only tokens and placement;',len(archived),'profiles archived')


if __name__=='__main__':main()
