"""Replay the published fixed-history workload. Default: plan only, no GPU work.

This portable entry was written after measurement. See historical-drivers for the
executed drivers; this file is not the source of the archived measurements.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
MODEL_SHA='55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1'


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=['4335','4347'],required=True)
    parser.add_argument('--exe',type=Path,required=True,help='File-argument adapter linked against the pinned PR build')
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New result directory, required only for execution')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--check-recorded',action='store_true',help='Require this machine to match archived slots and argmax IDs')
    args=parser.parse_args()
    fixture=json.loads((HERE/'fixtures/manifest.json').read_text())
    data=HERE/'data'/('pr'+args.experiment)/'recorded-runs.json'
    archived=json.loads(data.read_text(encoding='utf-8'))['rows']
    configs=([('2048',2048,None,63),('4096',4096,None,50)] if args.experiment=='4335'
             else [('keep',2048,True,63),('drop',2048,False,89)])
    tasks=[]
    for pair in range(5):
        cases=list(fixture) if pair%2==0 else list(reversed(fixture))
        for case in cases:
            for arm,chunk,keep,slots in (configs if pair%2==0 else list(reversed(configs))):
                command=[str(args.exe.resolve()),'--native',str(args.model.resolve()),'--max-context','32768',
                         '--prefill',str(chunk),'--pool-workers','7','--mtp','0',
                         '--tokens-file',str(HERE/'fixtures'/(case+'.tokens')),
                         '--force-file',str(HERE/'fixtures'/(case+'.forced'))]
                tasks.append((pair,case,arm,keep,slots,command))
    if not args.execute:
        print(json.dumps({'plan_only':True,'runs':len(tasks),'experiment':args.experiment,
                          'first_command':tasks[0][-1],
                          'requirements':'Build pinned PR, unload other models, pause downloads, check PCIe bandwidth before/after.'},indent=2))
        return
    if digest(args.model)!=MODEL_SHA:raise RuntimeError('Model SHA-256 differs from measured file')
    manifest=json.loads((HERE/'manifest.json').read_text())
    for filename,sha in manifest.items():
        if digest(HERE/filename)!=sha:raise RuntimeError('Package hash mismatch: '+filename)
    args.output.mkdir(parents=True,exist_ok=False)
    result={'experiment':args.experiment,'replay_driver':'post-measurement portable entry',
            'exe_sha256':digest(args.exe),'model_sha256':MODEL_SHA,'rows':[]}
    def save():
        (args.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    save()
    for pair,case,arm,keep,slots,command in tasks:
        env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}
        env.update(STRATA_Q8_PACKED='1',STRATA_MMQ_KQ='1',STRATA_Q36_FIXED_SLOTS='1',
                   CUDA_VISIBLE_DEVICES='0',CUDA_MODULE_LOADING='EAGER')
        env['PATH']=str(args.exe.resolve().parent)+os.pathsep+env.get('PATH','')
        if keep:env['STRATA_Q36_KEEP_PACKED']='1'
        label=f'pair-{pair}-{case}-{arm}'
        log=args.output/(label+'.log')
        with log.open('w',encoding='utf-8') as stream:
            child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
            try:code=child.wait(timeout=900)
            except BaseException:child.kill();child.wait();raise
        text=log.read_text(encoding='utf-8',errors='replace')
        timing=re.search(r'prompt (\d+) tokens in ([\d.]+) ms \([\d.]+ tok/s\), (\d+) generated in ([\d.]+) ms',text)
        placement=re.findall(r'hybrid experts: (\d+) of 256 per layer',text)
        swaps=[int(x) for x in re.findall(r'(\d+) slot replacements',text)]
        output=re.search(r'^gen: ([0-9 ]+)$',text,re.M)
        ids=[int(x) for x in output[1].split()] if output else []
        row={'pair':pair,'case':case,'arm':arm,'exit_code':code,'slots':placement,'replacement_counts':swaps,
             'output_ids':ids,'completed_utc':datetime.now(timezone.utc).isoformat()}
        if timing:row.update(prompt_tokens=int(timing[1]),prefill_ms=float(timing[2]),
                             argmax_evaluations=int(timing[3]),generation_ms=float(timing[4]))
        row['protocol_passed']=bool(code==0 and timing and int(timing[1])==fixture[case]['prompt_tokens']
                                    and int(timing[3])==97 and len(ids)==97 and swaps and not any(swaps))
        if args.check_recorded:
            reference=next(v for v in archived if v['case']==case and
                           (str(v['requested_chunk'])==arm if args.experiment=='4335' else v['arm']==('A' if keep else 'B')))
            row['recorded_match']=ids==reference['output_ids'] and placement==[str(slots)]
            row['protocol_passed']=row['protocol_passed'] and row['recorded_match']
        result['rows'].append(row);save()
        if not row['protocol_passed']:raise RuntimeError('Protocol check failed: '+label)


if __name__=='__main__':main()
