"""Measure fixed fresh normal-prefill and token-level paths without fitting tolerances."""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--normal',type=Path,required=True)
    parser.add_argument('--historical',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--lengths',type=int,nargs='+',help='Diagnostic prefix lengths; default: one token and the full retained fixture')
    parser.add_argument('--chunks',type=int,nargs='+',default=[1,31,128],help='Ordinary prefill chunk controls; zero disables prefill')
    args=parser.parse_args()
    if not args.chunks or len(set(args.chunks))!=len(args.chunks) or any(n<0 for n in args.chunks):
        parser.error('Chunks must be distinct nonnegative sizes')
    destination=args.destination.resolve()
    destination.mkdir(parents=True,exist_ok=True)
    if any(destination.iterdir()): raise RuntimeError('Refusing to overwrite measured vectors')
    profile=json.loads(args.profile.read_text())
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from context_time_travel.strata_backend import _validate_execution, _validate_model
    _validate_execution(profile)
    verification=_validate_model(profile)
    historical=json.loads(args.historical.read_text())
    fixture=next(row['exact_token_ids'] for row in historical['progress']['comparisons']
                 if row['max_absolute_error']==historical['max_absolute_error'])
    lengths=args.lengths if args.lengths is not None else [1,len(fixture)]
    if not lengths or len(set(lengths))!=len(lengths) or any(n<1 or n>len(fixture) for n in lengths):
        raise ValueError('Lengths must be distinct positive prefixes of the retained fixture')
    report=dict(schema=1,status='RUNNING',atol=None,rtol=None,
        scope='Fresh arithmetic-path diagnosis only; no branching or numerical admission',
        profile_sha256=sha(args.profile),model_contract_id=profile['model_contract_id'],
        execution_profile_id=profile['execution_profile_id'],model_verification=verification,
        native_build=profile['native_build'],hardware=profile['hardware'],
        historical=dict(path=str(args.historical),sha256=sha(args.historical),
                        max_absolute_error=historical['max_absolute_error']),
        binaries=dict(normal=dict(path=str(args.normal),sha256=sha(args.normal)),
                      token=dict(path=profile['execution']['oracle_command'][0],
                                 sha256=sha(profile['execution']['oracle_command'][0]))),
        environment=profile['execution']['environment'],forwards=[],comparisons=[],vocab=248320)
    def save():
        (destination/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        for length in lengths:
            ids=fixture[:length]
            token_file=destination/('n'+str(length)+'.tokens')
            token_file.write_text(','.join(map(str,ids)))
            vectors={}
            for operator,chunk in [('token',1)]+[('normal',chunk) for chunk in args.chunks]:
                label='n'+str(length)+'-'+operator+'-'+str(chunk)
                logits=destination/(label+'.f32')
                command=list(profile['execution']['oracle_command'])
                if operator=='normal':
                    command[0]=str(args.normal)
                    command[command.index('--prefill')+1]=str(chunk)
                command+=['--tokens-file',str(token_file)]
                environment=dict(os.environ,**profile['execution']['environment'],
                                 STRATA_DUMP_FIRST_LOGITS=str(logits))
                start=time.monotonic()
                result=subprocess.run(command,env=environment,capture_output=True,timeout=900)
                (destination/(label+'.stdout')).write_bytes(result.stdout)
                (destination/(label+'.stderr')).write_bytes(result.stderr)
                row=dict(operator=operator,length=length,chunk=chunk,command=command,
                    exit_code=result.returncode,seconds=time.monotonic()-start,
                    logits_file=logits.name,stdout_file=label+'.stdout',stderr_file=label+'.stderr')
                report['forwards'].append(row)
                if result.returncode: raise RuntimeError(label+' failed: '+result.stderr.decode(errors='replace')[-2000:])
                if operator=='token':
                    work=json.loads(result.stdout)
                    if work['consumed']!=length or work['position_max']!=length-1:
                        raise RuntimeError('Fresh token-level frontier is incomplete')
                    row.update(computed_tokens=work['forward_tokens'],position_max=work['position_max'],windows=work['windows'])
                else:
                    match=re.search(rb'strata forward: prefill_tokens=(\d+) verifier_tokens=(\d+)',result.stderr)
                    if match is None: raise RuntimeError('Normal path omitted actual work counters')
                    prefill,verifier=map(int,match.groups())
                    row.update(prefill_tokens=prefill,verifier_tokens=verifier,computed_tokens=prefill+verifier)
                if row['computed_tokens']!=length: raise RuntimeError('Full-history work mismatch')
                raw=logits.read_bytes()
                if len(raw)!=report['vocab']*4: raise RuntimeError('Incomplete numerical frontier')
                values=array.array('f');values.frombytes(raw)
                if any(not math.isfinite(x) for x in values): raise RuntimeError('Nonfinite logits')
                row.update(logits_sha256=sha(logits),bytes=len(raw))
                vectors[(operator,chunk)]=values
                if operator=='normal':
                    expected=vectors[('token',1)]
                    differences=[abs(x-y) for x,y in zip(values,expected)]
                    report['comparisons'].append(dict(length=length,chunk=chunk,exact_token_ids=ids,
                        compared_logits=len(values),max_absolute_error=max(differences),
                        worst_token_id=max(range(len(values)),key=differences.__getitem__),
                        top1_normal=max(range(len(values)),key=values.__getitem__),
                        top1_token=max(range(len(expected)),key=expected.__getitem__)))
                save()
        report['status']='MEASURED'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));save();raise
    save()
    print(json.dumps({'status':report['status'],'comparisons':report['comparisons']},indent=2))


if __name__=='__main__': main()
